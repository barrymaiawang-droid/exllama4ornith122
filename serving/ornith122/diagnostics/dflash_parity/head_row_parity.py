#!/usr/bin/env python3
"""Compare batched Ornith head outputs with the same rows run individually.

Example: EXL3_HEAD_AR_PARITY=1 python head_row_parity.py --model /path/to/model --output /new/report.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='checkpoint directory with safetensors index')
    parser.add_argument('--output', type=Path, required=True, help='new JSON report path')
    args = parser.parse_args()
    # Reserve the report before any GPU work, and never overwrite prior evidence.
    with args.output.open('x') as report_file:
        report = {'model': str(args.model.resolve()), 'seed': 123, 'flag': os.getenv('EXL3_HEAD_AR_PARITY', '0'),
                  'cases': [], 'passed': False}
        try:
            import exllamav3_ext as ext
            from exllamav3.util import route_trace
            extension = Path(ext.__file__).resolve(strict=True)
            report.update(extension=str(extension), extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
                          torch_version=torch.__version__, hip_version=torch.version.hip,
                          environment={k: v for k, v in os.environ.items() if k.startswith(('EXL3_', 'HIP_', 'ROCR_', 'CUDA_'))})
            if torch.version.hip is None or not torch.cuda.is_available():
                raise RuntimeError('ROCm GPU required')
            index = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']
            names = ('lm_head.trellis', 'lm_head.suh', 'lm_head.svh')
            weights = {}
            for name in names:
                with safe_open(args.model / index[name], framework='pt', device='cpu') as f:
                    weights[name] = f.get_tensor(name).to('cuda')
            trellis, suh, svh = (weights[n] for n in names)
            if tuple(trellis.shape) != (192, 15520, 96):
                raise ValueError(f'unexpected head trellis shape: {tuple(trellis.shape)}')
            generator = torch.Generator(device='cpu').manual_seed(123)
            x = torch.randn((8, 3072), generator=generator, dtype=torch.float32).half().to('cuda').contiguous()
            original = x.clone()
            scratch = torch.empty_like(x)
            single = torch.empty((8, 248320), dtype=torch.float16, device='cuda')
            batch = torch.empty_like(single)
            trace_available = route_trace.available()
            report['route_trace_available'] = trace_available
            report['device'] = torch.cuda.get_device_name(0)
            def invoke(inp, out, temp):
                return ext.exl3_gemm(inp, trellis, out, suh, temp, svh, 0, False, True, 0)
            def run(fn):
                if trace_available:
                    with route_trace.capture() as trace:
                        route = fn()
                        torch.cuda.synchronize()
                    return route, [r._asdict() for r in trace.records], trace.dropped
                route = fn()
                torch.cuda.synchronize()
                return route, [], 0
            singles = [run(lambda i=i: invoke(x[i:i+1], single[i:i+1], scratch[i:i+1])) for i in range(8)]
            report['single_routes'] = [r for r, _, _ in singles]
            route_ok = all(r == 90 for r, _, _ in singles)
            if trace_available:
                route_ok &= all(records and not dropped and any(r.get('route') == 90 for r in records)
                                for _, records, dropped in singles)
            all_ok = bool(route_ok and single.dtype == torch.float16 and torch.isfinite(single).all() and torch.equal(x, original))
            for width in range(1, 9):
                route, records, dropped = run(lambda: invoke(x[:width], batch[:width], scratch[:width]))
                expected_route = 90 if width == 1 else 92
                valid = bool(route == expected_route and batch.dtype == torch.float16 and
                             torch.isfinite(batch[:width]).all() and torch.equal(x, original))
                if trace_available:
                    valid &= bool(records and not dropped and any(r.get('route') == expected_route for r in records))
                unequal = batch[:width] != single[:width]
                count = int(unequal.sum())
                item = {'rows': width, 'route': route, 'expected_route': expected_route,
                        'trace': records, 'trace_dropped': dropped, 'unequal': count,
                        'unequal_by_row': unequal.sum(dim=1).cpu().tolist(),
                        'max_abs': float((batch[:width].float() - single[:width].float()).abs().max()),
                        'single_sha256': hashlib.sha256(single[:width].cpu().numpy().tobytes()).hexdigest(),
                        'batch_sha256': hashlib.sha256(batch[:width].cpu().numpy().tobytes()).hexdigest(),
                        'valid': valid}
                report['cases'].append(item)
                all_ok &= valid and count == 0
            report['passed'] = bool(all_ok)
        except Exception as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
        json.dump(report, report_file, indent=2)
        report_file.write('\n')
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {args.output}")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
