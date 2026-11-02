"""Exercise Ornith's first MoE with identical token inputs at decode widths 1..8.

Loads only one MoE, not the full model. Requires an RDNA extension with route
tracing enabled and model weights supplied via --model. Uses synthetic inputs;
no prompt or private fixture is required. Run with the deployment's Python path.
"""
import argparse
import json
from pathlib import Path

import torch
from exllamav3 import Config, Model
from exllamav3.util import route_trace

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--device', default='cuda:0')
p.add_argument('--seed', type=int, default=123)
p.add_argument('--output', type=Path, required=True, help='New JSON report path')
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
if not route_trace.available():
    raise RuntimeError('Native extension lacks route tracing; use an extension built with route_trace support')
config = Config.from_directory(str(a.model.resolve()))
model = Model.from_config(config)
block = model.modules[1]
moe = block.mlp
if type(moe).__name__ != 'BlockSparseMLP' or block.layer_idx != 0:
    raise RuntimeError('Expected an Ornith layer-0 BlockSparseMLP')
moe.load(torch.device(a.device))
# CPU-generated values avoid depending on the device RNG implementation.
generator = torch.Generator(device='cpu').manual_seed(a.seed)
rows = torch.randn((1, 8, moe.hidden_size), generator=generator, dtype=torch.float32).half()
report = {'seed': a.seed, 'device': a.device, 'model': str(a.model), 'runs': [], 'errors': []}

with torch.inference_mode():
    for mode in ('distinct', 'repeated'):
        baseline = None
        for n in range(1, 9):
            x = (rows[:, :n] if mode == 'distinct' else rows[:, :1].expand(1, n, -1).contiguous()).to(a.device)
            original = moe.routing_fn
            calls = []
            def route(bsz, cfg, y, params):
                selected, weights = original(bsz, cfg, y, params)
                calls.append((selected.detach().cpu().clone(), weights.detach().cpu().clone()))
                return selected, weights
            moe.routing_fn = route
            try:
                with route_trace.capture() as trace:
                    out = moe.forward(x, {}).detach().cpu().clone()
                # Exercise warmup and replay as well as the initial uncaptured call.
                repeats = [moe.forward(x, {}).detach().cpu().clone() for _ in range(3)]
            finally:
                moe.routing_fn = original
            if len(calls) != 4:
                raise RuntimeError(f'Expected one routing call per forward, got {len(calls)}')
            current = (out[:, 0], calls[0][0][0], calls[0][1][0])
            if baseline is None:
                baseline = current
            diffs = (current[0].float() - baseline[0].float()).abs()
            routes = [r.route_name for r in trace.records]
            entry = {'mode': mode, 'rows': n, 'output_equal': bool(torch.equal(current[0], baseline[0])),
                     'output_max_abs': float(diffs.max()),
                     'selected_equal': bool(torch.equal(current[1], baseline[1])),
                     'weights_equal': bool(torch.equal(current[2], baseline[2])),
                     'repeat_equal': all(torch.equal(out, v) for v in repeats),
                     'rdna_moe_decode_calls': routes.count('rdna_moe_decode'),
                     'trace_dropped': trace.dropped, 'route_table': trace.table()}
            report['runs'].append(entry)
            for key in ('output_equal', 'selected_equal', 'weights_equal', 'repeat_equal'):
                if not entry[key]:
                    report['errors'].append(f'{mode} rows={n}: {key} failed')
            if entry['rdna_moe_decode_calls'] != 1 or entry['trace_dropped']:
                report['errors'].append(f'{mode} rows={n}: expected one traced RDNA decode call with no dropped records')
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps({'report': str(a.output), 'errors': report['errors']}, indent=2))
if report['errors']:
    raise SystemExit(1)
