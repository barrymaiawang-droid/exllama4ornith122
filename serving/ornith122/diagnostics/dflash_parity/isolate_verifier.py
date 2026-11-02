"""Node B diagnostic: compare AR and target-only, fully accepted verification.

No serving files are edited; generated content is never executed. Known AR tokens
make every draft match, separating block forward numerics from rejection replay.

Each block width W runs one cold generator whose every verification pass is W
proposals plus the current input token, i.e. W + 1 rows. The default widths
reproduce the 2 November runs (15 -> 16 rows, 1 -> 2 rows); `--widths 1,3,7,8,15`
sweeps the row-count boundary (8 rows is the widest pass on the RDNA tiers whose
per-row arithmetic is AR decoding's; 9 rows leaves them, see doc/env_vars.md).
"""
import argparse
import gc
import sys
import hashlib
import json
import os
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity_compare import compare, dump_json, parse_widths, provenance, summarize

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--fixtures', type=Path, required=True,
                    help='Local first_write prompt, prompt-manifest.json and saved AR reference directory')
parser.add_argument('--model', type=Path, required=True,
                    help='Pinned Ornith target model directory')
parser.add_argument('--output', type=Path, required=True,
                    help='New output directory; must not already exist')
parser.add_argument('--widths', default='15,1',
                    help='Comma-separated proposals per block run, each 1..16 (rows = width + 1); '
                         'default 15,1 reproduces the published oracle_block16 / oracle_block2 runs')
parser.add_argument('--new-tokens', type=int, default=32,
                    help='Positions generated and compared per run (needs that many saved AR ids)')
parser.add_argument('--no-repeat', action='store_true',
                    help='Skip the closing AR repeat run (the bitwise-reproducibility control)')
parser.add_argument('--assert-greedy-parity', action='store_true',
                    help='Exit 1 after saving results if any verifier block changes any top token')
args = parser.parse_args()
widths = parse_widths(args.widths)
N = args.new_tokens
assert 1 <= N <= 128, "--new-tokens must be 1..128"

import torch
from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import ArgmaxSampler
from exllamav3.modules.gated_delta_net import GDNState

os.umask(0o077)
os.environ.pop('EXL3_DRAFT_TOKENS', None)
os.environ.pop('EXL3_DRAFT_MAX_ROWS', None)
root = args.fixtures.resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
config = Config.from_directory(str(args.model.resolve()))
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=262144, max_history=15,
              max_batch_size=1, recurrent_dtype=torch.float32)
tok = Tokenizer.from_config(config)
prompt = (root / 'first_write-prompt.txt').read_text()
manifest = next(x for x in json.loads((root / 'prompt-manifest.json').read_text())
                if x['label'] == 'first_write')
assert hashlib.sha256(prompt.encode()).hexdigest() == manifest['prompt_sha256']
input_ids = tok.encode(prompt, encode_special_tokens=True)
assert hashlib.sha256(input_ids.numpy().tobytes()).hexdigest() == manifest['token_ids_sha256']
known_ids = json.loads((root / 'ar-first_write-repeat128.json').read_text())['ids']
assert len(known_ids) >= N, f"saved AR reference holds {len(known_ids)} ids, need {N}"
records = []
model.load()
print('TARGET_LOADED', flush=True)
prov = provenance()
dump_json(out / 'provenance.json', prov)
print('PROVENANCE', json.dumps({k: prov[k] for k in ('source_git_head', 'extension_sha256', 'flags')}), flush=True)


def run(name, width=None, force=False):
    gen = Generator(model, cache, tok, num_draft_tokens=0,
                    max_batch_size=1, max_chunk_size=2048, max_q_size=16)
    job = Job(input_ids=input_ids, max_new_tokens=N, min_new_tokens=1,
              sampler=ArgmaxSampler(), seed=123, stop_conditions=[],
              decode_special_tokens=True, return_logits=True, repetition_guard=True)
    gen.enqueue(job)
    if force:
        job.constrain_output_now(torch.tensor([known_ids[:N]], dtype=torch.long))
    original_iterate_gen = gen.iterate_gen
    original_rewind = GDNState.rewind
    calls, rewinds = [], []

    def inject(results, draft_tokens=None):
        active = [j for j in gen.active_jobs if j.is_prefill_done()]
        if width is not None and active:
            assert len(active) == 1
            pos = active[0].new_tokens
            # The last block may run short of saved ids; verification then takes a narrower pass
            draft_tokens = torch.tensor([known_ids[pos:pos + width]], dtype=torch.long)
            assert draft_tokens.shape[0] == 1 and 1 <= draft_tokens.shape[1] <= width
        before = job.new_tokens
        original_iterate_gen(results, draft_tokens)
        if active:
            calls.append({'before': before, 'after': job.new_tokens,
                          'draft_width': None if draft_tokens is None else draft_tokens.shape[-1]})

    def record_rewind(state, n):
        rewinds.append({'rejected': n, 'history': state.last_history,
                        'position': state.position})
        return original_rewind(state, n)

    gen.iterate_gen = inject
    GDNState.rewind = record_rewind
    ids, scores = [], []
    start, last_progress = time.perf_counter(), time.perf_counter()
    print('START', name, flush=True)
    try:
        while gen.num_remaining_jobs():
            for e in gen.iterate():
                if e.get('stage') == 'error':
                    raise e['error']
                if e.get('stage') == 'streaming':
                    if e.get('token_ids') is not None:
                        ids.extend(e['token_ids'].reshape(-1).tolist())
                    if e.get('logits') is not None:
                        scores.extend(e['logits'].reshape(-1, e['logits'].shape[-1]).float().cpu().unbind())
            if time.perf_counter() - last_progress > 10:
                print('PROGRESS', name, 'seconds', round(time.perf_counter() - start, 1),
                      'tokens', len(ids), flush=True)
                last_progress = time.perf_counter()
    finally:
        GDNState.rewind = original_rewind
    assert len(ids) == len(scores) == N
    assert ids == known_ids[:N], (name, 'continuation changed')
    if width is not None:
        assert not any(x['rejected'] > 0 for x in rewinds), rewinds
    scores = torch.stack(scores)
    torch.save({'ids': ids, 'logits': scores}, out / (name + '-scores.pt'))
    rec = {'name': name, 'width': width, 'rows': None if width is None else width + 1,
           'seconds': time.perf_counter() - start,
           'same_saved_ar_ids': True, 'calls': calls, 'rewinds': rewinds,
           'nan_count': int(scores.isnan().sum()),
           'positive_inf_count': int(scores.isposinf().sum())}
    records.append(rec)
    dump_json(out / 'runs.json', records)
    print('DONE', name, 'seconds', round(rec['seconds'], 1), 'rejections',
          sum(x['rejected'] for x in rewinds), flush=True)
    del gen, job
    gc.collect()
    return scores


# Position 17 of the first_write fixture: AR keeps 8042 (" refer"), the 16-row pass takes 5604 (" reference")
PROBE = (17, 8042, 5604)

with torch.inference_mode():
    ar = run(f'ar{N}')
    comparisons = []
    for w in widths:
        name = f'oracle_block{w + 1}'
        block = run(name, width=w, force=True)
        c = compare(name, ar, block, probe=PROBE)
        c['width'] = w
        c['rows'] = w + 1
        comparisons.append(c)
        print('COMPARISON', json.dumps(summarize([c])[0]), flush=True)
        dump_json(out / 'comparisons.json', comparisons)
    if not args.no_repeat:
        repeat = run(f'ar{N}_repeat')
        c = compare(f'ar{N}_repeat', ar, repeat, probe=PROBE)
        comparisons.append(c)
        print('COMPARISON', json.dumps(summarize([c])[0]), flush=True)
        dump_json(out / 'comparisons.json', comparisons)
    dump_json(out / 'sweep.json', {'new_tokens': N, 'widths': widths, 'table': summarize(comparisons),
                                   'provenance': prov})
print('ISOLATION_COMPLETE', flush=True)

block_fail = [c['name'] for c in comparisons if c.get('width') is not None and c['mismatch_positions']]
if args.assert_greedy_parity and block_fail:
    print('GREEDY_PARITY_FAIL: top-token mismatch in', block_fail, '(see comparisons.json)', flush=True)
    sys.exit(1)
