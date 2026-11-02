"""Node B diagnostic: compare AR and target-only, fully accepted verification.

No serving files are edited; generated content is never executed. Known AR tokens
make every draft match, separating block forward numerics from rejection replay.
"""
import argparse
import gc
import sys
import hashlib
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fixtures', type=Path, required=True,
                    help='Local first_write prompt, prompt-manifest.json and saved AR reference directory')
parser.add_argument('--model', type=Path, required=True,
                    help='Pinned Ornith target model directory')
parser.add_argument('--output', type=Path, required=True,
                    help='New output directory; must not already exist')
parser.add_argument('--assert-greedy-parity', action='store_true',
                    help='Exit 1 after saving results if a verifier block changes any top token')
args = parser.parse_args()

import torch
from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import ArgmaxSampler
from exllamav3.modules.gated_delta_net import GDNState

os.umask(0o077)
os.environ.pop('EXL3_DRAFT_TOKENS', None)
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
records = []
model.load()
print('TARGET_LOADED', flush=True)


def run(name, width=None, force=False):
    gen = Generator(model, cache, tok, num_draft_tokens=0,
                    max_batch_size=1, max_chunk_size=2048, max_q_size=16)
    job = Job(input_ids=input_ids, max_new_tokens=32, min_new_tokens=1,
              sampler=ArgmaxSampler(), seed=123, stop_conditions=[],
              decode_special_tokens=True, return_logits=True, repetition_guard=True)
    gen.enqueue(job)
    if force:
        job.constrain_output_now(torch.tensor([known_ids[:32]], dtype=torch.long))
    original_iterate_gen = gen.iterate_gen
    original_rewind = GDNState.rewind
    calls, rewinds = [], []

    def inject(results, draft_tokens=None):
        active = [j for j in gen.active_jobs if j.is_prefill_done()]
        if width is not None and active:
            assert len(active) == 1
            pos = active[0].new_tokens
            draft_tokens = torch.tensor([known_ids[pos:pos + width]], dtype=torch.long)
            assert draft_tokens.shape == (1, width)
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
    assert len(ids) == len(scores) == 32
    assert ids == known_ids[:32], (name, 'continuation changed')
    if width is not None:
        assert not any(x['rejected'] > 0 for x in rewinds), rewinds
    scores = torch.stack(scores)
    torch.save({'ids': ids, 'logits': scores}, out / (name + '-scores.pt'))
    rec = {'name': name, 'seconds': time.perf_counter() - start,
           'same_saved_ar_ids': True, 'calls': calls, 'rewinds': rewinds,
           'nan_count': int(scores.isnan().sum()),
           'positive_inf_count': int(scores.isposinf().sum())}
    records.append(rec)
    (out / 'runs.json').write_text(json.dumps(records, indent=2))
    print('DONE', name, 'seconds', round(rec['seconds'], 1), 'rejections',
          sum(x['rejected'] for x in rewinds), flush=True)
    del gen, job
    gc.collect()
    return scores


def compare(name, ref, candidate):
    allowed = ref.isfinite() & candidate.isfinite()
    a = ref.masked_fill(~allowed, -torch.inf)
    b = candidate.masked_fill(~allowed, -torch.inf)
    ia, ib = a.argmax(-1), b.argmax(-1)
    mismatches = (ia != ib).nonzero().flatten().tolist()
    deltas = (ref - candidate).masked_fill(~allowed, 0)
    result = {'name': name, 'mismatch_positions': mismatches,
              'first_block_mismatches': [p for p in mismatches if p < 16],
              'max_logit_abs_delta': float(deltas.abs().max()),
              'mean_logit_abs_delta': float(deltas.abs().sum() / allowed.sum()),
              'position_17': {'ar_id': int(ia[17]), 'candidate_id': int(ib[17]),
                              'ar_gap_refer_over_reference': float(a[17, 8042] - a[17, 5604]),
                              'candidate_gap_refer_over_reference': float(b[17, 8042] - b[17, 5604])}}
    print('COMPARISON', json.dumps(result), flush=True)
    return result


with torch.inference_mode():
    ar = run('ar32')
    block16 = run('oracle_block16', width=15, force=True)
    block2 = run('oracle_block2', width=1, force=True)
    repeat = run('ar32_repeat')
    comparisons = [compare('oracle_block16', ar, block16),
                   compare('oracle_block2', ar, block2),
                   compare('ar32_repeat', ar, repeat)]
    (out / 'comparisons.json').write_text(json.dumps(comparisons, indent=2))
print('ISOLATION_COMPLETE', flush=True)

if args.assert_greedy_parity and any(r['mismatch_positions'] for r in comparisons[:2]):
    print('GREEDY_PARITY_FAIL: inspect comparisons.json; baseline has position 17 mismatch', flush=True)
    sys.exit(1)
