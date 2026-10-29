# Repetition guard

Opt-in, per request. Detects a job copying its own output and steers or stops it without the client
noticing, in tiers. Off by default: unguarded output is byte-identical to before.

```python
job = Job(input_ids, max_new_tokens = 512, repetition_guard = True)          # defaults
job = Job(input_ids, repetition_guard = {"mode": "shadow"})                  # detect and log only
job = Job(input_ids, repetition_guard = {"min_match": 8, "confirm": 32, "max_interventions": 2,
                                         "escalate_within": 64, "nudge": "..."})
generator.generate(prompt, repetition_guard = True)
```

| tier | trigger | action | visible to the client |
|---|---|---|---|
| 0 detect | the last `min_match` tokens equal an earlier span of this turn's own output | hold output from this token on; release it untouched the moment the copy breaks | a short delay while the copy extends |
| 1 escape | the copy reaches `confirm` tokens, or one full period if shorter | roll back to the first held token, ban that token for one step | no |
| 2 nudge | a loop again within `escalate_within` tokens of tier 1 | roll back again, inject `nudge` as hidden tokens the model continues from | no: the text, token ids and `full_completion` never contain it |
| 3 stop | a loop after the nudge, or `max_interventions` rollbacks spent | end the job | `eos_reason: "repetition"` |

Nothing already streamed is ever retracted: holding starts at the token that completed the match, so at
most `min_match - 1` tokens of a copy are out before the hold, and a rollback discards held tokens only.
The detector compares token ids against the turn's own output, so quoting the prompt or a tool result
cannot trip it. The holdback adapts on its own, from nothing while the output is fresh to the length of
the copy in progress; there is no window to size.

Inside a constrained span (the job has filters: a tool call, JSON mode) tier 2 is skipped, since text must
not be injected into structured output, and the guard goes from tier 1 to tier 3. While a banned-string
match holds output, the guard defers to it.

## What the client sees

Stream events of an intervention carry `repetition_guard_event`: `tier`, `position` (output token
index), `period`, `matched`, `copy_start`, `source_start`, `rollback_tokens`, `injected_tokens`, with the
discarded span as `suppressed_text` / `suppressed_tokens` like a banned-string rewind. Each hidden nudge
token is an event with `repetition_guard_event: {"hidden_token_id": id}` and no text. The final result
carries `repetition_guard`: `mode`, `detections`, `interventions` (the list above), `tier` (the highest
reached), `rollback_tokens`, `injected_tokens`, `visible_new_tokens`. `new_tokens` still counts every
token the model produced, including hidden ones; a serve layer reports `visible_new_tokens` as
`completion_tokens` and the hidden counts in its timings.

Every intervention is logged on `exllamav3.generator.repetition_guard` at INFO (tier, position, period,
copy length, rollback and injection counts) and the discarded text at DEBUG. Shadow mode logs the same
with tier 0 and changes nothing.

## Mechanics

The hold and the rollback are the banned-string machinery (`Job.receive_sample`): the checkpoint holds
output buffers, a rewind truncates the sequence and the K/V cache, rolls the recurrent state back in
place or restores a page-aligned stash and replays, and leaves the first held token banned for the next
sample. The checkpoint records its owner so the two holds do not release each other. The nudge rides the
`constrain_output_now()` forced-token path with a hidden count: hidden tokens are sampled and appended to
the sequence, skipped by the output buffers, and still fed to the detector so output positions stay
aligned. A rollback inside a speculative window abandons the window exactly as a banned-string rewind
does, so drafting on or off gives the same internal token stream.

## Rolling it out

1. **Shadow first.** Run `{"mode": "shadow"}` on real traffic for a while. The INFO log gives the period
   and copy length of every loop, and which content it occurred in. That is the data for the two numbers
   that matter: `min_match` (the trigger; 8 by default) and `confirm` (the copy length that counts as a
   loop; 32 by default, so loops with a period under 32 confirm after one period and longer ones after 32
   copied tokens).
2. **Then active**, per request, with the goldens unchanged for every request that does not opt in.
3. After a nudge the prefix cache cannot be hit on the client's next turn from that point on (their
   history has no nudge); correctness is unaffected, the tail is re-prefilled. The pages after a nudge are
   still hashed; skipping that is a possible follow-up.

## Running the tests

```
# no GPU, no model: detector, tiers, and the Job's sampling path on a stub generator (15 tests)
pytest tests/generator/test_repetition_guard.py

# on the engine, model roles dense-llama and recurrent from tests/models.yaml: the gates above, guard off
# unchanged, spec + guard == plain + guard, rollback on a recurrent model
pytest tests/generator/test_repetition_guard_model.py --model-root <root>

# the existing generator and sampling suites must be unchanged (the guard is off unless requested)
pytest tests/generator tests/e2e/test_speculative.py --model-root <root>
```

No native code changed, so no rebuild is needed for this feature. The engine-side tests were written on a
host without a GPU and have not been run against a model yet; run them before relying on the guard.
