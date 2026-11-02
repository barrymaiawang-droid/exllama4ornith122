from types import SimpleNamespace
import pytest
import torch
from exllamav3.generator.job import Job


@pytest.mark.nogpu
@pytest.mark.parametrize('healing', [False, True])
def test_target_prefill_uses_ar_reservation_at_lookahead_page_boundary(healing):
    class StopForward(Exception):
        pass
    class IDs:
        def __len__(self): return 270
        def torch_slice(self, start, end): return torch.zeros((1, end-start), dtype=torch.long)
    class Model:
        caps = {}
        def prefill(self, *, input_ids, params):
            assert input_ids.shape[-1] == 13
            raise StopForward(params['block_table'].clone())
    observed = []
    # Choose budget so AR total including healing lands exactly on a page boundary.
    budget = 241 - int(healing)
    for draft in (False, True):
        pages = 3 if draft else 2
        seq = SimpleNamespace(prefill_complete=False, kv_position=256, sequence_ids=IDs(),
                              allocated_pages=[SimpleNamespace(kv_position=0) for _ in range(pages)],
                              block_index_tensor=torch.arange(pages, dtype=torch.int32).view(1, -1))
        model = Model()
        generator = SimpleNamespace(max_chunk_size=2048, recurrent_cache=object(), model=model, cache=object(),
            dflash_draft=draft, mtp_draft=False,
            draft_model=SimpleNamespace(draft_verifier_params={}) if draft else None)
        job = Job.__new__(Job)
        job.time_first_prefill = None
        job.sequences = [seq]
        job.recurrent_state = SimpleNamespace(position=-1, target=object())
        job.generator = generator
        job.embeddings = None
        job.alt_rope_freqs = None
        job.orig_max_rq_tokens = None
        job.max_new_tokens = budget
        job.prefix_token = torch.tensor([1]) if healing else None
        with pytest.raises(StopForward) as caught:
            job.prefill([])
        observed.append(caught.value.args[0])
    assert torch.equal(observed[0], observed[1])
    assert observed[1].tolist() == [[0, 1]]
