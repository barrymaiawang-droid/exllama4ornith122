"""CPU checks of the parity runner's comparison and width parsing (no model, no GPU).

The comparison must find exactly the flips a verifier block introduces, report the AR
near-tie gap at each flip, treat padded (non-finite) vocabulary entries as absent, and
be exact (zero KL, bitwise flag) on identical scores. Skipped without torch.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'diagnostics', 'dflash_parity'))

try:
    import torch
except ImportError:  # the serving unit tests otherwise need no torch
    torch = None


@unittest.skipUnless(torch is not None, 'torch not installed')
class ParityCompare(unittest.TestCase):
    def setUp(self):
        from parity_compare import compare, parse_widths, summarize
        self.compare, self.parse_widths, self.summarize = compare, parse_widths, summarize
        g = torch.Generator().manual_seed(123)
        self.ref = torch.randn((32, 256), generator=g) * 4

    def test_identical_scores(self):
        c = self.compare('same', self.ref, self.ref.clone())
        self.assertEqual(c['mismatch_positions'], [])
        self.assertIsNone(c['first_mismatch'])
        self.assertTrue(c['bitwise_equal'])
        self.assertEqual(c['max_logit_abs_delta'], 0.0)
        self.assertEqual(c['mean_kl'], 0.0)
        self.assertEqual(c['max_kl'], 0.0)
        self.assertEqual(len(c['per_position_kl']), 32)

    def test_single_flip_is_located_with_its_gap(self):
        cand = self.ref.clone()
        top2 = self.ref[17].topk(2)
        a, b = int(top2.indices[0]), int(top2.indices[1])
        gap = float(top2.values[0] - top2.values[1])
        cand[17, b] = self.ref[17, a] + 0.5          # the runner-up now wins at 17 only
        c = self.compare('flip', self.ref, cand, probe=(17, a, b))
        self.assertEqual(c['mismatch_positions'], [17])
        self.assertEqual(c['first_mismatch'], 17)
        self.assertEqual(c['first_block_mismatches'], [])
        self.assertFalse(c['bitwise_equal'])
        d = c['mismatch_detail'][0]
        self.assertEqual((d['position'], d['ref_id'], d['candidate_id']), (17, a, b))
        self.assertAlmostEqual(d['ref_top2_gap'], gap, places=5)
        self.assertAlmostEqual(d['candidate_gap_ref_over_candidate'], -0.5, places=5)
        self.assertAlmostEqual(c['probe']['ref_gap_a_over_b'], gap, places=5)
        self.assertAlmostEqual(c['probe']['candidate_gap_a_over_b'], -0.5, places=5)
        # Only position 17 carries any delta or KL
        nz = [i for i, x in enumerate(c['per_position_max_abs_delta']) if x > 0]
        self.assertEqual(nz, [17])
        self.assertGreater(c['per_position_kl'][17], 0)
        self.assertEqual(c['per_position_kl'][16], 0.0)
        self.assertEqual(c['max_kl'], c['per_position_kl'][17])

    def test_first_block_flip_is_flagged(self):
        cand = self.ref.clone()
        cand[3] = self.ref[3].flip(0) * 2 + 7
        c = self.compare('early', self.ref, cand)
        self.assertIn(3, c['mismatch_positions'])
        self.assertEqual(c['first_block_mismatches'], c['mismatch_positions'])

    def test_padded_entries_are_ignored(self):
        ref = self.ref.clone()
        cand = self.ref.clone()
        ref[:, 250:] = -torch.inf                      # the free run masks padded columns
        cand[:, 250:] = 1e4                            # the forced run leaves them unmasked: never the argmax
        c = self.compare('pad', ref, cand)
        self.assertEqual(c['mismatch_positions'], [])
        self.assertEqual(c['max_logit_abs_delta'], 0.0)
        self.assertEqual(c['mean_kl'], 0.0)
        self.assertFalse(c['bitwise_equal'])

    def test_additive_shift_has_no_kl(self):
        c = self.compare('shift', self.ref, self.ref + 3.0)
        self.assertEqual(c['mismatch_positions'], [])
        self.assertAlmostEqual(c['max_logit_abs_delta'], 3.0, places=5)
        self.assertLess(c['max_kl'], 1e-9)

    def test_summary_rows(self):
        rows = self.summarize([self.compare('x', self.ref, self.ref)])
        self.assertEqual(rows[0]['name'], 'x')
        self.assertEqual(set(rows[0]), {'name', 'mismatch_positions', 'first_mismatch', 'max_logit_abs_delta',
                                        'mean_logit_abs_delta', 'mean_kl', 'max_kl', 'bitwise_equal'})

    def test_parse_widths(self):
        self.assertEqual(self.parse_widths('15,1'), [15, 1])
        self.assertEqual(self.parse_widths(' 1, 3 ,7,8,15 '), [1, 3, 7, 8, 15])
        for bad in ('0', '17', '3,3', '', 'x'):
            with self.assertRaises(ValueError):
                self.parse_widths(bad)


if __name__ == '__main__':
    unittest.main()
