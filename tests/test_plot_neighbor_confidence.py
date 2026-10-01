import gzip
import json
from pathlib import Path
import tempfile
import unittest

from confidence_geography.plot_neighbor_confidence import (
    calculate_neighbor_distributions, histogram, trace_pairs,
)


class NeighborHistogramTests(unittest.TestCase):
    def test_requested_distributions_recompute_delta_without_mutating_inputs(self):
        pair = {'before': .25, 'after': .75, 'delta': 999}
        result = calculate_neighbor_distributions([pair], bins=4)
        self.assertEqual(result['after_probabilities'], [.75])
        self.assertEqual(result['probability_deltas'], [.5])
        self.assertEqual(result['pairs'][0]['delta'], .5)
        self.assertEqual(pair['delta'], 999)
        self.assertEqual(sum(r['count'] for r in result['histograms']['top1p_after']), 1)

    def test_distance_filter_selects_seed_distance_not_neighbor_distance(self):
        pairs = [{'before': .8, 'after': .6, 'max_adjacent_seed_distance': 1},
                 {'before': .25, 'after': .75, 'max_adjacent_seed_distance': 4}]
        result = calculate_neighbor_distributions(pairs, min_seed_distance=4)
        self.assertEqual(result['probability_deltas'], [.5])

    def test_histogram_includes_exact_zero_and_one(self):
        rows = histogram([0, .5, 1, 1], 0, 1, 2)
        self.assertEqual([r['count'] for r in rows], [1, 3])
        self.assertEqual(sum(r['percent'] for r in rows), 100)

    def test_signed_delta_keeps_negative_values(self):
        rows = histogram([-1, -.2, 0, .2, 1], -1, 1, 4)
        self.assertEqual([r['count'] for r in rows], [1, 1, 2, 1])
        with self.assertRaises(ValueError):
            histogram([1.1], 0, 1, 10)

    def collect_fixture(self, seeds):
        def row(p, after=False):
            return {'position': p, 'token_id': 2 if after else 1,
                    'text': 'new' if after else 'old', 'special': False,
                    'eligible': True, 'confidence': .9 if after else .8,
                    'previous_confidence': .8, 'committed': not after and p in seeds}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'samples'/'s'
            folder.mkdir(parents=True)
            result = {'sample_id': 's', 'answer_token_length': 8}
            (folder/'result.json').write_text(json.dumps(result))
            records = [
                {'type': 'header', 'config': {'mask_id': 999, 'policy': 'threshold'}},
                {'type': 'step', 'step': 0, 'state_ids': [1]+[999]*7,
                 'positions': [row(p) for p in range(1, 8)]},
                {'type': 'step', 'step': 1,
                 'state_ids': [1 if p == 0 or p in seeds else 999 for p in range(8)],
                 'positions': [row(p, True) for p in range(1, 8) if p not in seeds]},
            ]
            with gzip.open(folder/'trace.jsonl.gz', 'wt') as f:
                f.write('\n'.join(map(json.dumps, records))+'\n')
            return list(trace_pairs(Path(tmp)))

    def test_shared_neighbor_is_counted_once(self):
        pairs = self.collect_fixture({4, 6})
        self.assertEqual([r['position'] for r in pairs], [3, 5, 7])
        self.assertEqual(pairs[1]['max_adjacent_seed_distance'], 6)
        self.assertTrue(all(r['changed'] for r in pairs))
        self.assertAlmostEqual(pairs[0]['delta'], .1)

    def test_cocommitted_neighbors_excluded(self):
        pairs = self.collect_fixture({4, 5})
        self.assertEqual([r['position'] for r in pairs], [3, 6])


if __name__ == '__main__':
    unittest.main()
