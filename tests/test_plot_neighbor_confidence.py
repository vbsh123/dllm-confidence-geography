"""Tiny synthetic traces only; no downloaded experiment data or model inference."""
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from confidence_geography.plot_neighbor_confidence import generate_neighbor_confidence_plots


class NeighborHistogramTests(unittest.TestCase):
    def fixture(self, filled, previous, seeds, end=20):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root/'input'/'samples'/'s'
            sample.mkdir(parents=True)
            result = {'sample_id': 's', 'answer_token_length': end,
                      'token_dictionary': {'1': {'special': False}, '2': {'special': False}}}
            (sample/'result.json').write_text(json.dumps(result))
            def row(p, after=False):
                return {'position': p, 'token_id': 2 if after else 1,
                        'text': 'new prediction' if after else 'old prediction',
                        'special': False, 'eligible': True,
                        'confidence': .9 if after else .4,
                        'previous_confidence': .4,
                        'committed': not after and p in seeds}
            records = [
                {'type': 'header', 'config': {'mask_id': 999, 'policy': 'threshold'}},
                {'type': 'step', 'step': 4, 'previous_commits': list(previous),
                 'commit_positions': sorted(seeds),
                 'state_ids': [1 if p in filled else 999 for p in range(20)],
                 'positions': [row(p) for p in range(20) if p not in filled]},
                {'type': 'step', 'step': 5, 'previous_commits': sorted(seeds),
                 'commit_positions': [],
                 'state_ids': [1 if p in filled | seeds else 999 for p in range(20)],
                 'positions': [row(p, True) for p in range(20) if p not in filled | seeds]},
            ]
            with gzip.open(sample/'trace.jsonl.gz', 'wt') as f:
                f.write('\n'.join(map(json.dumps, records))+'\n')
            out = root/'plots'
            summary = generate_neighbor_confidence_plots(root/'input', out, bins=10)
            with gzip.open(out/'paired_neighbors.csv.gz', 'rt') as f:
                pairs = list(csv.DictReader(f))
            with gzip.open(out/'selected_reveals.jsonl.gz', 'rt') as f:
                reveals = [json.loads(line) for line in f]
            for name in ['top1p_after', 'top1p_delta']:
                self.assertTrue((out/(name+'.png')).is_file())
                self.assertTrue((out/(name+'.pdf')).is_file())
                with (out/(name+'.csv')).open() as f:
                    bins = list(csv.DictReader(f))
                self.assertEqual(sum(int(b['count']) for b in bins), len(pairs))
                if pairs:
                    self.assertAlmostEqual(sum(float(b['percent']) for b in bins), 100)
            return summary, pairs, reveals

    def test_region_extension_excluded_even_if_far_from_latest_token(self):
        summary, pairs, _ = self.fixture({0, 1, 2, 3}, {1}, {4})
        self.assertEqual(summary['qualifying_seeds_with_next_pass'], 0)
        self.assertEqual(pairs, [])

    def test_return_beside_older_region_is_included(self):
        summary, pairs, reveals = self.fixture({0, 1, 10}, {1}, {9})
        self.assertEqual(summary['qualifying_seeds_with_next_pass'], 1)
        self.assertEqual([int(p['position']) for p in pairs], [8])
        self.assertEqual(reveals[0]['current_regions'], [[0, 1]])
        self.assertEqual(reveals[0]['seed_positions'], [9])

    def test_one_mask_gap_is_enough_no_distance_four_cutoff(self):
        summary, pairs, _ = self.fixture({0, 1, 2}, {1}, {4})
        self.assertEqual(summary['qualifying_seeds_with_next_pass'], 1)
        self.assertEqual([int(p['position']) for p in pairs], [3, 5])
        self.assertAlmostEqual(summary['mean_after'], .9)
        self.assertAlmostEqual(summary['mean_delta'], .5)
        self.assertTrue(all(p['prediction_changed'] == 'True' for p in pairs))

    def test_no_previous_current_region_is_excluded(self):
        summary, pairs, _ = self.fixture({0}, set(), {5})
        self.assertEqual(summary['qualifying_seeds_with_next_pass'], 0)
        self.assertEqual(pairs, [])

    def test_all_previous_batch_regions_are_checked(self):
        summary, pairs, _ = self.fixture({0, 1, 8, 9}, {1, 8}, {7})
        self.assertEqual(summary['qualifying_seeds_with_next_pass'], 0)
        self.assertEqual(pairs, [])

    def test_shared_neighbor_counted_once(self):
        _, pairs, _ = self.fixture({0, 1}, {1}, {5, 7})
        self.assertEqual([int(p['position']) for p in pairs], [4, 6, 8])
        self.assertEqual(json.loads(pairs[1]['seed_positions']), [5, 7])

    def test_cocommitted_neighbors_excluded(self):
        _, pairs, _ = self.fixture({0, 1}, {1}, {5, 6})
        self.assertEqual([int(p['position']) for p in pairs], [4, 7])

    def test_post_stop_reveal_excluded(self):
        summary, pairs, _ = self.fixture({0, 1}, {1}, {18}, end=17)
        self.assertEqual(summary['qualifying_seeds_with_next_pass'], 0)
        self.assertEqual(pairs, [])


if __name__ == '__main__':
    unittest.main()
