"""Synthetic-trace checks for questions 3.1/3.2. Requires Matplotlib, not a model."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from confidence_geography.plot_region_activity import generate_region_activity_plots


class RegionActivityTests(unittest.TestCase):
    def analyze(self, filled, previous, commits, policy='top1'):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root/'input'/'samples'/'s'
            sample.mkdir(parents=True)
            result = {'sample_id': 's', 'answer_token_length': 20,
                      'token_dictionary': {'1': {'special': False}, '2': {'special': False}}}
            (sample/'result.json').write_text(json.dumps(result))
            records = [
                {'type': 'header', 'config': {'mask_id': 999, 'policy': policy}},
                {'type': 'step', 'step': 4, 'previous_commits': sorted(previous),
                 'commit_positions': sorted(commits),
                 'state_ids': [1 if p in filled else 999 for p in range(20)],
                 'positions': [{'position': p, 'token_id': 2, 'special': False,
                                'committed': p in commits}
                               for p in range(20) if p not in filled]},
            ]
            with gzip.open(sample/'trace.jsonl.gz', 'wt') as f:
                f.write('\n'.join(map(json.dumps, records))+'\n')
            summary = generate_region_activity_plots(root/'input', root/'out', mask_gap=4)
            with gzip.open(root/'out'/'region_steps.jsonl.gz', 'rt') as f:
                steps = [json.loads(line) for line in f]
            return summary, steps[0]

    def test_three_masks_still_same_region_but_skipping_is_not_left_to_right(self):
        summary, step = self.analyze({0, 4}, {4}, {2})
        self.assertEqual(summary['top1_transitions'], {'stay_same_region': 1})
        self.assertEqual(summary['local_order'], {'skips_earlier_masks': 1})
        self.assertEqual(step['updates'][0]['required_masks_for_prefix'], [1, 2])

    def test_exactly_four_masks_separates_regions(self):
        summary, _ = self.analyze({0, 5}, {0}, {6})
        self.assertEqual(summary['top1_transitions'], {'jump_to_other_existing_region': 1})

    def test_new_region_is_separate_from_a_jump_to_old_text(self):
        summary, _ = self.analyze({0, 1}, {1}, {10})
        self.assertEqual(summary['top1_transitions'], {'open_new_region': 1})
        self.assertEqual(summary['denominators']['local_order_updates'], 0)

    def test_top1_merge_is_not_arbitrarily_stay_or_jump(self):
        summary, step = self.analyze({0, 8}, {0}, {4})
        self.assertEqual(summary['top1_transitions'], {'merge_regions': 1})
        self.assertEqual(step['old_regions_touched'], 2)
        self.assertEqual(summary['denominators']['local_order_updates'], 0)

    def test_threshold_can_update_two_regions_at_once(self):
        summary, _ = self.analyze({0, 1, 10, 11}, {1}, {2, 12}, 'threshold')
        self.assertEqual(summary['regions_receiving_commits_per_batch'], {2: 1})
        self.assertEqual(summary['commits_per_active_region'], {1: 2})

    def test_threshold_merged_region_counted_once_after_batch(self):
        summary, step = self.analyze({0, 8}, {0}, {3, 6}, 'threshold')
        self.assertEqual(summary['regions_receiving_commits_per_batch'], {1: 1})
        self.assertEqual(step['old_regions_touched'], 2)
        self.assertEqual(summary['totals']['merge_updates'], 1)

    def test_threshold_prefix_is_set_based_not_invented_order(self):
        summary, _ = self.analyze({0, 4}, {4}, {1, 2}, 'threshold')
        self.assertEqual(summary['local_order'], {'left_to_right_prefix': 1})
        summary, _ = self.analyze({0, 4}, {4}, {1, 3}, 'threshold')
        self.assertEqual(summary['local_order'], {'skips_earlier_masks': 1})

    def test_left_expansion_and_missing_previous_anchor(self):
        summary, _ = self.analyze({5, 8}, set(), {4})
        self.assertEqual(summary['local_order'], {'left_expansion': 1})
        self.assertEqual(summary['denominators']['top1_transitions'], 0)


if __name__ == '__main__':
    unittest.main()
