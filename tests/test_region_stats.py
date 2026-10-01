"""Offline statistics tests; these do not import torch or execute a model."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from confidence_geography.region_stats import (
    Analyzer, Rates, Source, components, continuation, neighbor_pair,
    snapshot, top1_destination,
)


def make_state(filled, commits, previous=(), end=20, policy='threshold', probabilities=None):
    probabilities = probabilities or {}
    config = {'mask_id': 999, 'policy': policy, 'commit_threshold': .9}
    result = {'sample_id': 'sample', 'answer_token_length': end,
              'token_dictionary': {'1': {'special': False}, '2': {'special': True}}}
    rows = [{'position': p, 'token_id': 1, 'text': 'word', 'special': False,
             'confidence': probabilities.get(p, .95), 'eligible': True,
             'committed': p in commits} for p in range(20) if p not in filled]
    step = {'type': 'step', 'step': 0, 'positions': rows,
            'commit_positions': list(commits), 'previous_commits': list(previous),
            'state_ids': [1 if p in filled else 999 for p in range(20)]}
    return snapshot(step, result, config, 4), result, config


class RegionStatsTests(unittest.TestCase):
    def test_components_and_mask_gap(self):
        self.assertEqual(components([4, 2, 3, 2, 8]), [[2, 3, 4], [8]])
        state, _, _ = make_state({0}, {4}, [0])
        self.assertEqual(state['distance'](4), 4)  # Three masks, not four.
        self.assertEqual(state['regions'], [[4]])

    def test_simultaneous_bridge_is_not_an_isolated_region(self):
        state, _, _ = make_state({0}, {1, 2, 3, 4, 5}, [0])
        self.assertEqual(state['seeds'], {4, 5})
        self.assertEqual(state['regions'], [])

    def test_threshold_adjacent_seeds_count_as_one_region(self):
        state, _, _ = make_state({0}, {4, 5, 9}, [0])
        self.assertEqual(state['regions'], [[4, 5], [9]])

    def test_post_stop_cocommit_does_not_erase_valid_region(self):
        state, _, _ = make_state({0}, {16, 17, 18, 19}, [0], end=18)
        self.assertEqual(state['regions'], [[16, 17]])

    def test_next_step_categories_and_two_sided_continuation(self):
        b, _, _ = make_state({0}, {5, 6}, [0])
        for commits, category in [({4}, 'new_only'), ({7, 1}, 'new_and_old'),
                                  ({1}, 'old_only'), ({12}, 'neither')]:
            with self.subTest(category=category):
                a, _, _ = make_state({0, 5, 6}, commits, [5, 6])
                _, _, actual = continuation(b, a)
                self.assertEqual(actual, category)

    def test_same_batch_old_extension_is_older_on_next_step(self):
        b, _, _ = make_state({0}, {1, 6}, [0])
        a, _, _ = make_state({0, 1, 6}, {2, 7}, [1, 6])
        self.assertEqual(continuation(b, a)[2], 'new_and_old')

    def test_current_region_means_connected_component_not_last_token(self):
        state, _, _ = make_state({0, 1, 2}, {8}, [1])
        self.assertEqual(state['active'], [[0, 1, 2]])
        self.assertEqual(state['alternatives'], {3})

    def test_next_special_not_silently_skipped(self):
        b, _, _ = make_state({0}, {6}, [0], policy='top1')
        a, _, _ = make_state({0, 6}, {19}, [6], end=18, policy='top1')
        self.assertEqual(top1_destination(b, a, 6, 4), 'special_or_after_final_stop')
        a, _, _ = make_state({0, 6}, {8}, [6], policy='top1')
        self.assertEqual(top1_destination(b, a, 6, 4), 'nonadjacent_intermediate_gap')

    def test_changed_top_prediction_does_not_imply_same_token_crossing(self):
        analyzer = Analyzer(4, [.85], .15, lambda _: None)
        pair = {'before': {'confidence': .7},
                'after': {'confidence': .95, 'special': False},
                'same_token_probability_after': .02, 'same_token_delta': -.68,
                'prediction_changed': True}
        analyzer.confidence_stats('a', 'test', [pair])
        rates = analyzer.rates.summary()
        self.assertEqual(rates['test/cutoff_0.85/neighbors_crossing']['numerator'], 1)
        self.assertEqual(rates['test/cutoff_0.85/same_token_crossing']['numerator'], 0)
        self.assertEqual(rates['test/same_token_drop_gt1pp']['numerator'], 1)

    def test_cocommitted_neighbor_has_no_next_prediction(self):
        b, _, _ = make_state({0}, {5, 6}, [0])
        a, _, _ = make_state({0, 5, 6}, {7}, [5, 6])
        self.assertIsNone(neighbor_pair(b, a, 6, 20))

    def test_pooled_and_equal_question_denominators_differ(self):
        rates = Rates()
        rates.add('x', 'a', 1, 1)
        rates.add('x', 'b', 0, 9)
        result = rates.summary()['x']
        self.assertEqual(result['percent'], 10)
        self.assertEqual(result['mean_per_problem_percent'], 50)

    def test_zip_and_directory_read_same_trace_without_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root/'samples'/'one'
            sample.mkdir(parents=True)
            trace = [{'type': 'header'}, {'type': 'step'}]
            payload = gzip.compress(('\n'.join(map(json.dumps, trace))+'\n').encode())
            (sample/'trace.jsonl.gz').write_bytes(payload)
            (sample/'result.json').write_text('{"sample_id": "one"}')
            archive = root/'saved.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('run/samples/one/trace.jsonl.gz', payload)
                z.writestr('run/samples/one/result.json', '{"sample_id": "one"}')
            for path in [root/'samples', archive]:
                source = Source(path)
                try:
                    self.assertEqual(list(source.records(source.names[0])), trace)
                    self.assertEqual(source.result(source.names[0])['sample_id'], 'one')
                finally:
                    source.close()


if __name__ == '__main__':
    unittest.main()
