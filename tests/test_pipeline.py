import tempfile
import unittest
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None
from confidence_geography.analyze import analyze_run, records, word_distance
from confidence_geography.demo import make_demo
from confidence_geography.run import score_logits

@unittest.skipIf(torch is None, 'torch needed for scorer and simulated pipeline')
class PipelineTests(unittest.TestCase):
    def test_score_preserves_raw_distribution_and_previous_token_probability(self):
        z = torch.tensor([[0., 1., 3., 4.], [3., 1., 0., -1.]])
        old = {0: {'token_id': 1, 'confidence': .2}}
        rows = score_logits(z, [0, 1], old, mask_id=3, top_k=2, chunk_size=1)
        p = torch.softmax(z[0], dim=-1)
        self.assertEqual(rows[0]['raw_top1_id'], 3)
        self.assertEqual(rows[0]['token_id'], 2)
        self.assertAlmostEqual(rows[0]['confidence'], p[2].item(), places=6)
        self.assertAlmostEqual(rows[0]['same_token_delta'], p[1].item()-.2, places=6)
        self.assertAlmostEqual(rows[0]['entropy'], -(p*p.log()).sum().item(), places=6)
        self.assertIsNone(rows[1]['delta_confidence'])
    def test_trace_reconstructs_and_analysis_detects_nonadjacent_fills(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = make_demo(root / 'run')
            trace = list(records(root / 'run/samples/demo/trace.jsonl.gz'))
            steps = [r for r in trace if r['type'] == 'step']
            self.assertEqual([s['commit_positions'][0] for s in steps], [0, 4, 3, 7, 2, 6, 1, 5])
            self.assertTrue(steps[1]['metrics']['nonlocal'])
            self.assertFalse(steps[2]['metrics']['nonlocal'])
            self.assertEqual(result['final_ids'], list(range(1, 9)))
            self.assertTrue(result['correct_strict'])
            self.assertEqual(result['first_stop_step'], 3)
            for before, after in zip(steps, steps[1:]):
                state = list(before['state_ids'])
                for p, token in zip(before['commit_positions'], before['commit_ids']): state[p] = token
                self.assertEqual(state, after['state_ids'])
                self.assertEqual(len(after['positions']), sum(t == 9 for t in state))
            summary = analyze_run(root / 'run', root / 'analysis', 1, .15)
            self.assertGreater(summary['remote_rise_events'], 0)
            self.assertTrue((root / 'analysis/overview.png').exists())
            self.assertTrue((root / 'analysis/sample_demo.png').exists())
            self.assertTrue((root / 'analysis/events.csv').read_text().startswith('sample_id'))
    def test_scheduled_and_threshold_terminate_with_complete_trace(self):
        for policy in ['scheduled', 'threshold', 'left_to_right']:
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory:
                result = make_demo(Path(directory), policy)
                self.assertEqual(result['final_ids'], list(range(1, 9)))
                self.assertEqual(result['steps'], 4 if policy == 'scheduled' else 8)
    def test_word_view_treats_same_word_fragments_as_local(self):
        mapping = {'valid': True, 'token_word_indices': [[0], [0], [1], [2], [3]]}
        self.assertEqual(word_distance(1, [0], mapping), 0)
        self.assertEqual(word_distance(2, [0], mapping), 1)
        self.assertEqual(word_distance(4, [0], mapping), 3)

if __name__ == '__main__': unittest.main()
