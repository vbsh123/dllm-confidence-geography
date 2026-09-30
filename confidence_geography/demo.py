"""Tiny simulated CPU trace to test the pipeline; these are NOT model findings."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from .run import collect_sample, dump

class DemoTokenizer:
    all_special_ids = [0, 8, 9]
    eos_token_id = 8
    pieces = ['<prompt>', 'First', ' ', '2', '+', '2', ' = ', '4', '<eos>', '<mask>']
    def apply_chat_template(self, messages, **kwargs): return messages[0]['content']
    def encode(self, text, **kwargs): return [0]
    def get_vocab(self): return {p: i for i, p in enumerate(self.pieces)}
    def convert_ids_to_tokens(self, token_id): return self.pieces[token_id]
    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return ''.join(self.pieces[i] for i in ids if not skip_special_tokens or i not in self.all_special_ids)

class DemoModel:
    device = 'cpu'
    config = SimpleNamespace(max_sequence_length=128)
    def __call__(self, x, **kwargs):
        import torch
        state = x[0, 1:].tolist()
        schedule = [0, 4, 3, 7, 2, 6, 1, 5]
        leader = next(p for p in schedule if state[p] == 9)
        z = torch.zeros((1, x.shape[1], 10), device=x.device)
        for p in range(8): z[0, p+1, p+1] = 1.2 + p * .1
        z[0, leader+1, leader+1] = 8
        return SimpleNamespace(logits=z)

def make_demo(out, policy='top1'):
    out = Path(out)
    path = out / 'samples' / 'demo'
    path.mkdir(parents=True, exist_ok=True)
    config = {'model': 'SIMULATED-TEST-FIXTURE', 'revision': 'none', 'mask_id': 9,
              'length': 8, 'block_length': 8, 'policy': policy, 'steps': 4,
              'top_k': 3, 'thresholds': [.5, .9, .99], 'commit_threshold': .9, 'log_every': 100}
    dump(out / 'manifest.json', {'SIMULATED': True, 'config': config})
    sample = {'id': 'demo', 'question': 'SIMULATED PIPELINE TEST: what is 2+2?', 'answer': '#### 4'}
    return collect_sample(DemoModel(), DemoTokenizer(), config, sample, path)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    make_demo(args.out)
    print(f'Simulated trace written to {args.out}. Not empirical model data.')

if __name__ == '__main__': main()
