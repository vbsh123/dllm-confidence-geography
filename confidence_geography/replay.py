"""Replay one transition, withholding previous-step commits. GPU required."""
import argparse
import json
from pathlib import Path

from .analyze import records
from .run import dump, load_model, score_logits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--step', type=int, required=True, help='Post-transition step, at least 1')
    p.add_argument('--position', type=int, required=True, help='Still-masked response position to investigate')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-individual', type=int, default=8)
    args = p.parse_args()
    if args.step < 1 or args.max_individual < 0:
        p.error('step must be >=1; max-individual must be nonnegative')
    header, selected, before = None, None, None
    for record in records(args.trace):
        if record['type'] == 'header': header = record
        if record['type'] == 'step' and record['step'] == args.step-1: before = record
        if record['type'] == 'step' and record['step'] == args.step: selected = record
    if not header or not selected or not before:
        raise ValueError('Requested transition missing from trace')
    config = header['config']
    mask_id = config['mask_id']
    if not 0 <= args.position < len(selected['state_ids']) or selected['state_ids'][args.position] != mask_id:
        raise ValueError('Target must still be masked at the selected step')
    commits = selected['previous_commits']
    if commits != before['commit_positions']:
        raise ValueError('Trace transition is inconsistent')
    observed = next(r for r in selected['positions'] if r['position'] == args.position)
    old = next(r for r in before['positions'] if r['position'] == args.position)
    import torch
    model, _ = load_model(config)
    target_id = observed['token_id']
    rows = []
    interventions = [('observed', []), ('withhold_all', commits)]
    interventions += [(f'withhold_{i}', [i]) for i in commits[:args.max_individual]]
    with torch.inference_mode():
        for name, withheld in interventions:
            state = list(selected['state_ids'])
            for position in withheld: state[position] = mask_id
            if name == 'withhold_all' and state != before['state_ids']:
                raise ValueError('Withholding prior batch did not reproduce prior state')
            x = torch.tensor([header['prompt_ids'] + state], device=model.device)
            logits = model(x, use_cache=False).logits[0, len(header['prompt_ids']):]
            target_prob = torch.softmax(logits[args.position].float(), dim=-1)[target_id].item()
            row = score_logits(logits, [args.position], {args.position: old}, mask_id, config['top_k'])[0]
            rows.append({'intervention': name, 'withheld_positions': withheld,
                         'observed_target_id': target_id, 'observed_target_probability': target_prob, **row})
    baseline = rows[0]['observed_target_probability']
    for row in rows:
        row['target_probability_difference_from_observed'] = row['observed_target_probability']-baseline
    result = {'trace': str(args.trace), 'step': args.step, 'position': args.position,
              'recorded_confidence': observed['confidence'],
              'replay_confidence_error': rows[0]['confidence']-observed['confidence'],
              'replay_top1_matches': rows[0]['token_id'] == observed['token_id'],
              'prior_state_confidence_error': rows[1]['confidence']-old['confidence'],
              'interventions': rows,
              'interpretation': 'Conditional effect on one forward prediction of withholding committed tokens. Includes mask-count effects; not proof of semantic reasoning or general causality.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
