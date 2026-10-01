"""Plot neighbor confidence AFTER reveals away from the CURRENT filled region.

All selection, pairing, calculation, and plotting live in
 generate_neighbor_confidence_plots(). Saved traces only; no model inference.
"""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

if __package__:
    from .region_stats import Source  # ZIP/directory file reading only.
else:
    from region_stats import Source


def generate_neighbor_confidence_plots(run, out, bins=50, label=None, log_y=False):
    """Read traces, select non-neighboring reveals, pair neighbors, and plot.

    At reveal step t:
      * CURRENT region = contiguous filled response text containing a valid
        commitment from step t-1. A threshold batch may define several regions.
      * Select a token committed at t only if it is neither inside nor directly
        beside ANY current region, measured in the state BEFORE that reveal.
        No arbitrary distance-4 cutoff. Returning beside OLDER text is allowed.
      * For its neighbors (-1/+1), compare top1-p before the reveal (forward t)
        with top1-p after the reveal (forward t+1). Keep only still-masked ones.
      * Plot after-p and after-p minus before-p. The predicted token may change.

    Threshold observations follow the entire batch; they cannot establish one
    seed's individual effect. Same-batch bridging is not an extra exclusion:
    selection always refers to the PRE-reveal current-region boundaries.
    """
    run, out = Path(run), Path(out)
    if bins < 2:
        raise ValueError('bins must be at least 2')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new/empty output folder')
    out.mkdir(parents=True, exist_ok=True)

    pairs = []
    selected_reveals = []
    samples_seen = set()
    policy = None
    source = Source(run)

    # 1. Read each question's saved forward passes. Source only handles file I/O.
    try:
        for number, name in enumerate(source.names, 1):
            result = source.result(name)
            sample_id = str(result['sample_id'])
            if sample_id in samples_seen:
                raise ValueError('Duplicate question: use one policy/run at a time')
            samples_seen.add(sample_id)
            answer_end = result['answer_token_length']
            token_dictionary = result['token_dictionary']
            before_step = None

            for record in source.records(name):
                if record['type'] == 'header':
                    config = record['config']
                    mask_id = config['mask_id']
                    if policy is not None and policy != config['policy']:
                        raise ValueError('Mixed policies: analyze top1 and threshold separately')
                    policy = config['policy']
                    continue
                if record['type'] != 'step':
                    continue
                if before_step is None:
                    before_step = record
                    continue
                if record['step'] != before_step['step'] + 1:
                    raise ValueError('Trace steps must be consecutive')

                # before_step is forward t, before its commitments are inserted.
                # record is forward t+1, after those commitments were inserted.
                before_rows = {r['position']: r for r in before_step['positions']}
                after_rows = {r['position']: r for r in record['positions']}
                state_before_reveal = before_step['state_ids']

                # 2. Find the CURRENT filled region(s), using commitments at t-1.
                # Expand from each previous commitment through contiguous filled
                # positions. This avoids calling an extension of the same region
                # a jump merely because the latest token lies inside that region.
                current_regions = set()
                for previous_position in before_step['previous_commits']:
                    if previous_position >= answer_end:
                        continue
                    previous_token = state_before_reveal[previous_position]
                    if previous_token == mask_id:
                        raise ValueError('A previous commitment is unexpectedly masked')
                    if token_dictionary[str(previous_token)]['special']:
                        continue
                    left = right = previous_position
                    while left > 0 and state_before_reveal[left-1] != mask_id:
                        left -= 1
                    while right+1 < answer_end and state_before_reveal[right+1] != mask_id:
                        right += 1
                    current_regions.add((left, right))

                if not current_regions:
                    # No valid previous region, e.g. only EOS was just revealed.
                    before_step = record
                    continue

                # 3. Select reveals AWAY FROM the current region(s).
                # Example: current region [0..4], reveal 5 -> exclude (adjacent).
                #          current region [0..4], reveal 6 -> include.
                # A reveal can be next to an older region and still qualify.
                selected_seeds = []
                for position, row in before_rows.items():
                    if not row['committed'] or position >= answer_end or row['special']:
                        continue
                    touches_current_region = any(
                        left-1 <= position <= right+1
                        for left, right in current_regions
                    )
                    if not touches_current_region:
                        selected_seeds.append(position)

                if selected_seeds:
                    selected_reveals.append({
                        'sample': sample_id, 'step': before_step['step'],
                        'current_regions': sorted(current_regions),
                        'seed_positions': sorted(selected_seeds),
                        'seed_tokens': [before_rows[p]['text'] for p in sorted(selected_seeds)],
                        'all_batch_commits': before_step['commit_positions'],
                    })

                # 4. Inspect exactly -1/+1 around the selected NEWLY revealed
                # tokens. A neighbor shared by two seeds is counted only once.
                neighbor_positions = {p+d for p in selected_seeds for d in (-1, 1)}
                for position in sorted(neighbor_positions):
                    before = before_rows.get(position)
                    after = after_rows.get(position)
                    if before is None or after is None or position >= answer_end:
                        continue  # Already filled, or co-committed in batch t.
                    if before['special'] or not before['eligible']:
                        continue
                    if not math.isclose(after['previous_confidence'], before['confidence'], abs_tol=1e-6):
                        raise ValueError('Neighbor confidence is not aligned with the previous pass')

                    # 5. These are the two requested quantities, for this SAME
                    # still-masked position on two consecutive forward passes.
                    probability_before = before['confidence']
                    probability_after = after['confidence']
                    delta = probability_after - probability_before

                    pairs.append({
                        'sample': sample_id, 'step': before_step['step'],
                        'position': position,
                        'seed_positions': [p for p in selected_seeds if abs(p-position) == 1],
                        'current_regions': sorted(current_regions),
                        'before': probability_before, 'after': probability_after, 'delta': delta,
                        'before_token': before['text'], 'after_token': after['text'],
                        'prediction_changed': before['token_id'] != after['token_id'],
                        'after_special': after['special'],
                    })
                before_step = record
            if number % 10 == 0 or number == len(source.names):
                print(f'Read {number}/{len(source.names)} saved traces', flush=True)
    finally:
        source.close()

    after_values = [pair['after'] for pair in pairs]
    delta_values = [pair['delta'] for pair in pairs]
    label = label or policy or 'Saved decoding run'

    # 6. Build and draw the two histograms. Y is percent of ALL selected paired
    # observations; delta is in probability units (0.1 = 10 percentage points).
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for filename, values, low, high, bin_count, title, xlabel in [
        ('top1p_after', after_values, 0, 1, bins,
         'Neighbor top1 probability AFTER a non-neighboring reveal',
         'Top1 probability after reveal'),
        ('top1p_delta', delta_values, -1, 1, bins*2,
         'Neighbor top1 probability change AFTER a non-neighboring reveal',
         'Top1-p after - top1-p before (0.1 = 10 percentage points)'),
    ]:
        width = (high-low)/bin_count
        counts = [0]*bin_count
        for value in values:
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'Probability value {value} outside [{low}, {high}]')
            index = min(bin_count-1, int((value-low)/width))
            counts[index] += 1
        hist = [{'left': low+i*width, 'right': low+(i+1)*width,
                 'count': count, 'percent': 100*count/len(values) if values else None}
                for i, count in enumerate(counts)]
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        ax.bar([(r['left']+r['right'])/2 for r in hist],
               [r['percent'] or 0 for r in hist], width=width*.95, color='#3478aa')
        ax.set_xlim(low, high)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Percentage of paired neighbor observations')
        ax.set_title(f'{title}\n{label} | away from current region(s) | n={len(values):,}', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        if filename == 'top1p_delta':
            ax.axvline(0, color='#333333', linewidth=1)
        if not values:
            ax.text(.5, .5, 'No qualifying neighbor observations', ha='center', transform=ax.transAxes)
        elif log_y:
            ax.set_yscale('log')
        for extension in ('png', 'pdf'):
            fig.savefig(out/f'{filename}.{extension}', dpi=180)
        plt.close(fig)
        with (out/f'{filename}.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['left', 'right', 'count', 'percent'])
            writer.writeheader()
            writer.writerows(hist)

    # 7. Save the exact selected reveals and before/after pairs for inspection.
    with gzip.open(out/'selected_reveals.jsonl.gz', 'wt', encoding='utf-8') as f:
        for reveal in selected_reveals:
            f.write(json.dumps(reveal, ensure_ascii=False)+'\n')
    with gzip.open(out/'paired_neighbors.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        fields = ['sample', 'step', 'position', 'seed_positions', 'current_regions',
                  'before', 'after', 'delta', 'before_token', 'after_token',
                  'prediction_changed', 'after_special']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in pair.items()})
    summary = {
        'label': label, 'input': str(run.resolve()), 'questions_read': len(samples_seen),
        'observations': len(pairs), 'questions_with_pairs': len({p['sample'] for p in pairs}),
        'qualifying_reveal_steps_with_next_pass': len(selected_reveals),
        'qualifying_seeds_with_next_pass': sum(len(r['seed_positions']) for r in selected_reveals),
        'mean_after': statistics.mean(after_values) if pairs else None,
        'median_after': statistics.median(after_values) if pairs else None,
        'mean_delta': statistics.mean(delta_values) if pairs else None,
        'median_delta': statistics.median(delta_values) if pairs else None,
        'definition': 'Current region: contiguous pre-reveal filled response text containing a '
                      'valid previous-step commitment. Select commits neither inside nor adjacent to '
                      'any such region. Return to older regions is allowed. No distance-4 cutoff. '
                      'Pair immediate seed neighbors still masked on next actual forward pass. '
                      'Seeds and pre-reveal neighbor predictions nonspecial/pre-final-stop; neighbors '
                      'eligible before reveal. After predictions retained even if special. '
                      'Threshold effects are after the entire batch; same-batch bridges are not excluded. '
                      'Histogram percentages pool unique neighbor positions per step. '
                      'Delta compares maxima even when the predicted token changes.',
    }
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='One saved policy directory or ZIP')
    parser.add_argument('--out', type=Path, required=True, help='New/empty output folder')
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--label')
    parser.add_argument('--log-y', action='store_true')
    args = parser.parse_args()
    summary = generate_neighbor_confidence_plots(args.run, args.out, args.bins, args.label, args.log_y)
    print(f"Plotted {summary['observations']:,} paired neighbors")
    print(args.out/'top1p_after.png')
    print(args.out/'top1p_delta.png')


if __name__ == '__main__':
    main()
