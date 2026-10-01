"""Make two offline histograms: neighbor top1-p AFTER reveal, and AFTER-BEFORE.

Run directly or as a module. No model, torch, or GPU is used.
Plotting requires matplotlib (already a project dependency).
"""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

if __package__:
    from .region_stats import Source
else:
    from region_stats import Source


def trace_pairs(path):
    """Yield each adjacent still-masked position once per actual transition."""
    source = Source(path)
    policy = None
    seen = set()
    try:
        for number, name in enumerate(source.names, 1):
            result = source.result(name)
            sid = str(result['sample_id'])
            if sid in seen:
                raise ValueError('Duplicate sample: use one policy/run at a time')
            seen.add(sid)
            end = result['answer_token_length']
            previous = None
            for record in source.records(name):
                if record['type'] == 'header':
                    config = record['config']
                    if policy is not None and policy != config['policy']:
                        raise ValueError('Mixed policies: analyze top1 and threshold separately')
                    policy = config['policy']
                    mask_id = config['mask_id']
                    continue
                if record['type'] != 'step':
                    continue
                rows = {r['position']: r for r in record['positions']}
                if previous is not None:
                    if record['step'] != previous['step']+1:
                        raise ValueError('Trace steps must be consecutive')
                    seeds = [p for p, r in previous['rows'].items()
                             if r['committed'] and p < end and not r['special']]
                    neighbors = {p+d for p in seeds for d in (-1, 1)}
                    anchors = {-1} | {p for p, tid in enumerate(previous['state']) if tid != mask_id}
                    seed_distances = {p: min(abs(p-a) for a in anchors) for p in seeds}
                    for p in sorted(neighbors):
                        before, after = previous['rows'].get(p), rows.get(p)
                        if before is None or after is None or p >= end:
                            continue  # A co-committed position is no longer masked.
                        if before['special'] or not before['eligible']:
                            continue
                        if not math.isclose(after['previous_confidence'], before['confidence'], abs_tol=1e-6):
                            raise ValueError('Previous confidence does not match the preceding pass')
                        yield {
                            'sample': sid, 'step': previous['step'], 'position': p,
                            'before': before['confidence'], 'after': after['confidence'],
                            'delta': after['confidence']-before['confidence'],
                            'changed': before['token_id'] != after['token_id'],
                            'before_token': before['text'], 'after_token': after['text'],
                            'after_special': after['special'], 'policy': policy,
                            'max_adjacent_seed_distance': max(seed_distances[q] for q in seeds if abs(q-p) == 1),
                        }
                previous = {'step': record['step'], 'rows': rows, 'state': record['state_ids']}
            if number % 10 == 0 or number == len(source.names):
                print(f'Read {number}/{len(source.names)} saved traces', flush=True)
    finally:
        source.close()


def saved_pairs(path):
    """Also accept neighbors.csv.gz produced by region_dynamics, avoiding a rescan."""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            for key in ('before', 'after', 'delta'):
                row[key] = float(row[key])
            for key in ('step', 'position', 'max_adjacent_seed_distance'):
                row[key] = int(row[key])
            for key in ('changed', 'after_special'):
                row[key] = row[key].lower() == 'true'
            yield row


def histogram(values, low, high, bins):
    width = (high-low)/bins
    counts = [0]*bins
    for value in values:
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'Value {value} outside histogram bounds [{low}, {high}]')
        index = min(bins-1, int((value-low)/width))
        counts[index] += 1
    return [{'left': low+i*width, 'right': low+(i+1)*width, 'count': n,
             'percent': 100*n/len(values) if values else None}
            for i, n in enumerate(counts)]


def generate(pairs, out, minimum=0, bins=50, label=None, log_y=False):
    pairs = [p for p in pairs if p['max_adjacent_seed_distance'] >= minimum]
    if not pairs:
        raise ValueError('No qualifying before/after neighbors; check input and distance filter')
    # Compute requested delta explicitly, rather than trusting a cached derived column.
    for row in pairs:
        row['delta'] = row['after']-row['before']
    policies = {p.get('policy') for p in pairs if p.get('policy')}
    if len(policies) > 1:
        raise ValueError('Mixed policies in neighbor data')
    label = label or next(iter(policies), 'Saved decoding run')
    after = [p['after'] for p in pairs]
    delta = [p['delta'] for p in pairs]
    distributions = {'top1p_after': histogram(after, 0, 1, bins),
                     'top1p_delta': histogram(delta, -1, 1, bins*2)}
    scope = ('All valid reveals' if minimum == 0 else
             f'Seed distance >= {minimum}: at least {minimum-1} intervening MASKs')
    summary = {
        'label': label, 'observations': len(pairs),
        'questions': len({p['sample'] for p in pairs}),
        'reveal_steps': len({(p['sample'], p['step']) for p in pairs}),
        'min_seed_distance': minimum, 'scope': scope,
        'mean_after': statistics.mean(after), 'median_after': statistics.median(after),
        'mean_delta': statistics.mean(delta), 'median_delta': statistics.median(delta),
        'prediction_changed_percent': 100*statistics.mean(p['changed'] for p in pairs),
        'definitions': {
            'neighbor': 'Exactly -1 or +1 from a revealed token; still masked on the next pass.',
            'after': 'Maximum candidate-token probability at that position on the next forward pass.',
            'delta': 'max-p(after) minus max-p(before); the maximizing token may change.',
            'weighting': 'Pooled neighbor observations, not equal-question averaging. Each position counted once per transition.',
            'population': 'Revealed seeds and pre-reveal neighbor predictions are nonspecial and before final stop. '
                          'Eligible pre-reveal positions only. After predictions retained even if special; actual next step never skipped.',
            'threshold': 'Neighbors committed in the same batch are excluded; observations follow the entire batch.',
            'filter': 'Distance from seed to nearest pre-reveal filled response position or prompt boundary, not distance from neighbor.',
        },
    }
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, rows in distributions.items():
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        centers = [(r['left']+r['right'])/2 for r in rows]
        width = rows[0]['right']-rows[0]['left']
        ax.bar(centers, [r['percent'] for r in rows], width=width*.95, color='#3478aa')
        if name == 'top1p_after':
            title = 'Adjacent masked tokens: top1 probability AFTER reveal'
            ax.set_xlabel('Top1 probability after reveal')
            ax.set_xlim(0, 1)
            ax.set_xticks([i/10 for i in range(11)])
        else:
            title = 'Adjacent masked tokens: change in top1 probability'
            ax.set_xlabel('Top1-p after - top1-p before (0.1 = 10 percentage points)')
            ax.set_xlim(-1, 1)
            ax.set_xticks([i/5 for i in range(-5, 6)])
            ax.axvline(0, color='#333333', linewidth=1)
        ax.set_ylabel('Percentage of paired neighbor observations')
        ax.set_title(f'{title}\n{label} | {scope} | n={len(pairs):,}', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        if log_y:
            ax.set_yscale('log')
        for extension in ('png', 'pdf'):
            fig.savefig(out/f'{name}.{extension}', dpi=180)
        plt.close(fig)
        with (out/f'{name}.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['left', 'right', 'count', 'percent'])
            writer.writeheader()
            writer.writerows(rows)
    with gzip.open(out/'paired_neighbors.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        fields = ['sample', 'step', 'position', 'before', 'after', 'delta', 'changed',
                  'before_token', 'after_token', 'after_special', 'max_adjacent_seed_distance']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(pairs)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', type=Path, help='One saved policy directory or ZIP')
    source.add_argument('--neighbors', type=Path, help='Existing neighbors.csv.gz from region_dynamics')
    parser.add_argument('--out', type=Path, required=True, help='New/empty output folder')
    parser.add_argument('--min-seed-distance', type=int, default=0,
                        help='Optional index-distance filter; 4 = at least 3 intervening MASKs. Default: all reveals.')
    parser.add_argument('--bins', type=int, default=50, help='Bins over probability range 0..1; delta uses twice as many')
    parser.add_argument('--label', help='Optional plot label, useful with --neighbors')
    parser.add_argument('--log-y', action='store_true', help='Logarithmic y-axis to inspect smaller histogram bars')
    args = parser.parse_args()
    if args.min_seed_distance < 0 or args.bins < 2:
        parser.error('Distance must be nonnegative and bins must be >=2')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Use a new/empty output folder')
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = trace_pairs(args.run) if args.run else saved_pairs(args.neighbors)
    summary = generate(pairs, args.out, args.min_seed_distance, args.bins, args.label, args.log_y)
    print(f"Plotted {summary['observations']:,} paired neighbors from {summary['questions']} questions")
    print(args.out/'top1p_after.png')
    print(args.out/'top1p_delta.png')


if __name__ == '__main__':
    main()
