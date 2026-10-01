"""Plot exact signed distances from an existing events.csv; no GPU or model needed."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def distribution(events):
    by_problem = defaultdict(list)
    for event in events:
        distance = int(event['signed_distance'])
        if distance == 0:
            raise ValueError('A commit cannot occupy a previous commit position (distance zero)')
        by_problem[event['sample_id']].append(distance)
    counts = Counter()
    percentages = defaultdict(float)
    for distances in by_problem.values():
        counts.update(distances)
        for distance in distances:
            percentages[distance] += 100 / (len(by_problem) * len(distances))
    return dict(percentages), dict(counts), len(by_problem)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='Output directory')
    p.add_argument('--zoom', type=int, default=20, help='Show integer distances from -zoom to +zoom')
    p.add_argument('--label', default='Saved decoding run', help='Plot title label, e.g. TOP1 or THRESHOLD')
    args = p.parse_args()
    if args.zoom < 2:
        p.error('--zoom must be at least 2')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with args.events.open(newline='', encoding='utf-8') as handle:
        primary = [r for r in csv.DictReader(handle)
                   if r['kind'] == 'commit' and r['primary_population'].lower() == 'true']
    populations = [('All counted fills', primary),
                   ('Adjacent option available', [r for r in primary
                    if r['answer_local_option_available'].lower() == 'true'])]
    # Keep the JSON keys stable, but make the two plotted populations explicit.
    panel_titles = {
        'All counted fills': 'All valid commitments',
        'Adjacent option available': 'Only steps where local continuation was available\n'
                                     '(eligible masked neighbor of the previous commitment/batch)',
    }
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    summaries = {}
    for row, (name, events) in enumerate(populations):
        percentages, counts, problems = distribution(events)
        summary = {'problems': problems, 'commits': len(events),
                   'weighting': 'Mean within-problem percentage; each problem has equal weight',
                   'distance_percentages': dict(sorted(percentages.items())),
                   'distance_counts': dict(sorted(counts.items()))}
        summaries[name] = summary
        if not events:
            for ax in axes[row]: ax.text(.5, .5, 'No qualifying fills', ha='center', transform=ax.transAxes)
            continue
        summary.update({
            'nonadjacent_percent': sum(v for d, v in percentages.items() if abs(d) > 1),
            'left_neighbor_percent': percentages.get(-1, 0),
            'right_neighbor_percent': percentages.get(1, 0),
            'outside_zoom_left_percent': sum(v for d, v in percentages.items() if d < -args.zoom),
            'outside_zoom_right_percent': sum(v for d, v in percentages.items() if d > args.zoom)})
        for col, ax in enumerate(axes[row]):
            low, high = (min(min(percentages), -1), max(max(percentages), 1)) if col == 0 else (-args.zoom, args.zoom)
            xx = list(range(low, high+1))
            ax.bar(xx, [percentages.get(d, 0) for d in xx], width=.8,
                   color=['#e58b20' if abs(d) == 1 else '#3478aa' for d in xx])
            ax.set_xlim(low-.75, high+.75)
            ax.set_xlabel('Signed token distance: left < 0 < right')
            ax.set_ylabel('Mean within-problem percentage of fills')
            ax.grid(axis='y', alpha=.2)
            if col == 1:
                ax.set_xticks(xx)
                ax.tick_params(axis='x', labelsize=8, rotation=90)
                ax.set_title(f'{panel_titles[name]}\nZoom ±{args.zoom}; one bar per integer', fontsize=10)
                ax.text(.02, .97,
                        f"Outside zoom: left {summary['outside_zoom_left_percent']:.1f}%, right {summary['outside_zoom_right_percent']:.1f}%\n"
                        f"Neighbors: −1 = {summary['left_neighbor_percent']:.1f}%, +1 = {summary['right_neighbor_percent']:.1f}%\n"
                        f"Nonadjacent: {summary['nonadjacent_percent']:.1f}%",
                        transform=ax.transAxes, va='top', fontsize=10,
                        bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none'})
            else:
                tick_step = next((s for s in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
                                  if (high-low)/s <= 20), max(1, (high-low)//20))
                first_tick = -((-low)//tick_step) * tick_step
                ax.set_xticks(list(range(first_tick, high+1, tick_step)))
                ax.tick_params(axis='x', labelsize=8, rotation=60)
                ax.set_title(f'{panel_titles[name]}\nFull range; {problems} questions', fontsize=10)
        # Zoom is a crop, not a renormalized distribution; keep vertical scales equal.
        maximum = max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]: ax.set_ylim(0, maximum)
    fig.suptitle(f'{args.label}: distance from the previous commitment/batch\n'
                 'Orange = immediate neighbors; blue = nonadjacent. Zoom retains the full-population denominator.\n'
                 'For batch decoding, distance is measured from the nearest token in the previous batch.')
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / 'distances.png', dpi=170)
    plt.close(fig)
    # A standalone zoom avoids mistaking the full-range panel for the cropped view.
    zoom_fig, zoom_axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    xx = list(range(-args.zoom, args.zoom+1))
    for ax, (name, summary) in zip(zoom_axes, summaries.items()):
        percentages = summary['distance_percentages']
        ax.bar(xx, [percentages.get(d, 0) for d in xx], width=.8,
               color=['#e58b20' if abs(d) == 1 else '#3478aa' for d in xx])
        ax.set_xlim(-args.zoom-.5, args.zoom+.5)
        ax.set_xticks(xx, labels=[str(d) for d in xx])
        ax.tick_params(axis='x', labelsize=9, rotation=90)
        ax.set_xlabel('Signed token distance from previous step (negative = left, positive = right)')
        ax.set_ylabel('Mean within-problem percentage of fills')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        if summary['commits']:
            ax.set_title(
                f"{panel_titles[name]}\nNeighbors: −1 = {summary['left_neighbor_percent']:.1f}%, "
                f"+1 = {summary['right_neighbor_percent']:.1f}%\n"
                f"Outside this window: left {summary['outside_zoom_left_percent']:.1f}%, "
                f"right {summary['outside_zoom_right_percent']:.1f}% (included in denominator)")
        else:
            ax.set_title(f'{panel_titles[name]}\nNo qualifying fills')
    zoom_fig.suptitle(f'{args.label}: distance from the previous commitment/batch\n'
                      f'Zoom −{args.zoom} to +{args.zoom}; tails remain in the percentage denominator\n'
                      'Orange = immediate neighbors; blue = nonadjacent')
    zoom_fig.savefig(args.out / 'distances_zoom.png', dpi=170)
    plt.close(zoom_fig)
    (args.out / 'distances.json').write_text(json.dumps(summaries, indent=2) + '\n')
    print(f"Open the standalone zoom: {args.out / 'distances_zoom.png'}")
    print(f"Full-range comparison: {args.out / 'distances.png'}; exact values: {args.out / 'distances.json'}")


if __name__ == '__main__':
    main()
