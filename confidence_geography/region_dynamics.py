"""Saved-trace histograms and region dynamics; never loads a model.

python -m confidence_geography.region_dynamics --run RUN_OR_ZIP --out NEW_FOLDER
Collection uses the standard library; plots require numpy and matplotlib.
"""
import argparse
from collections import Counter, defaultdict, deque
import csv
import gzip
import json
from pathlib import Path
import statistics

from .region_stats import Source, Rates, components


def regions_at(filled, mask_gap):
    """Separate groups iff consecutive filled positions have >=X masks between."""
    groups = []
    for p in sorted(filled):
        if not groups or p-groups[-1][-1]-1 >= mask_gap:
            groups.append([p])
        else:
            groups[-1].append(p)
    return groups


def candidate_regions(position, regions, mask_gap):
    """Which old regions would this single fill join under the same gap rule?"""
    return [i for i, g in enumerate(regions)
            if g[0]-mask_gap <= position <= g[-1]+mask_gap]


def direction(position, region_ids, regions):
    if len(region_ids) > 1:
        return 'bridge_multiple_regions'
    if not region_ids:
        return 'isolated'
    group = regions[region_ids[0]]
    if position == group[-1]+1:
        return 'immediate_right'
    if position == group[0]-1:
        return 'immediate_left'
    if group[0] < position < group[-1]:
        return 'internal_hole'
    return 'right_skip' if position > group[-1] else 'left_skip'


def leftmost_unresolved(group, rows, end):
    return next((q for q in range(group[0]+1, min(end, group[-1]+2))
                 if q in rows and rows[q]['eligible']), None)


def pct(n, d):
    return 100*n/d if d else None


def collect(run, out, gaps, focus, windows, lookback):
    source = Source(run)
    rates = Rates()
    counts = Counter()
    signed = defaultdict(Counter)
    old_gap = defaultdict(Counter)
    neighbors = []
    birth_history = []
    all_stats = defaultdict(lambda: defaultdict(list))
    seen = set()
    policy = None
    configs = []
    with gzip.open(out/'frames.jsonl.gz', 'wt', encoding='utf-8') as frames:
        try:
            for number, name in enumerate(source.names, 1):
                result = source.result(name)
                sid = str(result['sample_id'])
                if sid in seen:
                    raise ValueError('Duplicate sample: use one policy/run at a time')
                seen.add(sid)
                end = result['answer_token_length']
                dictionary = result['token_dictionary']
                history = deque(maxlen=max(lookback+1, max(windows)))
                prev = None
                last_step = -1
                for record in source.records(name):
                    if record['type'] == 'header':
                        config = record['config']
                        if policy is not None and policy != config['policy']:
                            raise ValueError('Mixed policies in input')
                        policy = config['policy']
                        mask = config['mask_id']
                        if config not in configs:
                            configs.append(config)
                        continue
                    if record['type'] != 'step':
                        continue
                    step = record['step']
                    if step != last_step+1:
                        raise ValueError('Nonconsecutive steps')
                    last_step = step
                    rows = {r['position']: r for r in record['positions']}
                    good_rows = {p: r for p, r in rows.items()
                                 if p < end and r['eligible'] and not r['special']}
                    chosen = {p for p, r in rows.items() if r['committed']}
                    good_chosen = chosen.intersection(good_rows)
                    filled = {p for p, t in enumerate(record['state_ids']) if p < end and t != mask}
                    # All full-window filled tokens anchor the distance statistic,
                    # matching the earlier analysis, including terminal stop tokens.
                    full_filled = {p for p, t in enumerate(record['state_ids']) if t != mask}
                    anchors = full_filled | {-1}
                    counts['steps'] += 1
                    counts['valid_commits'] += len(good_chosen)
                    prior = record['previous_commits']
                    primary = bool(prior) and all(p < end and not dictionary[
                        str(record['state_ids'][p])]['special'] for p in prior)
                    for p in good_chosen:
                        if primary:
                            anchor = min(prior, key=lambda q: (abs(p-q), q))
                            signed[sid][p-anchor] += 1
                        old_gap[sid][min(abs(p-q) for q in anchors)-1] += 1
                    if prev:
                        if step != prev['step']+1:
                            raise ValueError('Missing previous frame')
                        adjacent = {p+d for p in prev['good_chosen'] for d in (-1, 1)}
                        for p in sorted(adjacent):
                            before, after = prev['good_rows'].get(p), rows.get(p)
                            if before is None or after is None or p >= end:
                                continue  # Excludes simultaneous fills.
                            adjacent_seeds = [q for q in prev['good_chosen'] if abs(p-q) == 1]
                            distance = max(min(abs(q-f) for f in prev['anchors']) for q in adjacent_seeds)
                            same = after.get('previous_token_probability_now')
                            neighbors.append({
                                'sample': sid, 'step': prev['step'], 'position': p,
                                'before': before['confidence'], 'after': after['confidence'],
                                'delta': after['confidence']-before['confidence'],
                                'same_token_delta': same-before['confidence'] if same is not None else None,
                                'changed': before['token_id'] != after['token_id'],
                                'after_special': after['special'],
                                'max_adjacent_seed_distance': distance,
                                'before_token': before['text'], 'after_token': after['text'],
                            })
                    if any(p < end for p in rows):
                        progress = len(filled)/end if end else 0
                        for gap in gaps:
                            groups = regions_at(filled, gap)
                            mapping = {p: i for i, group in enumerate(groups) for p in group}
                            recent = {}
                            for window in windows:
                                touched = {p for h in list(history)[-window:] for p in h['good_chosen']}
                                recent[window] = {mapping[p] for p in touched if p in mapping}
                            last_regions = recent[1]
                            active = recent[5]
                            candidates = {p: candidate_regions(p, groups, gap) for p in good_rows}
                            region_max = [None]*len(groups)
                            maxima = {'recent': None, 'older': None, 'unassigned': None}
                            for p, ids in candidates.items():
                                confidence = good_rows[p]['confidence']
                                for i in ids:
                                    region_max[i] = max(region_max[i] or 0, confidence)
                                kind = 'recent' if active.intersection(ids) else ('older' if ids else 'unassigned')
                                maxima[kind] = max(maxima[kind] or 0, confidence)
                            # Gap-defined groups of valid simultaneous commitments;
                            # every member must be remote from existing groups AND prompt.
                            simultaneous = regions_at(good_chosen, gap)
                            born = [g for g in simultaneous
                                    if all(not candidates[p] and min(abs(p-a) for a in anchors) > gap for p in g)]
                            base = f'gap_{gap}'
                            rates.add(base+'/steps/occupied_le2', sid, len(groups) <= 2)
                            rates.add(base+'/steps/recent5_le2', sid, len(active) <= 2)
                            rates.add(base+'/steps/recent10_le2', sid, len(recent[10]) <= 2)
                            rates.add(base+'/steps/new_region', sid, bool(born))
                            all_stats[gap][sid].append((len(groups), len(active), len(recent[10])))
                            for p in good_chosen:
                                ids = candidates[p]
                                where = direction(p, ids, groups)
                                rates.add(base+'/commits/joins_existing', sid, bool(ids))
                                if ids:
                                    for d in ('immediate_right', 'immediate_left', 'internal_hole',
                                              'right_skip', 'left_skip', 'bridge_multiple_regions'):
                                        rates.add(base+'/existing_commits/'+d, sid, where == d)
                                    if active.intersection(ids):
                                        for d in ('immediate_right', 'immediate_left', 'internal_hole',
                                                  'right_skip', 'left_skip', 'bridge_multiple_regions'):
                                            rates.add(base+'/recent5_commits/'+d, sid, where == d)
                                        if len(ids) == 1:
                                            group = groups[ids[0]]
                                            frontier = leftmost_unresolved(group, rows, end)
                                            if frontier is not None:
                                                rates.add(base+'/recent5_single_region_commits/leftmost_unresolved',
                                                          sid, p == frontier)
                                if policy == 'top1' and last_regions:
                                    dest = 'same' if last_regions.intersection(ids) else ('other_existing' if ids else 'unassigned')
                                    for d in ('same', 'other_existing', 'unassigned'):
                                        rates.add(base+'/top1_next/'+d, sid, dest == d)
                            frame = {'sample': sid, 'step': step, 'gap': gap, 'progress': progress,
                                     'occupied': len(groups), 'recent': {str(w): len(v) for w, v in recent.items()},
                                     'regions': [[g[0], g[-1]] for g in groups], 'region_max': region_max,
                                     'maxima': maxima, 'new_regions': born,
                                     'commits': sorted(good_chosen)}
                            frames.write(json.dumps(frame)+'\n')
                            if gap == focus and born:
                                # Track fixed would-be seed positions BACKWARD, avoiding
                                # the misleading comparison of different winners over time.
                                for region in born:
                                    for p in region:
                                        current = good_rows[p]
                                        seq = list(history)[-lookback:]
                                        for old in seq + [{'step': step, 'good_rows': good_rows}]:
                                            row = old['good_rows'].get(p)
                                            if row is not None:
                                                birth_history.append({
                                                    'sample': sid, 'birth_step': step, 'position': p,
                                                    'offset': old['step']-step, 'confidence': row['confidence'],
                                                    'same_prediction_as_commit': row['token_id'] == current['token_id'],
                                                })
                    item = {'step': step, 'good_rows': good_rows, 'rows': rows,
                            'good_chosen': good_chosen, 'anchors': anchors}
                    history.append(item)
                    prev = item
                if number % 10 == 0 or number == len(source.names):
                    print(f'{policy}: analyzed {number}/{len(source.names)} traces', flush=True)
        finally:
            source.close()
    for filename, items in [('neighbors.csv.gz', neighbors), ('birth_history.csv.gz', birth_history)]:
        with gzip.open(out/filename, 'wt', newline='', encoding='utf-8') as f:
            if items:
                writer = csv.DictWriter(f, fieldnames=list(items[0]))
                writer.writeheader()
                writer.writerows(items)
    histograms = {}
    for label, samples in [('signed_previous', signed), ('masked_gap_to_nearest_filled', old_gap)]:
        pooled = sum(samples.values(), Counter())
        histograms[label] = {'count': dict(sorted(pooled.items())),
            'pooled_percent': {d: pct(n, pooled.total()) for d, n in sorted(pooled.items())},
            'mean_per_problem_percent': {d: statistics.mean(100*c[d]/c.total() for c in samples.values())
                                         for d in sorted(pooled)},
            'problems': len(samples), 'observations': pooled.total()}
    neighbor_summary = {}
    for label, minimum in [('all_reveals', 0), ('legacy_distance4', 4), (f'mask_gap{focus}', focus+1)]:
        subset = [n for n in neighbors if n['max_adjacent_seed_distance'] >= minimum]
        neighbor_summary[label] = {'n': len(subset)}
        if subset:
            same = [n['same_token_delta'] for n in subset if n['same_token_delta'] is not None]
            neighbor_summary[label].update({
                'mean_after': statistics.mean(n['after'] for n in subset),
                'median_after': statistics.median(n['after'] for n in subset),
                'mean_delta': statistics.mean(n['delta'] for n in subset),
                'median_delta': statistics.median(n['delta'] for n in subset),
                'mean_same_token_delta': statistics.mean(same) if same else None,
                'prediction_changed_percent': 100*statistics.mean(n['changed'] for n in subset),
                'after_ge90_percent': 100*statistics.mean(n['after'] >= .9 for n in subset),
                'delta_ge15pp_percent': 100*statistics.mean(n['delta'] >= .15 for n in subset),
            })
    region_summary = {}
    for gap, samples in all_stats.items():
        pooled = [v for values in samples.values() for v in values]
        region_summary[gap] = {
            'frames': len(pooled),
            'occupied_mean': statistics.mean(v[0] for v in pooled),
            'occupied_mean_per_problem': statistics.mean(statistics.mean(v[0] for v in values) for values in samples.values()),
            'occupied_median': statistics.median(v[0] for v in pooled),
            'occupied_max': max(v[0] for v in pooled),
            'recent5_mean': statistics.mean(v[1] for v in pooled),
            'recent10_mean': statistics.mean(v[2] for v in pooled),
        }
    summary = {'policy': policy, 'source': str(Path(run).resolve()), 'samples': len(seen),
               'counts': dict(counts), 'gaps': gaps, 'focus_gap': focus, 'windows': windows,
               'lookback': lookback, 'histograms': histograms, 'neighbors': neighbor_summary,
               'regions': region_summary, 'rates': rates.summary(), 'collector_configs': configs,
               'definitions': {
                   'gap': 'X actual intervening MASK positions; separate iff gap >= X. '
                          'New isolated fill requires nearest old token index distance >= X+1; '
                          'births and distance/neighbor cohorts also require this separation from prompt and full-window stop tokens.',
                   'occupied': 'Response regions containing filled tokens before final stop; prompt is not counted as a region.',
                   'recent': 'Current regions containing a token committed in previous W actual steps; merges collapse region counts.',
                   'candidate_assignment': 'A masked candidate belongs to every region it would join under the same X-gap rule. '
                                           'It can bridge multiple regions. Unassigned candidates join no old region.',
                   'maxima': 'Max eligible nonspecial candidate confidence. Recent = touched in previous 5 steps; '
                             'older = joins only untouched regions; unassigned = joins none. Empty groups have null maxima.',
                   'directions': 'Immediate right is index max(region)+1; skips are farther exterior fills still joining a region; '
                                 'internal holes lie between region endpoints. Leftmost unresolved means first eligible mask '
                                 'after the region left endpoint, including immediate right extension; excludes multi-region bridges. '
                                 'These are AR-like proxies, not causal explanations. Threshold commits are simultaneous, not ordered.',
                   'neighbors': 'Unique per-step immediate neighbors of valid revealed tokens, still masked next pass. '
                                'Co-committed neighbors excluded. Delta is max-p(after) minus max-p(before), even if token changes. '
                                'Also report probability change of the same previous token. Next special-only steps are retained.',
                   'birth_history': 'Track each committed seed position backward before gap-defined births. '
                                    'Top prediction may change; no confidence after a position is filled.',
                   'weighting': 'Signed histograms and progress curves give questions equal weight; neighbor histograms pool observations. '
                                'Rates include both pooled and equal-question percentages.',
                   'limitations': 'Retrospective stop filtering; unequal region sizes affect maxima; threshold batches prevent '
                                  'single-token attribution; no claim of causal mechanism or normality.',
               }}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def read_csv_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def plots(out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = json.loads((out/'summary.json').read_text())
    policy, focus = s['policy'], s['focus_gap']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    def save(fig, name):
        fig.savefig(out/(name+'.png'), dpi=170, bbox_inches='tight')
        fig.savefig(out/(name+'.pdf'), bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    h = {int(k): v for k, v in s['histograms']['signed_previous']['mean_per_problem_percent'].items()}
    for ax, zoom in zip(axes[:2], [None, 20]):
        xs = sorted(h)
        ax.bar(xs, [h[x] for x in xs], width=.85,
               color=['#df8b22' if abs(x) == 1 else '#397caf' for x in xs])
        if zoom:
            ax.set_xlim(-zoom-.5, zoom+.5)
            ax.set_xticks(range(-zoom, zoom+1, 2))
        ax.set_title('Previous-batch distance: '+('full range' if zoom is None else 'zoom'))
        ax.set_xlabel('Signed index distance (zero impossible)')
        ax.set_ylabel('Mean within-question % of commits')
        ax.grid(axis='y', alpha=.2)
    h = {int(k): v for k, v in s['histograms']['masked_gap_to_nearest_filled']['mean_per_problem_percent'].items()}
    axes[2].bar(sorted(h), [h[x] for x in sorted(h)], color='#397caf', width=.85)
    axes[2].set_xlim(-.5, 20.5)
    axes[2].set_xticks(range(0, 21, 2))
    axes[2].set_title('Gap to nearest filled token or prompt')
    axes[2].set_xlabel('Number of intervening MASK positions')
    axes[2].set_ylabel('Mean within-question % of valid commits')
    axes[2].grid(axis='y', alpha=.2)
    axes[2].text(.98, .94, f'Beyond 20 masks: {sum(v for k,v in h.items() if k>20):.2f}%',
                 transform=axes[2].transAxes, ha='right', va='top')
    fig.suptitle(f'{policy}: distance distributions (cropping does not renormalize)')
    save(fig, 'distances')
    pairs = read_csv_gz(out/'neighbors.csv.gz')
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for j, (label, minimum) in enumerate([('All reveals', 0), (f'New seed: >= {focus} separating masks', focus+1)]):
        selected = [r for r in pairs if int(r['max_adjacent_seed_distance']) >= minimum]
        if not selected:
            continue
        after = np.array([float(r['after']) for r in selected])
        delta = 100*np.array([float(r['delta']) for r in selected])
        same = 100*np.array([float(r['same_token_delta']) for r in selected if r['same_token_delta']])
        axes[j, 0].hist(after, bins=np.linspace(0, 1, 51), weights=np.ones(len(after))*100/len(after), color='#397caf')
        axes[j, 0].set_xlabel('Top1 probability AFTER reveal')
        axes[j, 0].axvline(.9, color='#c87012', ls='--', lw=1)
        axes[j, 1].hist(delta, bins=np.linspace(-100, 100, 81), weights=np.ones(len(delta))*100/len(delta), color='#397caf', alpha=.7, label='Change in max probability')
        if len(same):
            axes[j, 1].hist(same, bins=np.linspace(-100, 100, 81), weights=np.ones(len(same))*100/len(same), histtype='step', color='#b8443b', lw=1.5, label='Change for same prior token')
        axes[j, 1].axvline(0, color='black', lw=.7)
        axes[j, 1].set_xlabel('After - before (percentage points)')
        axes[j, 1].legend(fontsize=8)
        axes[j, 2].hexbin([float(r['before']) for r in selected], after, gridsize=35, extent=(0,1,0,1), bins='log', mincnt=1, cmap='Blues')
        axes[j, 2].plot([0,1],[0,1],color='#c87012',lw=1)
        axes[j, 2].set_xlabel('Before probability')
        axes[j, 2].set_ylabel('After probability')
        for ax in axes[j, :2]:
            ax.set_ylabel('% of paired neighbor observations')
            ax.grid(axis='y',alpha=.2)
        axes[j, 0].set_title(f'{label}\nn={len(selected):,}; median after={np.median(after):.3f}')
        axes[j, 1].set_title(f'Median top1-p change: {np.median(delta):+.2f} pp')
        axes[j, 2].set_title('Each observation before vs after (log counts)')
    fig.suptitle(f'{policy}: immediate masked neighbors; co-committed neighbors excluded')
    save(fig, 'neighbor_confidence')
    frames = []
    with gzip.open(out/'frames.jsonl.gz', 'rt') as f:
        for line in f:
            frame = json.loads(line)
            if frame['gap'] == focus:
                frames.append(frame)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    def curve(ax, field, label, color):
        bins = defaultdict(lambda: defaultdict(list))
        for frame in frames:
            value = field(frame)
            if value is not None:
                bins[min(19, int(frame['progress']*20))][frame['sample']].append(value)
        xs=[]; means=[]; low=[]; high=[]
        for b, samples in sorted(bins.items()):
            values = [statistics.mean(v) for v in samples.values()]
            xs.append((b+.5)*5); means.append(statistics.mean(values))
            low.append(float(np.percentile(values,25))); high.append(float(np.percentile(values,75)))
        ax.plot(xs,means,label=label,color=color)
        ax.fill_between(xs,low,high,alpha=.12,color=color)
    curve(axes[0], lambda f:f['occupied'], 'Occupied regions', '#397caf')
    curve(axes[0], lambda f:f['recent']['5'], 'Touched in previous 5 steps', '#dd8925')
    curve(axes[0], lambda f:f['recent']['10'], 'Touched in previous 10 steps', '#429575')
    axes[0].set_xlabel('Response positions filled (%)')
    axes[0].set_ylabel('Regions; equal-question mean + IQR band')
    axes[0].legend(fontsize=8)
    axes[0].set_title(f'Regions separated by >= {focus} MASK tokens')
    gaps = sorted(map(int,s['regions']))
    axes[1].plot(gaps,[s['regions'][str(g)]['occupied_mean_per_problem'] for g in gaps],marker='o',label='Mean occupied regions')
    axes[1].set_xticks(gaps)
    axes[1].set_xlabel('Minimum separating MASK tokens X')
    axes[1].set_ylabel('Equal-question mean region count')
    axes[1].set_title('Sensitivity to region definition')
    for kind,color in [('recent','#dd8925'),('older','#397caf'),('unassigned','#9d4b8a')]:
        curve(axes[2], lambda f,k=kind:f['maxima'][k], kind, color)
    axes[2].set_xlabel('Response positions filled (%)')
    axes[2].set_ylabel('Maximum candidate top1 probability')
    axes[2].set_ylim(0,1.02)
    axes[2].legend(fontsize=8)
    axes[2].set_title('Candidate maxima by region activity\nEmpty candidate groups omitted')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(f'{policy}: region occupancy, recent activity and confidence')
    save(fig, 'region_activity')
    fig, axes = plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    cats=['immediate_right','immediate_left','internal_hole','right_skip','left_skip','bridge_multiple_regions']
    vals=[s['rates'].get(f'gap_{focus}/recent5_commits/{k}',{}).get('percent') or 0 for k in cats]
    axes[0].barh([c.replace('_',' ') for c in cats],vals,color='#397caf')
    axes[0].invert_yaxis()
    axes[0].set_xlabel('% of commits joining a recently touched region')
    axes[0].set_title('How AR-like is work on recent regions?')
    histories=read_csv_gz(out/'birth_history.csv.gz')
    by_offset=defaultdict(list)
    for r in histories:by_offset[int(r['offset'])].append(float(r['confidence']))
    offsets=sorted(by_offset)
    if offsets:
        axes[1].plot(offsets,[statistics.median(by_offset[k]) for k in offsets],marker='o',label='Median')
        axes[1].fill_between(offsets,[np.percentile(by_offset[k],25) for k in offsets],
                             [np.percentile(by_offset[k],75) for k in offsets],alpha=.2,label='IQR')
    axes[1].set_ylim(0,1.02)
    axes[1].set_xlabel('Steps before seed commitment (0 = commit)')
    axes[1].set_ylabel('Top1 probability at the SAME seed position')
    axes[1].set_title('Was the new-region seed already confident?\nDifferent offsets can have different sample counts')
    axes[1].legend()
    fig.suptitle(f'{policy}: X={focus} masks; activity window=5 steps')
    save(fig,'region_behavior')
    # A full time series for a concrete saved question, rather than only averaged curves.
    if frames:
        sample = '00028' if any(f['sample']=='00028' for f in frames) else frames[0]['sample']
        selected=[f for f in frames if f['sample']==sample]
        fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,constrained_layout=True)
        for kind,color in [('recent','#dd8925'),('older','#397caf'),('unassigned','#9d4b8a')]:
            axes[0].plot([f['step'] for f in selected],[np.nan if f['maxima'][kind] is None else f['maxima'][kind] for f in selected],label=kind,color=color,lw=1)
        axes[0].set_ylabel('Max top1 probability');axes[0].legend();axes[0].set_ylim(0,1.02)
        for frame in selected:
            for lo,hi in frame['regions']:
                axes[1].plot([frame['step'],frame['step']],[lo,hi],color='#397caf',alpha=.5,lw=2)
                if lo==hi:axes[1].plot(frame['step'],lo,'.',color='#397caf',ms=2)
            for group in frame['new_regions']:
                axes[1].scatter([frame['step']]*len(group),group,color='#d95a32',s=12)
        axes[1].set_xlabel('Actual decoding step');axes[1].set_ylabel('Response token position')
        axes[1].invert_yaxis()
        axes[1].set_title('Filled-region spans (blue); isolated new-group commitments (orange)')
        fig.suptitle(f'{policy}, sample {sample}, X={focus} masks; recently touched = previous 5 steps')
        save(fig,'example_timeline')
    links=['distances','neighbor_confidence','region_activity','region_behavior','example_timeline']
    report=['# Region dynamics: '+policy,'',*[f'- **{k}:** {v}' for k,v in s['definitions'].items()],'']
    for name in links:report += [f'## {name.replace("_"," ")}', '',f'![{name}]({name}.png)','']
    (out/'report.md').write_text('\n'.join(report)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gaps',type=int,nargs='+',default=[1,2,3,4,8,16])
    parser.add_argument('--focus-gap',type=int,default=4)
    parser.add_argument('--lookback',type=int,default=10)
    parser.add_argument('--plots-only',action='store_true')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    if min(args.gaps)<1 or args.focus_gap not in args.gaps or args.lookback<1:
        parser.error('Positive gaps/lookback required; focus-gap must be among gaps')
    if args.plots_only:
        plots(args.out)
        return
    if args.run is None:parser.error('--run required unless --plots-only')
    if args.out.exists() and any(args.out.iterdir()):parser.error('Use a new/empty output folder')
    args.out.mkdir(parents=True,exist_ok=True)
    collect(args.run,args.out,sorted(set(args.gaps)),args.focus_gap,[1,5,10],args.lookback)
    if not args.no_plots:plots(args.out)
    print(f'Results: {args.out}',flush=True)


if __name__=='__main__':
    main()
