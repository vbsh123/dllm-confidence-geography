"""Offline region/distance/confidence analysis. Standard library only; no inference.

Usage: python -m confidence_geography.region_stats --run PATH_OR_ZIP --out OUTPUT
You can also execute this file directly without installing the package.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import io
import json
from pathlib import Path
import statistics
import zipfile


DEFINITIONS = {
    'valid': 'Nonspecial token before the final first stop (retrospective filter).',
    'distance': 'Token index difference, not mask count. Distance 4 to the nearest '
                'filled token implies 3 intervening masks. Prompt boundary is index -1.',
    'seed': 'Valid committed token at least min_distance from ALL pre-step filled '
            'response positions and the prompt boundary. Same-batch fills are not anchors.',
    'region': 'Maximal consecutive group of valid same-batch commitments, ALL qualifying as '
              'seeds. Groups reaching closer to old text through simultaneous fills are excluded.',
    'continuation': 'Valid commitment immediately beside an exterior end on the very '
                    'next actual decoding step. No skipping special-only steps.',
    'older': 'All filled positions after the creation batch except its strict new '
             'regions, plus prompt boundary. Thus includes same-batch extensions of old text.',
    'current': 'Contiguous pre-step filled components containing valid previous-batch '
               'commitments. For threshold, there can be multiple current regions.',
    'neighbors': 'Immediate positions -1 and +1 of a seed; confidence pairs require '
                 'a nonspecial, eligible, pre-stop prediction BEFORE reveal and a '
                 'still-masked position on the next pass. Co-committed neighbors are excluded.',
    'crossing': 'Before top confidence < cutoff and after top confidence >= cutoff, '
                'with nonspecial after prediction. same_token crossings instead track '
                'the probability of the PREVIOUSLY predicted token.',
    'weighting': 'Every rate includes numerator/denominator and pooled percentage. '
                 'Mean-per-problem percentages give equal weight to questions with '
                 'a nonzero denominator; do not mix them with pooled percentages.',
    'limits': 'Descriptive trace analysis, not a causal experiment or evidence of '
              'correctness/independence. Filled-token confidence is not recorded. '
              'Different policies have different trajectories.',
}


def components(positions):
    result = []
    for p in sorted(set(positions)):
        if not result or p != result[-1][-1] + 1:
            result.append([p])
        else:
            result[-1].append(p)
    return result


def compact(row):
    return {k: row.get(k) for k in
            ('position', 'token_id', 'text', 'confidence', 'special', 'category')}


class Rates:
    """Store denominators alongside every count, including conditional populations."""
    def __init__(self):
        self.data = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    def add(self, name, sid, yes, total=1):
        self.data[name][sid][0] += int(yes)
        self.data[name][sid][1] += int(total)

    def summary(self):
        result = {}
        for name, samples in sorted(self.data.items()):
            n = sum(v[0] for v in samples.values())
            d = sum(v[1] for v in samples.values())
            percentages = [100*a/b for a, b in samples.values() if b]
            result[name] = {
                'numerator': n, 'denominator': d,
                'percent': 100*n/d if d else None,
                'mean_per_problem_percent': statistics.mean(percentages) if percentages else None,
                'problems_with_denominator': len(percentages),
            }
        return result


class Source:
    """Stream gzip members directly from ZIP; never extract or load all traces."""
    def __init__(self, path):
        self.path = Path(path)
        self.archive = zipfile.ZipFile(path) if self.path.is_file() else None
        if self.archive:
            self.names = sorted(n for n in self.archive.namelist()
                                if n.endswith('/trace.jsonl.gz'))
        else:
            self.names = sorted(str(p) for p in self.path.rglob('trace.jsonl.gz'))
        if not self.names:
            self.close()
            raise ValueError(f'No completed trace.jsonl.gz files in {path}')

    def close(self):
        if self.archive:
            self.archive.close()

    def raw(self, name):
        return self.archive.open(name) if self.archive else open(name, 'rb')

    def result(self, name):
        with self.raw(name.replace('trace.jsonl.gz', 'result.json')) as f:
            return json.load(f)

    def records(self, name):
        with self.raw(name) as raw, gzip.GzipFile(fileobj=raw) as gz:
            with io.TextIOWrapper(gz, encoding='utf-8') as f:
                for line in f:
                    yield json.loads(line)


def snapshot(step, result, config, minimum):
    rows = {r['position']: r for r in step['positions']}
    chosen = {r['position'] for r in rows.values() if r['committed']}
    if chosen != set(step['commit_positions']):
        raise ValueError('Commit flags and commit_positions disagree')
    end = result['answer_token_length']
    valid = {p for p in chosen if p < end and not rows[p]['special']}
    filled = {p for p, t in enumerate(step['state_ids']) if t != config['mask_id']}
    if filled & chosen:
        raise ValueError('Trace re-commits an already-filled position')
    anchors = filled | {-1}
    distance = lambda p: min(abs(p-q) for q in anchors)
    seeds = {p for p in valid if distance(p) >= minimum}
    regions = [g for g in components(valid) if set(g) <= seeds]
    previous = step['previous_commits']
    dictionary = result['token_dictionary']
    valid_previous = {p for p in previous if p < end and
                      not dictionary[str(step['state_ids'][p])]['special']}
    active = [g for g in components(filled) if valid_previous.intersection(g)]
    boundary = {p for g in active for p in (g[0]-1, g[-1]+1)
                if p in rows and p < end and rows[p]['eligible']}
    alternatives = {p for p in boundary if not rows[p]['special']}
    return dict(step=step, rows=rows, chosen=chosen, valid=valid, filled=filled,
                seeds=seeds, regions=regions, active=active, boundary=boundary,
                alternatives=alternatives, distance=distance,
                all_previous_valid=bool(previous) and set(previous) == valid_previous)


def neighbor_pair(before, after, position, end):
    b, a = before['rows'].get(position), after['rows'].get(position)
    if b is None or a is None or position >= end or b['special'] or not b['eligible']:
        return None
    same_probability = a.get('previous_token_probability_now')
    if same_probability is None and a['token_id'] == b['token_id']:
        same_probability = a['confidence']
    return {'position': position, 'before': compact(b), 'after': compact(a),
            'same_token_probability_after': same_probability,
            'same_token_delta': same_probability-b['confidence'] if same_probability is not None else None,
            'prediction_changed': b['token_id'] != a['token_id']}


def continuation(before, after):
    new_positions = {p for g in before['regions'] for p in g}
    older = (before['filled'] | before['chosen']) - new_positions | {-1}
    continued = [bool({g[0]-1, g[-1]+1} & after['valid']) for g in before['regions']]
    old_extended = any(p-1 in older or p+1 in older for p in after['valid'])
    new_extended = any(continued)
    category = ('new_and_old' if old_extended else 'new_only') if new_extended else (
        'old_only' if old_extended else 'neither')
    return continued, old_extended, category


def top1_destination(before, after, seed, minimum):
    if len(after['chosen']) != 1:
        raise ValueError('top1 trace has a non-singleton batch')
    p = next(iter(after['chosen']))
    if p not in after['valid']:
        return 'special_or_after_final_stop'
    new_distance, old_distance = abs(p-seed), before['distance'](p)
    if new_distance == 1:
        return 'continue_new_region'
    if old_distance == 1:
        return 'return_old_region'
    if min(new_distance, old_distance) >= minimum:
        return 'another_new_region'
    return 'nonadjacent_intermediate_gap'


class Analyzer:
    def __init__(self, minimum, cutoffs, rise, emit):
        self.minimum, self.cutoffs, self.rise, self.emit = minimum, cutoffs, rise, emit
        self.counts = Counter()
        self.rates = Rates()
        self.histograms = defaultdict(lambda: defaultdict(Counter))
        self.configs = []
        self.policy = None
        self.seen = set()

    def step(self, sid, state, result, config):
        c, r = self.counts, self.rates
        c['steps'] += 1
        c['all_commits'] += len(state['chosen'])
        c['valid_commits'] += len(state['valid'])
        c['seed_commits'] += len(state['seeds'])
        c['strict_regions_created'] += len(state['regions'])
        c['strict_region_creation_batches'] += bool(state['regions'])
        c['seed_batches'] += bool(state['seeds'])
        r.add('steps/multiple_commits', sid, len(state['chosen']) > 1)
        if config['policy'] == 'threshold':
            fallback = max(state['rows'][p]['confidence'] for p in state['chosen']) < config['commit_threshold']
            r.add('steps/fallback', sid, fallback)
            if state['seeds']:
                r.add('seed_batches/fallback', sid, fallback)
        r.add('valid_commits/seeds', sid, len(state['seeds']), len(state['valid']))
        r.add('valid_commits/strict_region_creation_tokens', sid,
              sum(map(len, state['regions'])), len(state['valid']))
        step = state['step']
        if state['seeds']:
            r.add('seed_batches/co_commit_old_boundary', sid,
                  any(state['distance'](p) == 1 for p in state['valid']))
        if state['all_previous_valid']:
            anchors = step['previous_commits']
            local_available = any(row['eligible'] and not row['special'] and
                                  p < result['answer_token_length'] and
                                  min(abs(p-a) for a in anchors) == 1
                                  for p, row in state['rows'].items())
            for p in state['valid']:
                a = min(anchors, key=lambda a: (abs(p-a), a))
                d = p-a
                for population in ['all_primary'] + (['local_option_available'] if local_available else []):
                    self.histograms[population][sid][d] += 1
                    for label, yes in [('minus_one', d == -1), ('plus_one', d == 1),
                                       ('nonadjacent', abs(d) > 1)]:
                        r.add(f'distances/{population}/{label}', sid, yes)
        if state['active']:
            r.add('current_region_steps/open_strict_region', sid, bool(state['regions']))
            if state['regions']:
                extended = bool(state['valid'] & state['boundary'])
                available = bool(state['alternatives'])
                r.add('opening_with_current/no_current_extension', sid, not extended)
                r.add('opening_with_current/extension_option_available', sid, available)
                if available:
                    r.add('opening_with_current_option/no_current_extension', sid, not extended)
                if not extended:
                    r.add('switches/current_alternative_ge90', sid,
                          any(state['rows'][p]['confidence'] >= .9 for p in state['alternatives']))
        for p in sorted(state['seeds']):
            row = state['rows'][p]
            previous = row.get('previous_confidence')
            if previous is not None:
                r.add('seeds/prior_confidence_ge90', sid, previous >= .9)
            delta = row.get('delta_confidence')
            if delta is not None:
                r.add('seeds/top_confidence_rise', sid, delta >= self.rise)
            self.emit({'type': 'seed', 'sample': sid, 'step': step['step'],
                       'token': compact(row), 'distance_any': state['distance'](p),
                       'previous_confidence': previous, 'delta_confidence': delta})

    def confidence_stats(self, sid, prefix, pairs):
        r = self.rates
        for pair in pairs:
            delta = pair['same_token_delta']
            if delta is not None:
                for name, yes in [('same_token_rise_gt1pp', delta > .01),
                                  ('same_token_drop_gt1pp', delta < -.01),
                                  ('same_token_within1pp', abs(delta) <= .01),
                                  ('same_token_large_rise', delta >= self.rise)]:
                    r.add(prefix+'/'+name, sid, yes)
            r.add(prefix+'/prediction_changed', sid, pair['prediction_changed'])
        for cutoff in self.cutoffs:
            base = prefix+f'/cutoff_{cutoff:g}'
            crossing = []
            ready = []
            for pair in pairs:
                b, a = pair['before'], pair['after']
                below = b['confidence'] < cutoff
                above = not a['special'] and a['confidence'] >= cutoff
                crosses = below and above
                crossing.append(crosses)
                ready.append(above)
                r.add(base+'/neighbors_crossing', sid, crosses)
                if below:
                    r.add(base+'/below_before_crossing', sid, crosses)
                    r.add(base+'/below_before_still_below', sid, a['confidence'] < cutoff)
                    r.add(base+'/below_before_special_ge_cutoff', sid,
                          a['special'] and a['confidence'] >= cutoff)
                if pair['same_token_probability_after'] is not None:
                    same_cross = below and pair['same_token_probability_after'] >= cutoff
                    r.add(base+'/same_token_crossing', sid, same_cross)
                if crosses:
                    r.add(base+'/crossings_prediction_changed', sid, pair['prediction_changed'])
            if pairs:
                r.add(base+'/events_any_ready', sid, any(ready))
                r.add(base+'/events_any_crossing', sid, any(crossing))
                r.add(base+'/events_no_ready_before_any_ready_after', sid,
                      all(p['before']['confidence'] < cutoff for p in pairs) and any(ready))
                if len(pairs) == 2:
                    r.add(base+'/two_neighbor_events_both_ready', sid, all(ready))

    def transition(self, sid, b, a, result):
        if a['step']['step'] != b['step']['step']+1:
            raise ValueError('Nonconsecutive trace steps; cannot analyze next-step transitions')
        r, c = self.rates, self.counts
        if b['regions']:
            continued, old, category = continuation(b, a)
            c['region_creation_batches_with_next_step'] += 1
            for name in ['new_only', 'new_and_old', 'old_only', 'neither']:
                r.add('creation_batches_next/'+name, sid, category == name)
            r.add('creation_batches_next/continues_new', sid, any(continued))
            if any(continued):
                r.add('continuing_batches/no_older_extension', sid, not old)
            r.add('creation_batches_next/opens_another_without_extending_any', sid,
                  category == 'neither' and bool(a['regions']))
            r.add('created_regions/continued_next', sid, sum(continued), len(continued))
            r.add('created_regions/continued_without_older_extension', sid,
                  sum(continued) if not old else 0, len(continued))
            continued_tokens = sum(len(g) for g, yes in zip(b['regions'], continued) if yes)
            c['creation_tokens_in_continued_regions'] += continued_tokens
            self.emit({'type': 'region_transition', 'sample': sid, 'step': b['step']['step'],
                       'regions': b['regions'], 'continued': continued,
                       'next_category': category,
                       'next_commits': [compact(a['rows'][p]) for p in sorted(a['chosen'])]})
        if not b['seeds']:
            return
        end = result['answer_token_length']
        if self.policy == 'top1':
            for seed in b['seeds']:
                dest = top1_destination(b, a, seed, self.minimum)
                valid_next = dest != 'special_or_after_final_stop'
                r.add('top1_seed_next/valid_next', sid, valid_next)
                pairs = [pair for p in (seed-1, seed+1)
                         if (pair := neighbor_pair(b, a, p, end)) is not None]
                self.confidence_stats(sid, 'top1_neighbors_all_next', pairs)
                if valid_next:
                    for name in ['continue_new_region', 'return_old_region',
                                 'another_new_region', 'nonadjacent_intermediate_gap']:
                        r.add('top1_seed_next_valid/'+name, sid, dest == name)
                    self.confidence_stats(sid, 'top1_neighbors_valid_next', pairs)
                    if dest != 'continue_new_region':
                        self.confidence_stats(sid, 'top1_neighbors_when_went_elsewhere', pairs)
                    boundary = [p for p, row in b['rows'].items() if row['eligible'] and
                                not row['special'] and p < end and b['distance'](p) == 1]
                    if boundary:
                        best = max(boundary, key=lambda p: b['rows'][p]['confidence'])
                        old_pair = neighbor_pair(b, a, best, end)
                        if old_pair:
                            self.confidence_stats(sid, 'top1_best_old_boundary', [old_pair])
                self.emit({'type': 'top1_seed_transition', 'sample': sid,
                           'step': b['step']['step'], 'seed': seed, 'destination': dest,
                           'neighbors': pairs,
                           'next_commits': [compact(a['rows'][p]) for p in sorted(a['chosen'])]})
        else:
            positions = {p+d for p in b['seeds'] for d in (-1, 1)}
            pairs = [pair for p in sorted(positions)
                     if (pair := neighbor_pair(b, a, p, end)) is not None]
            self.confidence_stats(sid, 'batch_remaining_seed_neighbors', pairs)
            fallback = max(b['rows'][p]['confidence'] for p in b['chosen']) < self.commit_threshold
            if fallback:
                r.add('seed_fallback_batches/next_multiple_commits', sid, len(a['chosen']) > 1)
            self.emit({'type': 'batch_seed_transition', 'sample': sid,
                       'step': b['step']['step'], 'seeds': sorted(b['seeds']),
                       'neighbors': pairs})

    def sample(self, result, records):
        sid = str(result['sample_id'])
        if sid in self.seen:
            raise ValueError(f'Duplicate sample {sid}; pass one policy/run at a time')
        self.seen.add(sid)
        previous = None
        initial_continued_tokens = self.counts['creation_tokens_in_continued_regions']
        initial_valid_commits = self.counts['valid_commits']
        header = next(records)
        if header['type'] != 'header':
            raise ValueError('Trace must start with header')
        config = header['config']
        if self.policy is not None and self.policy != config['policy']:
            raise ValueError('Mixed policies: pass top1 and threshold separately')
        self.policy = config['policy']
        self.commit_threshold = config.get('commit_threshold', .9)
        if config not in self.configs:
            self.configs.append(config)
        for step in records:
            if step['type'] != 'step':
                continue
            state = snapshot(step, result, config, self.minimum)
            if self.policy == 'top1' and len(state['chosen']) != 1:
                raise ValueError('top1 must commit exactly one token')
            if previous is not None:
                self.transition(sid, previous, state, result)
            self.step(sid, state, result, config)
            previous = state
        if previous is None:
            raise ValueError(f'No steps for sample {sid}')
        self.counts['regions_without_next_step'] += len(previous['regions'])
        self.counts['seed_commits_without_next_step'] += len(previous['seeds'])
        self.rates.add('valid_commits/creation_tokens_in_continued_regions', sid,
                       self.counts['creation_tokens_in_continued_regions']-initial_continued_tokens,
                       self.counts['valid_commits']-initial_valid_commits)
        self.counts['samples'] += 1

    def summary(self):
        rates = self.rates.summary()
        return {'schema_version': 1, 'policy': self.policy,
                'analysis_parameters': {'min_distance': self.minimum,
                                        'cutoffs': self.cutoffs, 'large_rise': self.rise},
                'definitions': DEFINITIONS, 'counts': dict(self.counts), 'rates': rates,
                'collector_configs': self.configs}


def save_outputs(out, analyzer, source):
    summary = analyzer.summary()
    summary['input'] = str(source.path.resolve())
    summary['trace_count'] = len(source.names)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    fields = ['metric', 'numerator', 'denominator', 'percent',
              'mean_per_problem_percent', 'problems_with_denominator']
    with (out/'rates.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, rate in summary['rates'].items():
            writer.writerow({'metric': name, **rate})
    with (out/'signed_distances.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['population', 'signed_distance', 'count', 'pooled_percent',
                         'mean_per_problem_percent', 'problems'])
        for population, samples in analyzer.histograms.items():
            pooled = sum(samples.values(), Counter())
            total = pooled.total()
            for d in sorted(pooled):
                mean = statistics.mean(100*c[d]/c.total() for c in samples.values())
                writer.writerow([population, d, pooled[d], 100*pooled[d]/total, mean, len(samples)])
    lines = [f'# Saved-trace analysis: {analyzer.policy}', '',
             'No model loading or inference. Percentages below are pooled; rates.csv also '
             'includes equal-question means. Every row states its denominator.', '']
    lines += [f'- **{k}:** {v}' for k, v in DEFINITIONS.items()]
    lines += ['', '## Counts', ''] + [f'- {k}: {v:,}' for k, v in summary['counts'].items()]
    lines += ['', '## Rates', '', '| Metric | Count / denominator | Percent |', '|---|---:|---:|']
    for name, rate in summary['rates'].items():
        pct = f"{rate['percent']:.2f}%" if rate['percent'] is not None else 'N/A'
        lines.append(f"| {name} | {rate['numerator']:,} / {rate['denominator']:,} | {pct} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='One policy directory or ZIP')
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    parser.add_argument('--min-distance', type=int, default=4)
    parser.add_argument('--cutoffs', type=float, nargs='+', default=[.85, .9])
    parser.add_argument('--rise', type=float, default=.15)
    parser.add_argument('--no-events', action='store_true', help='Omit detailed event JSONL')
    args = parser.parse_args()
    if args.min_distance < 2 or not all(0 < c < 1 for c in args.cutoffs) or not 0 < args.rise < 1:
        parser.error('min-distance must be >=2; cutoffs and rise must lie between 0 and 1')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Output directory must be empty; choose a new analysis folder')
    args.out.mkdir(parents=True, exist_ok=True)
    source = Source(args.run)
    event_path = args.out/'events.jsonl.gz.partial'
    try:
        with gzip.open(event_path, 'wt', encoding='utf-8') as events:
            def emit(event):
                if not args.no_events:
                    events.write(json.dumps(event, ensure_ascii=False)+'\n')
            analyzer = Analyzer(args.min_distance, sorted(set(args.cutoffs)), args.rise, emit)
            for i, name in enumerate(source.names, 1):
                analyzer.sample(source.result(name), source.records(name))
                if i % 10 == 0 or i == len(source.names):
                    print(f'Analyzed {i}/{len(source.names)} saved traces', flush=True)
            summary = save_outputs(args.out, analyzer, source)
        if args.no_events:
            event_path.unlink()
        else:
            event_path.replace(args.out/'events.jsonl.gz')
    finally:
        source.close()
    print(f"Policy: {summary['policy']}; report: {args.out/'report.md'}")
    for name in ['created_regions/continued_next', 'creation_batches_next/new_only',
                 'top1_neighbors_when_went_elsewhere/cutoff_0.9/events_any_ready',
                 'top1_neighbors_valid_next/cutoff_0.85/below_before_crossing']:
        if name in summary['rates']:
            rate = summary['rates'][name]
            pct = f"{rate['percent']:.2f}%" if rate['percent'] is not None else 'N/A'
            print(f"{name}: {rate['numerator']}/{rate['denominator']} = {pct}")


if __name__ == '__main__':
    main()
