"""Read distant commits or confidence increases from saved CSVs and traces. No inference."""
import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

from .analyze import records
from .core import nearest


def esc(value):
    return html.escape(str(value), quote=True)


def fmt(value):
    return 'unavailable' if value is None else f'{value:.4f}'


def token_text(token_id, dictionary):
    return dictionary.get(str(token_id), {}).get('text', f'<id:{token_id}>')


def token_strip(ids, dictionary, mask_id, target, anchors, start=0, end=None):
    cells = []
    for pos in range(start, len(ids) if end is None else min(end, len(ids))):
        token_id = ids[pos]
        classes = ['cell']
        if pos == target: classes.append('target')
        elif pos in anchors: classes.append('anchor')
        if token_id == mask_id: classes.append('masked')
        text = '[MASK]' if token_id == mask_id else token_text(token_id, dictionary)
        cells.append(f'<span class="{" ".join(classes)}"><small>{pos}</small><code>{esc(text)}</code></span>')
    return '<div class="tokens">' + ''.join(cells) + '</div>'


def prediction_table(rows, anchors):
    heading = '<table><tr><th>Position</th><th>Prediction (quoted)</th><th>Probability</th><th>Change in top confidence</th><th>Signed distance</th><th>Filled this step?</th><th>Special?</th></tr>'
    body = []
    for row in rows:
        body.append('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in [
            row['position'], repr(row['text']), fmt(row['confidence']), fmt(row.get('delta_confidence')),
            nearest(row['position'], anchors)['signed_distance'], row['committed'], row['special']]) + '</tr>')
    return heading + ''.join(body) + '</table>'


def render_case(case):
    event, step, result, config = (case[k] for k in ['event', 'step', 'result', 'config'])
    target = int(event['position'])
    row = next(r for r in step['positions'] if r['position'] == target)
    anchors = step['previous_commits']
    dictionary = result['token_dictionary']
    mask_id = config['mask_id']
    nearby = sorted([r for r in step['positions'] if r['eligible'] and nearest(r['position'], anchors)['distance'] == 1],
                    key=lambda r: (-r['confidence'], r['position']))
    top = sorted([r for r in step['positions'] if r['eligible']], key=lambda r: (-r['confidence'], r['position']))[:8]
    alternatives = ', '.join(f'{token_text(tid, dictionary)!r}: {prob:.4f}'
                             for tid, prob in zip(row['topk_ids'], row['topk_probs']))
    before = list(step['state_ids'])
    for pos in anchors: before[pos] = mask_id
    distance = int(event['signed_distance'])
    sign = '+' if distance > 0 else ''
    policy = config['policy']
    if not row['committed']:
        reason = 'This prediction was NOT filled at this step. It is shown because its confidence increased at a distant position.'
    elif policy == 'threshold':
        reason = (f"Prediction passed the {config['commit_threshold']:.2f} threshold. Other positions may have been filled simultaneously."
                  if row['confidence'] >= config['commit_threshold'] else
                  'No eligible prediction passed the threshold; this was the highest-confidence fallback.')
    elif policy == 'top1':
        reason = 'Highest-confidence eligible position; ties are resolved by the lower position index.'
    else:
        reason = f'Chosen by the {policy} policy; do not assume it beat every adjacent prediction.'
    return f'''<article>
<h2>Sample {esc(result['sample_id'])}, step {step['step']}: {esc(repr(row['text']))} at position {target}, distance {sign}{distance}</h2>
<p><b>Question:</b> {esc(result['question'])}</p>
<p><b>Actual selection rule:</b> {esc(reason)}</p>
<p>Previous batch: {esc(anchors)}. This batch: {esc(step['commit_positions'])}. Distance to <b>any</b> already-filled token: {esc(row.get('nearest_filled_distance'))}.</p>
<p><b>Tokens revealed in the previous step:</b> {esc(', '.join(f'{pos}: {token_text(step["state_ids"][pos], dictionary)!r}' for pos in anchors))}</p>
<p>Current confidence: <b>{fmt(row['confidence'])}</b>; prior top confidence here: {fmt(row.get('previous_confidence'))}; change: {fmt(row.get('delta_confidence'))}.
Prior predicted token: {esc(repr(token_text(row['previous_token_id'], dictionary))) if row.get('previous_token_id') is not None else 'unavailable'};
prediction changed: {esc(row.get('prediction_changed'))}. Change in probability of the <b>prior predicted token</b>: {fmt(row.get('same_token_delta'))}.</p>
<h3>Inspected region BEFORE this step's fills</h3>
{token_strip(step['state_ids'], dictionary, mask_id, target, anchors, max(0, target-10), target+11)}
<p>Green border = inspected position (still masked here); orange = previous-step fills. All numbers are response token indices.</p>
<h3>Adjacent alternatives around the previous batch</h3>
{prediction_table(nearby, anchors) if nearby else '<p>No eligible adjacent alternatives.</p>'}
<details><summary>Eight highest-confidence eligible positions</summary>{prediction_table(top, anchors)}</details>
<details><summary>Top token alternatives at the inspected position</summary><pre>{esc(alternatives)}</pre></details>
<details><summary>Full state BEFORE this step's fills (what the model actually saw, after the prompt)</summary>
{token_strip(step['state_ids'], dictionary, mask_id, target, anchors)}</details>
<details><summary>State BEFORE the previous batch was revealed (reconstructed)</summary>
{token_strip(before, dictionary, mask_id, target, [])}</details>
<details><summary>Final text and outcome — retrospective, not available at this step</summary>
<p>Strict answer correct: {esc(result['correct_strict'])}</p>
<pre>{esc(result['answer'])}</pre>
<h3>Final token positions</h3>{token_strip(result['final_ids'], dictionary, mask_id, target, anchors)}
<h3>Reference answer</h3><pre>{esc(result['reference'])}</pre></details>
</article>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, help='Default: RUN/analysis/jumps')
    p.add_argument('--min-distance', type=int, default=10)
    p.add_argument('--max-distance', type=int)
    p.add_argument('--limit', type=int, default=30)
    p.add_argument('--per-problem', type=int, default=3)
    p.add_argument('--sort', choices=['distance', 'rise'], default='distance')
    p.add_argument('--event-kind', choices=['commit', 'remote_rise'], default='commit',
                   help='remote_rise includes predictions not committed immediately; limited to events saved by analyze --rise')
    p.add_argument('--min-rise', type=float, default=None, help='Absolute probability increase, e.g. 0.20 = 20 percentage points')
    p.add_argument('--min-confidence', type=float, default=0, help='Minimum current top candidate probability')
    p.add_argument('--same-token-only', action='store_true', help='Require the predicted token to be unchanged across the transition')
    p.add_argument('--require-local', action='store_true', help='Require an adjacent in-answer nonspecial option')
    args = p.parse_args()
    if args.min_distance < 2 or args.limit < 1 or args.per_problem < 1:
        p.error('min-distance must be >=2, and limits positive')
    if args.max_distance is not None and args.max_distance < args.min_distance:
        p.error('max-distance must be >=min-distance')
    if not 0 <= args.min_confidence <= 1 or (args.min_rise is not None and not 0 <= args.min_rise <= 1):
        p.error('min-confidence and min-rise must be in [0,1]')
    events_path = args.run / 'analysis/events.csv'
    with events_path.open(newline='', encoding='utf-8') as handle:
        candidates = []
        for row in csv.DictReader(handle):
            if row['kind'] != args.event_kind or row['primary_population'].lower() != 'true': continue
            distance = abs(int(row['signed_distance']))
            if distance < args.min_distance or (args.max_distance is not None and distance > args.max_distance): continue
            if args.require_local and row['answer_local_option_available'].lower() != 'true': continue
            if float(row['confidence']) < args.min_confidence: continue
            if args.min_rise is not None and (not row['delta_confidence'] or float(row['delta_confidence']) < args.min_rise): continue
            if args.same_token_only and row['prediction_changed'].lower() != 'false': continue
            candidates.append(row)
    def sort_key(row):
        score = abs(int(row['signed_distance'])) if args.sort == 'distance' else float(row['delta_confidence'] or '-inf')
        return (-score, row['sample_id'], int(row['step']), int(row['position']))
    selected, counts, used_steps = [], Counter(), set()
    for row in sorted(candidates, key=sort_key):
        sid, step = row['sample_id'], int(row['step'])
        if counts[sid] >= args.per_problem or (sid, step) in used_steps: continue
        selected.append(row)
        counts[sid] += 1
        used_steps.add((sid, step))
        if len(selected) == args.limit: break
    required = {}
    for row in selected: required.setdefault(row['sample_id'], set()).add(int(row['step']))
    loaded = {}
    for sid, needed in required.items():
        directory = args.run / 'samples' / sid
        result = json.loads((directory / 'result.json').read_text())
        found = set()
        config = None
        for record in records(directory / 'trace.jsonl.gz'):
            if record['type'] == 'header': config = record['config']
            if record['type'] == 'step' and record['step'] in needed:
                loaded[(sid, record['step'])] = {'step': record, 'result': result, 'config': config}
                found.add(record['step'])
            if found == needed: break
        if found != needed: raise ValueError(f'Missing trace steps for {sid}: {needed-found}')
    cases = [{'event': row, **loaded[(row['sample_id'], int(row['step']))]} for row in selected]
    out = args.out or args.run / 'analysis/jumps'
    out.mkdir(parents=True, exist_ok=True)
    meta = {'run': str(args.run), 'min_distance': args.min_distance, 'max_distance': args.max_distance,
            'sort': args.sort, 'require_local': args.require_local, 'qualifying_events': len(candidates),
            'event_kind': args.event_kind, 'min_rise': args.min_rise, 'min_confidence': args.min_confidence,
            'same_token_only': args.same_token_only,
            'selected_cases': len(cases), 'limit': args.limit, 'per_problem': args.per_problem,
            'selection': 'Primary population only; at most one case per step; not a representative sample'}
    (out / 'cases.json').write_text(json.dumps({'metadata': meta, 'cases': cases}, ensure_ascii=False, indent=2))
    content = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Distant confidence events</title><style>
body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 18px;color:#18212c;background:#f5f7fa}
article{background:white;border:1px solid #ccc;border-radius:10px;padding:22px;margin:24px 0}
pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border:1px solid #ddd;padding:8px;text-align:left}details{margin:18px 0}summary{cursor:pointer;font-weight:600}
.tokens{display:flex;flex-wrap:wrap;gap:5px;margin:14px 0}.cell{border:2px solid #ddd;padding:5px;border-radius:5px;min-width:25px}
.cell small{display:block;color:#555}.cell code{white-space:pre-wrap;overflow-wrap:anywhere}.masked{background:#f0f0f0;color:#777}
.target{border-color:#168452;background:#e5f7ed}.anchor{border-color:#dc8a13;background:#fff0d8}
</style><h1>Distant confidence events: actual saved states</h1>'''
    content += f'<p>{len(cases)} selected cases from {len(candidates)} qualifying {esc(args.event_kind)} events. Distances are in <b>tokens, not words</b>. Ranked by {esc(args.sort)}; these are illustrative extremes, not prevalence estimates.</p>'
    content += f'<p>Filters: minimum distance {args.min_distance}, maximum distance {esc(args.max_distance)}, minimum probability increase {esc(args.min_rise)}, minimum current confidence {args.min_confidence}, unchanged prediction required: {args.same_token_only}.</p>'
    if args.event_kind == 'remote_rise':
        content += '<p>This report only includes remote-rise events already saved by analysis (default minimum increase 0.15). Lowering this report filter cannot recover events excluded upstream.</p>'
    content += '<p>No new predictions were generated. Expand final text only after examining the masked context. For threshold decoding, distance refers to the nearest token in the previous batch.</p>'
    content += ''.join(render_case(case) for case in cases) if cases else '<p>No cases meet these filters. Try a smaller minimum distance or omit --require-local.</p>'
    (out / 'index.html').write_text(content + '</html>', encoding='utf-8')
    print(json.dumps(meta, indent=2))
    print(f"Open {out / 'index.html'}; share {out / 'cases.json'} for inspection.")


if __name__ == '__main__':
    main()
