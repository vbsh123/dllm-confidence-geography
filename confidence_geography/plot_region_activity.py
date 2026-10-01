"""Questions 3.1/3.2: region switching, batch activity, and local fill order.

Saved traces only. The complete analysis and plotting is in
generate_region_activity_plots(); it runs only when explicitly called.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

if __package__:
    from .region_stats import Source  # File/archive reading only.
else:
    from region_stats import Source


def generate_region_activity_plots(run, out, mask_gap=4):
    """Read saved traces and generate graphs for 3.1 and 3.2, in one function.

    SAME SPATIAL DEFINITION FOR BOTH POLICIES:
      Separate regions have >=mask_gap consecutive MASKs between filled tokens.
      With mask_gap=4, filled positions 0 and 4 share a region (3 masks between);
      positions 0 and 5 are separate (4 masks between). Regions may have holes.
      Work on the response before its final stop; the prompt is not a region.

    3.1 TOP1: compare the newly committed token's region with the region of the
      previous valid commitment. Count stay / jump to older region / new region
      / merge. Merges are separate rather than arbitrarily called stay or jump.

    3.1 THRESHOLD: count regions receiving >=1 valid commitment in each batch,
      and commitments per such region. Evaluate regions AFTER the batch so
      simultaneous new tokens have a defined region. Batch-created merges can
      turn several old regions into one; also record old regions touched.

    3.2 BOTH: for each updated region with exactly ONE pre-existing predecessor,
      check whether its new commitments fill every previously masked position
      from its old left endpoint through the rightmost new commitment. This is
      a local left-to-right prefix, not proof of an autoregressive mechanism.
      Left expansion, skipped holes, births, and merges are identified separately.
      A threshold batch has no within-batch order; we test its SET of commitments.

    No analysis of causal 'why' is attempted here (question 3.3).
    """
    run, out = Path(run), Path(out)
    if mask_gap < 1:
        raise ValueError('mask_gap must be at least one intervening MASK')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new/empty output directory')
    out.mkdir(parents=True, exist_ok=True)

    totals = Counter()
    top1_transitions = Counter()
    active_region_counts = Counter()
    tokens_per_active_region = Counter()
    local_order_counts = Counter()
    per_question = {}
    policy = None
    source = Source(run)

    # 1. Read each question independently. Each step contains the state BEFORE
    #    its commitments, plus the positions/token IDs committed by that step.
    try:
        with gzip.open(out/'region_steps.jsonl.gz', 'wt', encoding='utf-8') as events:
            for number, name in enumerate(source.names, 1):
                result = source.result(name)
                sid = str(result['sample_id'])
                if sid in per_question:
                    raise ValueError('Duplicate sample: use one policy/run at a time')
                question_counts = Counter()
                answer_end = result['answer_token_length']
                dictionary = result['token_dictionary']
                for record in source.records(name):
                    if record['type'] == 'header':
                        config = record['config']
                        mask_id = config['mask_id']
                        if config['policy'] not in ('top1', 'threshold'):
                            raise ValueError('This analysis supports top1 and threshold')
                        if policy is not None and policy != config['policy']:
                            raise ValueError('Use one decoding policy per input')
                        policy = config['policy']
                        continue
                    if record['type'] != 'step':
                        continue

                    totals['all_steps'] += 1
                    before_state = record['state_ids']
                    rows = {r['position']: r for r in record['positions']}
                    committed = record['commit_positions']
                    if policy == 'top1' and len(committed) != 1:
                        raise ValueError('Top1 must commit exactly one token per step')
                    valid_commits = [p for p in committed
                                     if p < answer_end and not rows[p]['special']]
                    if not valid_commits:
                        totals['steps_without_valid_commit'] += 1
                        continue
                    totals['valid_commit_steps'] += 1
                    totals['valid_commits'] += len(valid_commits)
                    question_counts['valid_commit_steps'] += 1

                    # 2. Reconstruct the state AFTER this step, without running
                    #    a model. Include all commits, even special tokens, in
                    #    the state; activity counts use only valid answer tokens.
                    after_state = before_state.copy()
                    for position in committed:
                        if before_state[position] != mask_id:
                            raise ValueError('A commitment was already filled')
                        after_state[position] = rows[position]['token_id']

                    # 3. Build spatial regions before AND after the batch.
                    #    Consecutive filled positions have only MASKs between.
                    #    Gap >=4 splits them; gap 0,1,2,3 keeps them together.
                    region_states = []
                    for state in (before_state, after_state):
                        groups = []
                        for position in range(answer_end):
                            if state[position] == mask_id:
                                continue
                            if not groups or position - groups[-1][-1] - 1 >= mask_gap:
                                groups.append([position])
                            else:
                                groups[-1].append(position)
                        owner = {position: region_id
                                 for region_id, group in enumerate(groups)
                                 for position in group}
                        region_states.append((groups, owner))
                    before_regions, before_owner = region_states[0]
                    after_regions, after_owner = region_states[1]

                    # 4. Find the active regions: those receiving commitments
                    #    NOW. Count them once each, even if they receive many.
                    active_after = sorted({after_owner[p] for p in valid_commits})
                    active_region_counts[len(active_after)] += 1
                    question_counts[f'active_regions_{len(active_after)}'] += 1
                    updates = []
                    old_regions_touched = set()
                    for after_id in active_after:
                        group = after_regions[after_id]
                        group_commits = sorted(p for p in valid_commits if after_owner[p] == after_id)
                        tokens_per_active_region[len(group_commits)] += 1

                        # Determine lineage by shared filled positions, not by
                        # numeric region IDs (which change after births/merges).
                        predecessors = sorted({before_owner[p] for p in group if p in before_owner})
                        old_regions_touched.update(predecessors)
                        update = {
                            'after_region': [group[0], group[-1]],
                            'commits': group_commits,
                            'predecessor_ids': predecessors,
                            'predecessor_regions': [[before_regions[i][0], before_regions[i][-1]]
                                                    for i in predecessors],
                        }

                        # 5. Question 3.2: did this update follow a local
                        #    left-to-right prefix, or leave earlier holes?
                        #    Births have no old frontier; merges have several.
                        if not predecessors:
                            update['order'] = 'new_region_excluded'
                            totals['new_region_updates'] += 1
                        elif len(predecessors) > 1:
                            update['order'] = 'merge_excluded'
                            totals['merge_updates'] += 1
                        else:
                            old_group = before_regions[predecessors[0]]
                            old_left, old_right = old_group[0], old_group[-1]
                            rightmost_commit = max(group_commits)
                            required_masks = [p for p in range(old_left, rightmost_commit+1)
                                              if before_state[p] == mask_id]
                            update['required_masks_for_prefix'] = required_masks
                            if min(group_commits) < old_left:
                                order = 'left_expansion'
                            elif group_commits == required_masks:
                                order = 'left_to_right_prefix'
                            else:
                                order = 'skips_earlier_masks'
                            update['order'] = order
                            local_order_counts[order] += 1
                            question_counts['order_'+order] += 1

                            # Extra context: appending at the right edge can
                            # still skip an internal hole, so keep it separate.
                            update['immediate_right_committed'] = old_right+1 in group_commits
                            update['internal_hole_commits'] = [p for p in group_commits
                                                               if old_left < p < old_right]
                        updates.append(update)

                    # 6. Question 3.1 for TOP1: compare this commitment with the
                    #    previous step's commitment under the region definition.
                    #    Skip invalid previous anchors; never jump across steps
                    #    to invent an earlier valid anchor.
                    transition = None
                    if policy == 'top1':
                        previous = [p for p in record['previous_commits']
                                    if p < answer_end and before_state[p] != mask_id
                                    and not dictionary[str(before_state[p])]['special']]
                        if len(previous) == 1:
                            old_current_id = before_owner[previous[0]]
                            predecessors = updates[0]['predecessor_ids']
                            if len(predecessors) > 1:
                                transition = 'merge_regions'
                            elif old_current_id in predecessors:
                                transition = 'stay_same_region'
                            elif predecessors:
                                transition = 'jump_to_other_existing_region'
                            else:
                                transition = 'open_new_region'
                            top1_transitions[transition] += 1
                            question_counts['transition_'+transition] += 1
                        else:
                            totals['top1_without_valid_previous_anchor'] += 1

                    # Store the actual masks/commitments behind every decision
                    # so each classification can be inspected, not just averaged.
                    events.write(json.dumps({
                        'sample': sid, 'step': record['step'], 'mask_gap': mask_gap,
                        'previous_commits': record['previous_commits'],
                        'before_regions': [[g[0], g[-1]] for g in before_regions],
                        'occupied_after': len(after_regions),
                        'active_after': len(active_after),
                        'old_regions_touched': len(old_regions_touched),
                        'top1_transition': transition, 'updates': updates,
                    })+'\n')
                per_question[sid] = dict(question_counts)
                if number % 10 == 0 or number == len(source.names):
                    print(f'Read {number}/{len(source.names)} saved questions', flush=True)
    finally:
        source.close()

    # 7. Build percentages with explicit denominators. All are pooled, not
    #    averages of percentages per question. Counts are saved for each question.
    summary = {
        'policy': policy, 'input': str(run.resolve()), 'mask_gap': mask_gap,
        'questions': len(per_question), 'totals': dict(totals),
        'top1_transitions': dict(top1_transitions),
        'regions_receiving_commits_per_batch': dict(sorted(active_region_counts.items())),
        'commits_per_active_region': dict(sorted(tokens_per_active_region.items())),
        'local_order': dict(local_order_counts),
        'denominators': {
            'top1_transitions': top1_transitions.total(),
            'active_region_batches': active_region_counts.total(),
            'tokens_per_active_region': tokens_per_active_region.total(),
            'local_order_updates': local_order_counts.total(),
        },
        'definitions': {
            'spatial_region': f'Filled response positions separated by fewer than {mask_gap} MASKs share a region. '
                              f'At least {mask_gap} intervening MASKs separates regions.',
            'top1_activity': 'Stay/jump compares consecutive valid commitments; merges are separate.',
            'threshold_activity': 'Regions receiving >=1 valid commitment, evaluated AFTER the entire batch; '
                                  'new regions included and batch-created merges count as one. '
                                  'Old regions touched is also saved in each step record.',
            'local_order': 'Updates with one old predecessor only. Prefix = no left expansion and every '
                           'pre-step mask from the old left endpoint through the rightmost commitment filled. '
                           'Left expansion takes precedence over skipped-hole classification. '
                           'New regions and merges excluded; their counts reported separately.',
            'limits': 'A spatial order proxy, not causal evidence or proof of autoregressive modeling. '
                      'Threshold has no order within its batch. Valid commits are nonspecial and '
                      'before final stop. Structural regions include all filled positions before stop. '
                      'Percentages are pooled; skipped invalid-anchor steps are counted.',
        },
        'per_question': per_question,
    }

    # 8. Draw the figures, stating policy, spatial definition and population.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gap_label = f'Regions separate at >= {mask_gap} intervening MASKs'
    figures = []
    if policy == 'top1':
        keys = ['stay_same_region', 'jump_to_other_existing_region', 'open_new_region', 'merge_regions']
        figures.append(('3_1_top1_stay_or_jump',
                        ['Stay in same region', 'Jump to older region', 'Open new region', 'Merge regions'],
                        [top1_transitions[k] for k in keys], top1_transitions.total(),
                        '3.1 TOP1: stay in the same region or jump?',
                        'Consecutive valid commitments; merges counted separately',
                        'Percentage of qualifying transitions'))
    elif policy == 'threshold':
        figures.append(('3_1_threshold_regions_per_batch',
                        [str(k) for k in sorted(active_region_counts)],
                        [active_region_counts[k] for k in sorted(active_region_counts)], active_region_counts.total(),
                        '3.1 THRESHOLD: how many regions receive commitments per batch?',
                        'X-axis: active regions AFTER the batch; batches with valid commitments',
                        'Percentage of qualifying batches'))
        figures.append(('3_1_threshold_tokens_per_region',
                        [str(k) for k in sorted(tokens_per_active_region)],
                        [tokens_per_active_region[k] for k in sorted(tokens_per_active_region)], tokens_per_active_region.total(),
                        '3.1 THRESHOLD: tokens committed in each active region',
                        'X-axis: number of new valid tokens in one region in one batch',
                        'Percentage of active-region updates'))
    order_keys = ['left_to_right_prefix', 'skips_earlier_masks', 'left_expansion']
    figures.append(('3_2_local_left_to_right',
                    ['Left-to-right prefix', 'Skips earlier masks', 'Expands left'],
                    [local_order_counts[k] for k in order_keys], local_order_counts.total(),
                    f'3.2 {str(policy).upper()}: local left-to-right order inside existing regions',
                    'One pre-existing region per update; births/merges excluded; no within-batch ordering assumed',
                    'Percentage of single-predecessor region updates'))
    summary['plotted_rates'] = {}
    for filename, labels, values, denominator, title, population, ylabel in figures:
        percentages = [100*v/denominator if denominator else 0 for v in values]
        summary['plotted_rates'][filename] = [
            {'label': label, 'numerator': value, 'denominator': denominator,
             'percent': 100*value/denominator if denominator else None}
            for label, value in zip(labels, values)]
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        bars = ax.bar(range(len(labels)), percentages, color='#3478aa')
        ax.set_xticks(range(len(labels)), labels=labels)
        if len(labels) > 15:
            ax.tick_params(axis='x', labelsize=8, rotation=90)
        ax.set_title(f'{title}\n{gap_label}\n{population} | n={denominator:,}', fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(10, max(percentages, default=0)*1.18))
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        if denominator and len(labels) <= 15:
            ax.bar_label(bars, labels=[f'{p:.1f}%\n({n:,})' for p,n in zip(percentages,values)], padding=3)
        if not denominator:
            ax.text(.5,.5,'No qualifying observations',ha='center',transform=ax.transAxes)
        for extension in ('png', 'pdf'):
            fig.savefig(out/f'{filename}.{extension}', dpi=180)
        plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mask-gap', type=int, default=4)
    args = parser.parse_args()
    generate_region_activity_plots(args.run, args.out, args.mask_gap)


if __name__ == '__main__':
    main()
