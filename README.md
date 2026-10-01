# Diffusion confidence geography

An independent GSM8K experiment with LLaDA-8B-Instruct. The question is: **when does a diffusion model choose a position that is not adjacent to its most recent fill, what kinds of tokens does it choose, and where does it go?** This repository contains instrumentation and an analysis plan; it does not incorporate findings from a literature review.

**Run model inference on Vast, not on the local development machine.** No model checkpoint is bundled. A smoke run on the actual GPU is required before treating the implementation as validated for LLaDA.

## The event we measure

Positions are zero-based response **token** indices. A choice is nonlocal when its distance from the most recent committed position is **greater than one**, to either the left or right. For a previous multi-token batch, distance means the minimum distance to any token in that batch. A batch has no invented internal temporal order. Initial choices have no previous anchor and are excluded from movement rates.

This differs from distance from the leftmost unresolved position, which is retained as a secondary feature. We distinguish:

- A distant choice, even if its confidence was already high.
- An actual increase in confidence at a still-masked distant position.
- A change of predicted token versus increased probability of the *same* token.
- Removal of the previous confidence leader versus overtaking a leader that remains masked.
- A jump with a local alternative versus a forced jump because no immediate masked neighbor remains.

Tokenization can split words. A second, retrospective view maps final decoded tokens to Unicode words and punctuation units; same-word pieces have distance zero. This is approximate, explicitly labeled, and not used by decoding. Unstable decoding prefixes make the word view unavailable rather than silently misaligning it. Raw IDs, pieces, and readable text are retained for better annotations later.

## Vast quick start

The environment script targets Linux, Python 3.10–3.12, and a single BF16-capable NVIDIA GPU with **48 GB or more VRAM**, plus at least 32 GB host RAM and 80 GB free disk for dependencies, weights, and traces. It installs PyTorch 2.6.0 with CUDA 12.4 wheels; the host driver must support that runtime. These are conservative provisioning targets, not measured minimum requirements. Actual time and peak GPU memory are logged.

Download or upload the source archive `dllm-confidence-geography-source.tar.gz` to your Vast instance. The source archive needs no GitHub credentials on the GPU machine. In the Vast terminal:

```bash
mkdir -p /workspace
cd /workspace
# Upload dllm-confidence-geography-source.tar.gz here first.
tar -xzf dllm-confidence-geography-source.tar.gz
cd dllm-confidence-geography
bash scripts/vast.sh smoke
```

`smoke` uses two questions, 64 response tokens, and both decoding policies. It checks setup and trace generation; its accuracy is not an experiment result. Inspect the terminal and generated plots before the pilot.

For the paired pilot of 100 questions and 256 response tokens per policy:

```bash
RUN_NAME=gsm8k_pilot bash scripts/vast.sh pilot
```

To keep it running after disconnecting:

```bash
nohup env RUN_NAME=gsm8k_pilot bash scripts/vast.sh pilot > vast-launch.log 2>&1 &
tail -f vast-launch.log
```

Do not launch the foreground and background versions simultaneously. Use the same `RUN_NAME` and identical settings to resume after interruption. Completed samples are skipped; the interrupted sample restarts. Changing arguments or Python source for an existing policy output directory is rejected. Do not change source/environment in the middle of a run; use a fresh name.

Optional settings:

```bash
# Longer response window if the pilot shows frequent length-limit hits:
RUN_NAME=gsm8k_len512 LENGTH=512 SAMPLES=100 bash scripts/vast.sh pilot

# Optional left-to-right fill-order control, using the same diffusion model:
RUN_NAME=gsm8k_with_control RUN_LTR=1 bash scripts/vast.sh pilot

# A separate block-constrained condition; full sequence is the default.
RUN_NAME=gsm8k_block64 BLOCK_LENGTH=64 bash scripts/vast.sh pilot

# Larger experiment, only after inspecting pilot traces:
RUN_NAME=gsm8k_full bash scripts/vast.sh full
```

`full` defaults to 500 questions. Other knobs: `SEED` (1729), `OFFSET` (0, after seeded shuffle), `COMMIT_THRESHOLD` (0.9), `MAX_PLOTS` (8), and `SKIP_INSTALL=1` after a successful environment setup. Keep `SAMPLES`, `LENGTH`, `BLOCK_LENGTH`, seed and offset identical when comparing policies. Sampling is without replacement from GSM8K test. The second policy uses the dataset commit resolved by the first. Increasing samples with a new run name includes earlier questions; use `OFFSET` for a disjoint additional shard.

All checks in `vast.sh`, including simulated CPU tests, run **on Vast**. A failed check stops the script. The script does not rent a machine or start a paid instance.

## Decoding conditions

The default experiment runs two conditions on the same shuffled questions:

1. `top1`: one highest-confidence masked token per forward pass. This gives the clearest interpretation of “the latest fill.”
2. `threshold`: simultaneously fill all eligible positions at or above 0.9; if none qualify, fill the most confident one. This probes disconnected confident regions.

This is an explicit uncached threshold baseline, **not a reproduction of Fast-dLLM's accelerated implementation**. Both conditions recompute the full sequence each step, with zero sampling temperature, no classifier-free guidance, and no EOS suppression. Full-sequence decoding is the default, so blocks cannot dictate the movement pattern. Optional `left_to_right` fixes fill order while leaving the model bidirectional; it is not an autoregressive-model baseline. `scheduled` supports a fixed total number of steps through the Python CLI.

The pinned model is `GSAI-ML/LLaDA-8B-Instruct` at `08b83a6feb34df1a6011b80c3c00c7563e963b07`. Loading uses the checkpoint's custom code. Chat formatting comes from its tokenizer. The user message is **exactly the dataset question**, with no added instruction to reason, solve, or use an answer marker. No reference reasoning is shown to the model. Manifests and results label this as `question_only_chat_v1`; trace headers also save the literal `user_message`, rendered prompt, and prompt token IDs. The original pilot used an added reasoning/`####` instruction and is a different prompt condition; retain it separately.

The mask symbol cannot be committed. Its original probability and the unfiltered argmax are logged. Candidate probabilities remain those of the original full-vocabulary softmax: excluding MASK from candidate ranking does **not** renormalize them. Confidence and exact entropy are computed in FP32 over vocabulary chunks to limit memory.

All response positions are filled, even if EOS/EOT appears early in decoding. Otherwise an early end token could censor the trajectories we want to study. For answer scoring, the final sequence is truncated at its first EOS/EOT. Analyses distinguish observed end tokens during decoding from the eventual final stopping position. `hit_length_limit` flags a completed fixed-size window without a stop token; inspect this before drawing conclusions from 256-token runs.

## What gets saved

```
runs/RUN_NAME/
  driver.log, packages.txt, nvidia-smi.txt
  top1/                         # also threshold/, optionally left_to_right/
    manifest.json               # config, source hash, code commit if available, packages, GPU, dataset revision
    samples.jsonl               # exact shuffled sample identities/questions/references
    summary.json
    samples/SAMPLE_ID/
      trace.jsonl.gz            # header, every pre-commit step, final result
      result.json
    analysis/
      overview.png
      sample_SAMPLE_ID.png
      events.csv, steps.csv, thresholds.csv, problems.csv
      summary.json, examples.json
```

Each step records **all still-masked positions**, including future blocks if block decoding is enabled:

- Top five non-mask candidate IDs/probabilities, readable top token, tokenizer piece dictionary, raw argmax, mask probability, top-two margin, exact distribution entropy.
- Top-candidate confidence change, prediction changes, and the current probability of the *previous* candidate (`same_token_delta` separates these).
- Previous fill batch, signed nearest distance, nonlocal flag, nearest-any-filled distance, left frontier, block eligibility, and local alternative availability.
- The full pre-step response state, committed IDs/positions, progress, model-forward time, block and stop-token context.
- Above-threshold counts, fractions, contiguous islands and newly crossing counts at **0.5, 0.7, 0.8, 0.9, 0.95 and 0.99**. These are measurements; only `COMMIT_THRESHOLD` controls threshold decoding.

The final result includes full token IDs, stop position, answer and reference, numeric scoring, timings, GPU peak allocation, token dictionary, and the retrospective word map. **`numeric_accuracy` is the primary answer score**: use the last `####` number if the model emits that marker naturally, otherwise the last numeric string. No marker is required. This is a heuristic that can misidentify answers when a response ends with another number; extraction methods are recorded for auditing. `correct_numeric` labels individual results. Legacy `strict_accuracy`/`correct_strict` fields still require a marker and remain only for compatibility/diagnostics, not as the primary score. `lenient_accuracy`/`correct_lenient` use the same numeric extraction as the primary score. None of these evaluates individual reasoning steps.

The trace plus model revision and prompt reconstructs each observed state for future interventions. We do not save every vocabulary logit, hidden state, or attention matrix. Those would greatly increase volume and are not needed for this first question. JSONL is compressed incrementally; unfinished samples carry `.partial` and are excluded from exports. Traces may still be large; size the pilot before expanding.

## Plotting and interpretation

The overview answers **when**, **what token category**, and **where**. Sample plots show confidence and confidence-change heatmaps over position × step, plus the commit trajectory. CSVs preserve individual events for richer annotation later.

Primary movement rates exclude special tokens, final post-stop positions, and events whose previous batch contains a special/post-stop anchor. Raw data remain available. Whitespace, punctuation, numbers, alphabetic fragments and mixed text are separate categories; these are surface categories, not claims about semantic reasoning or grammatical parts of speech. Read examples in context before assigning labels such as “unit” or “intermediate calculation.”

`events.csv` contains commits, leaders, and remote confidence increases of at least 0.15 (configurable with `--rise`). A remote increase is defined over consecutive pre-commit model evaluations; it is not inferred merely from movement of the leader. `matches_final_token` measures whether a prediction agrees with the eventual token at that position. This is trajectory stability, **not token correctness**. The trace can also show temporary reversals that return to the same final token.

Aggregate movement fractions average within each problem first and bootstrap **problems**, avoiding treating dependent token decisions as independent samples. `nonlocal_with_local_alternative_problem_bootstrap` restricts to choices where a local in-answer option exists. The unfiltered uniform-choice nonlocal rate is a geometric opportunity baseline, not a counterfactual model run. Bootstrap intervals with very few questions should not be overinterpreted.

`thresholds.csv` includes `all_masked`, `eligible`, `eligible_nonspecial`, and retrospective `answer_eligible_nonspecial` scopes. The latter recomputes groups after final stop filtering; raw island counts can otherwise be dominated by padding/end behavior. Islands are contiguous token positions, not semantic regions. `examples.json` selects the largest qualifying rises by a fixed ranking, without cherry-picking successes. It is illustrative, not a representative sample of all events.

You can regenerate analyses on Vast or later on a CPU machine without downloading the model:

```bash
python -m confidence_geography.analyze \
  --run runs/gsm8k_pilot/top1 \
  --out runs/gsm8k_pilot/top1/analysis --max-plots 20 --rise 0.15
```

For multiple runs, pass multiple paths after `--run`; analyses go in separate subdirectories and `comparison.json` collects summaries. A policy comparison measures different trajectories, not just different observations of the same trajectory.

## Optional later replay on Vast

Choose a `remote_rise` in `examples.json`, then replay its step and target position:

```bash
python -m confidence_geography.replay \
  --trace runs/gsm8k_pilot/top1/samples/SAMPLE_ID/trace.jsonl.gz \
  --step 12 --position 87 --max-individual 8 \
  --out runs/gsm8k_pilot/replay/sample_step12_pos87.json
```

Replace the IDs and numbers with an actual event. The replay evaluates the observed state, withholds all previous-step fills (recovering the prior state), and withholds up to eight individual fills. It measures the probability of the same observed target token in every condition and checks agreement with the recorded forward passes. This establishes a conditional effect on the model prediction, including mask-count effects; it does not by itself identify semantic reasoning or a general causal mechanism. Replay is optional and is not launched by the default script.

## Export results

The launch script automatically creates an archive and SHA-256 checksum in `exports/`. To export a stopped run manually, after stopping its writer:

```bash
bash scripts/export.sh runs/gsm8k_pilot
```

Download the generated `.tar.gz` and `.sha256` from Vast before removing the instance. The archive contains the requested run directory, including traces, question text, answers, plots and metadata; model weights and the Hugging Face cache are outside it. Do not point `export.sh` at a cache or home directory. On receipt, verify `sha256sum -c ARCHIVE.tar.gz.sha256` from the repository root (the checksum includes the `exports/` prefix).

## Region creation, continuation, and confidence statistics (offline)

`region_stats.py` reproduces the region analyses from saved raw traces. It uses
**only Python's standard library**: no package installation, model downloads,
PyTorch, GPU, or inference. Run once per policy, using either a policy directory
or its ZIP archive. The input must contain completed `trace.jsonl.gz` files with
their sibling `result.json` files. `cases.json` alone is not sufficient.

```bash
# From the repository root, on Vast or your own computer:
python -m confidence_geography.region_stats \
  --run runs/gsm8k_question_only_v1/top1 \
  --out runs/gsm8k_question_only_v1/top1/region_stats

python -m confidence_geography.region_stats \
  --run runs/gsm8k_question_only_v1/threshold \
  --out runs/gsm8k_question_only_v1/threshold/region_stats

# ZIPs can be read directly, without extraction:
python confidence_geography/region_stats.py \
  --run /path/to/threshold.zip --out analysis/threshold_regions
```

Choose a new/empty output directory for each analysis. Defaults are
`--min-distance 4 --cutoffs 0.85 0.9 --rise 0.15`. Add `--no-events` to omit the
detailed event export. This command does not alter input traces.

Outputs:

- `report.md`: readable counts, definitions, percentages, and denominators.
- `summary.json`: all metrics plus collector configuration and input provenance.
- `rates.csv`: numerator, denominator, pooled percentage, and equal-question mean
  for each metric. Undefined rates are null, never silently zero.
- `signed_distances.csv`: exact integer-distance counts and percentages, for all
  primary commitments and the subset with an adjacent alternative.
- `events.jsonl.gz`: seed choices, before/after neighbor predictions, and actual
  next-step region commitments, for inspecting examples.

Key report metrics:

| Question | Metric |
|---|---|
| What fraction of created regions continued next step? | `created_regions/continued_next` |
| Out of region-creating batches, how often did the next batch continue a new region without extending older regions? | `creation_batches_next/new_only` |
| Continue both new and older regions, older only, or neither? | Other `creation_batches_next/` categories |
| What fraction of all token commitments created regions that continued? | `valid_commits/creation_tokens_in_continued_regions` |
| Where did the next top1 choice go? | `top1_seed_next_valid/` categories |
| Did top1 leave a >=90% neighbor behind? | `top1_neighbors_when_went_elsewhere/cutoff_0.9/events_any_ready` |
| How many below-85% neighbors crossed 85%? | `top1_neighbors_valid_next/cutoff_0.85/below_before_crossing` |
| How many stayed below 85%? | Same prefix, `below_before_still_below` |
| Did the same predicted token cross, rather than a replacement prediction? | Same prefix, `same_token_crossing` (denominator: all paired neighbors with recorded same-token probability) |
| What happened to threshold neighbors that remained masked? | `batch_remaining_seed_neighbors/` metrics |
| Did a new-region fallback precede a multi-token batch? | `seed_fallback_batches/next_multiple_commits` |

**Definitions matter.** Index distance 4 means three intervening masks. A seed
is far from *all* pre-existing filled positions, including the prompt boundary.
A strict region groups adjacent simultaneous seeds, but excludes any group
connected by simultaneous commitments to positions closer to older text. Thus
seed counts and strict-region counts are intentionally different. "Current"
regions contain previous-batch commitments; "older" regions for continuation
are everything filled after creation except the newly created strict regions.

"Continued" always means directly adjacent on the **very next actual step**.
`new_only` means no older region was extended; it does not forbid opening an
additional isolated region in that same next batch. Batch percentages count each
creation step once, even if it creates several regions. Region percentages count
those regions individually. Final steps without a next pass are reported
separately. Special/post-final-stop tokens are excluded from valid commitments;
top1's primary next-destination/confidence population additionally excludes
special/post-stop next choices. Neighbor confidence observations exclude
co-committed neighbors because they are no longer masked on the next pass.

These measurements are descriptive, not evidence that parallel commitments are
independent or correct. Threshold and top1 follow different trajectories.

Offline checks for this module, without any model:

```bash
python -m unittest discover -s tests -p test_region_stats.py -v
```

Validation: 12 offline tests passed. Reprocessing the downloaded 100-question
top1 and threshold archives reproduced the discussed counts, including
`1227/1977` below-85% neighbors crossing 85%, `747/974` top1 departures leaving a
>=90% neighbor, `1685/1970` threshold regions continued, and `265/1488` threshold
creation batches followed by continuation of new regions without extending old
ones. No model inference was used for these checks. Note that top1's
`created_regions/continued_next` uses all 1,766 creations (781 continued), while
`top1_seed_next_valid/continue_new_region` excludes 11 special/post-stop next
choices (781/1,755); this explains the 44.2% versus 44.5% denominators.

## Distance/confidence histograms and active-region dynamics

This additional **offline** analysis reads the same raw traces and creates PNG
and PDF plots. It uses NumPy/Matplotlib (already project dependencies), never a
model. It can also collect data with standard-library Python using `--no-plots`,
then plot later in an environment with these two packages installed.

```bash
python -m confidence_geography.region_dynamics \
  --run /path/to/top1.zip --out analysis/top1_dynamics \
  --gaps 1 2 3 4 8 16 --focus-gap 4

python -m confidence_geography.region_dynamics \
  --run /path/to/threshold.zip --out analysis/threshold_dynamics \
  --gaps 1 2 3 4 8 16 --focus-gap 4

# To render again without re-reading the original traces:
python -m confidence_geography.region_dynamics \
  --out analysis/top1_dynamics --plots-only
```

Open `report.md`, or the PNG/PDF figures directly:

- `distances`: signed distance from the previous batch, full/zoomed views, and
  number of masks to the nearest filled token or prompt boundary.
- `neighbor_confidence`: distributions of top1 probability after reveal and
  top1-probability delta; overlays same-token probability changes and plots
  before versus after. Rows compare all reveals with isolated seed reveals.
- `region_activity`: region counts over decoding progress, sensitivity to X,
  and maximum candidate confidence for recent/older/unassigned regions.
- `region_behavior`: direction of commitments and confidence history at fixed
  would-be seed positions before the new-region commitment.
- `example_timeline`: a saved question's region spans and confidence maxima
  over actual decoding steps.

Raw outputs are `summary.json`, `frames.jsonl.gz`, `neighbors.csv.gz`, and
`birth_history.csv.gz`. Frames store each region's bounds and maximum candidate
confidence, so individual examples can be inspected beyond aggregate plots.

**This X counts actual intervening masks**, unlike the earlier index-distance
cutoff. With X=4, positions 0 and 5 are separate; positions 0 and 4 belong to the
same region. Such a region can contain internal masked holes. An occupied region
contains filled response text before the final stop. A recently touched region
contains a token committed in the previous 5 (or 10) actual steps. Regions can
merge. The prompt is not counted as an occupied response region, but it and
full-window filled stop tokens remain anchors when identifying isolated births.

A masked candidate is assigned to every occupied region it would join if
revealed under the X-gap rule. Some candidates bridge two regions. Maxima use
the previous-5-step activity definition; unassigned positions join no region.
Candidate counts differ, so these maxima are descriptive, not a controlled
comparison. Tracking future seed positions backward is also conditioned on their
eventual selection.

For AR-like behavior, the script separately measures immediate right expansion,
left expansion, internal-hole filling, skips, and bridges. The
`gap_X/recent5_single_region_commits/leftmost_unresolved` rate checks the first
eligible mask after a region's left endpoint, including immediate right
extension, among commitments assigned to a single recently touched region with
such a candidate. This is a local spatial proxy; threshold tokens in a batch do
not have a sequential order.

Neighbor observations retain the actual next step even when it commits only
special/post-stop tokens, and omit neighbors committed simultaneously. Therefore
the threshold neighbor distribution is a selected population of still-masked
neighbors and should not be directly interpreted as an improvement over top1.
Figures state weighting; region counts summarize pre-stop response states while
the answer still has masks. Mean probability and mean change are not substitutes
for the full distributions.

Offline tests: `python -m unittest discover -s tests -p 'test_region*.py' -v`.

## Initial experiment validation status

An initial 11-test CPU suite passed using a tiny, explicitly simulated model, including trace reconstruction, actual jump distances, EOS handling, policies, probability accounting and plot generation. Subsequent local-alternative instrumentation has an additional test queued for Vast. No real LLaDA inference or GSM8K experiment was run locally. GPU/checkpoint compatibility, actual throughput, memory usage and empirical findings remain to be established by the Vast smoke and pilot runs.

## Zoomed signed-distance plots from existing CSVs

No model inference or trace reprocessing is needed:

```bash
python -m confidence_geography.distances \
  --events runs/gsm8k_pilot/threshold/analysis/events.csv \
  --out runs/gsm8k_pilot/threshold/analysis --zoom 20
```

Open `distances_zoom.png` for the standalone ±20 zoom, with every integer labeled and immediate neighbors in orange. `distances.png` retains the full-range comparison. Both show out-of-window percentages. The two rows show all primary fills and only those with an adjacent option available. Percentages average within problems, matching the summary JSON; the zoom does not renormalize to the visible subset. Exact values and pooled counts are in `distances.json`. Replace `threshold` with `top1` for the single-fill condition. This addition has not been executed locally; run it on the exported data or Vast. Updating source is fine for analysis of completed runs; do not update source mid-collection if you intend to resume that run.

## Read actual distant token choices

Start with the single-token policy to make the previous fill unambiguous:

```bash
python -m confidence_geography.inspect_jumps \
  --run runs/gsm8k_pilot/top1 --min-distance 10 --max-distance 20 \
  --require-local --out runs/gsm8k_pilot/top1/analysis/jumps_10_20
```

Open the generated `index.html` in a browser. It shows the question, exact masked state, chosen token, previous fills, adjacent predictions and confidence changes. Final answers are hidden behind an expandable section to distinguish retrospective context from what the model actually saw. Download `cases.json` to share the selected cases. The report uses saved `analysis/events.csv` and traces; it does not load a model or run inference. Distances are tokens, not words. Cases rank by largest absolute distance, capped at three per problem and one per step; they are deliberately selected examples, not a prevalence estimate. Omit `--max-distance` for the largest jumps, use `--sort rise` to prioritize confidence increases, or replace `top1` with `threshold` to inspect batch choices. `--require-local` requires an adjacent eligible nonspecial in-answer alternative. The adjacent-alternative table also displays eligible special/post-stop candidates, explicitly marking special tokens. This report addition is awaiting execution on the saved Vast results; no local experiments were run.

### Large confidence increases, rather than just large spatial moves

```bash
python -m confidence_geography.inspect_jumps \
  --run runs/gsm8k_pilot/top1 --event-kind remote_rise \
  --min-distance 10 --min-rise 0.20 --min-confidence 0.90 \
  --same-token-only --sort rise \
  --out runs/gsm8k_pilot/top1/analysis/confidence_rises
```

This selects the same predicted token rising by at least 20 percentage points to at least 90% probability, at least ten token positions from the previous fill. It includes predictions not immediately committed and labels that distinction explicitly. Use `--event-kind commit` to restrict to actual fills. The report displays the tokens revealed in the previous step. Neither mode establishes causality; previous fills are temporal predecessors. Remote-rise candidates come from the existing analysis CSV, which by default includes only increases of at least 0.15. To study smaller uncommitted increases, first regenerate analysis with a smaller `--rise`; lowering the report filter alone cannot recover excluded rows. These report filters have not been executed locally.

## Rerun after removing the added prompt instruction

Run these commands on Vast, after the original pilot is finished. Keep the original results. If this is a new instance, clone the repository, restore the old `runs/gsm8k_pilot/` directory, and set up the environment with `bash scripts/vast.sh smoke` before reading its manifest.

```bash
cd /workspace/dllm-confidence-geography
git pull --ff-only
source .venv/bin/activate

# Use the original dataset commit, so the default seed/offset select the same questions.
export DATASET_REVISION="$(python -c 'import json; print(json.load(open("runs/gsm8k_pilot/top1/manifest.json"))["dataset_info"]["revision"])')"

RUN_NAME=gsm8k_question_only_smoke SKIP_INSTALL=1 bash scripts/vast.sh smoke

# After the smoke succeeds, run the complete paired pilot in a new directory.
nohup env RUN_NAME=gsm8k_question_only_v1 SKIP_INSTALL=1 \
  SAMPLES=100 LENGTH=256 BLOCK_LENGTH=0 SEED=1729 OFFSET=0 COMMIT_THRESHOLD=0.9 \
  bash scripts/vast.sh pilot > /workspace/gsm8k_question_only_v1.log 2>&1 &
tail -f /workspace/gsm8k_question_only_v1.log
```

These values match the original default pilot. If the original manifest has different settings, match them explicitly instead. `DATASET_REVISION` is inherited by the background process. `SKIP_INSTALL=1` assumes the existing environment was installed successfully; omit it on a fresh setup. **The smoke and pilot commands run LLaDA on Vast**, unlike the CSV/trace inspection commands. They run the updated simulated tests before loading the actual checkpoint. The question-only prompt/scoring changes have not been executed locally.

The rerun changes the user-message content and makes numeric extraction (without a required marker) the primary evaluation. It preserves the decoder: BF16 model, FP32 probabilities, no random token sampling, no guidance, no cache, full response window, and MASK excluded from commitment without probability renormalization. EOS/EOT does not halt filling: the entire fixed window is filled and only final scoring/primary analysis are truncated at the first stop. Surface token categories and subword distances are approximate descriptions, not semantic labels. These choices should remain explicit when interpreting the results. Accuracy uses a heuristic and is not a matched official benchmark evaluation.
