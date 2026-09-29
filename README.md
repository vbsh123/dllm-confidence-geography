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

The pinned model is `GSAI-ML/LLaDA-8B-Instruct` at `08b83a6feb34df1a6011b80c3c00c7563e963b07`. Loading uses the checkpoint's custom code. Chat formatting comes from its tokenizer. Prompts request reasoning and a final `#### number`. No reference reasoning is shown to the model.

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

The final result includes full token IDs, stop position, answer and reference, strict/lenient numeric scoring, timings, GPU peak allocation, token dictionary, and the retrospective word map. Strict accuracy requires the requested `####` marker. Lenient scoring falls back to the last numeric string and is reported separately; neither evaluates individual reasoning steps.

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

## Validation status

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
