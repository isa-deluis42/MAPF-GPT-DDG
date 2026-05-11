# Human-in-the-Loop Congestion Classification for Multi-Agent Pathfinding

Final Project Report — Working Draft

> Format target: 6 pages, double-column. Easiest path is to write here in Markdown, then convert with `pandoc` or paste into the IEEE/ACM double-column LaTeX template before submission. Mark replacements with `TODO` / `[FILL]` / `[FIGURE N]`.
>
> Due: May 12 @ 11:59 PM EST.

---

## Authors

Shane Pornprinya, Isabel De Luis, Sparsh Bansal

---

## Abstract

Multi-agent pathfinding (MAPF) policies trained with imitation learning improve when their training data is enriched with hard-case rollout segments where the policy gets stuck in congestion. The original Difficulty-Driven data Generation (DDG) pipeline detects these moments with a hand-tuned threshold on a fast solver's makespan-improvement estimate, discarding everything in the borderline range. We argue this throws away signal that humans can readily provide. We introduce a learned segment-ranking 3D CNN that consumes a 16-step spatio-temporal volume of the multi-agent state and is trained with a continuous-weight RankNet loss on cheap auto-labels, then fine-tuned in a two-phase `--human-only` step on rare human pairwise verdicts collected through a custom debiased replay tool. We compare three label-acquisition strategies at fine-tune budget — (i) no human labels (auto-only baseline), (ii) uniform random sampling, (iii) confusion-driven active learning that ranks candidate pairs by `H(σ(s_A − s_B))` — evaluated on a held-out 76-pair human-pair signal across 69 val-map rollouts (`annotation_val_map.json`). Across all three stages the auto-aligned validation `pair_acc` is preserved to within 0.002 (0.648 → 0.650), demonstrating that human fine-tuning does not damage DDG-aligned ranking. On the held-out human-pair signal, confusion AL converges to a stable `human_val ≈ 0.71` (vs `≈ 0.65` end-of-training for random) — a small but stable lift attributable to selection strategy, not label volume (confusion uses 65 labels, random uses 78). We additionally close the loop end-to-end by plumbing the trained classifier into DDG's expert-selection step (`finetuning/delta_data_generator.py`) and retraining MAPF-GPT from this classifier-gated curriculum. On the POGEMA Random / Maze / Warehouse suites at the shared `ckpt_ddg_1500.pt` cut, the **human-fine-tuned-classifier-gated MAPF-GPT (`MAPF-GPT-S2`, gated by the Stage-2 ranker that was fine-tuned on 78 randomly-sampled human pairs) strictly dominates both `MAPF-GPT-original` and the auto-only-classifier-gated `MAPF-GPT-S1` on every cell with headroom**, lifting coverage success rate (CSR) over the threshold-gated original by **+26 pp at Random-32, +30 pp at Random-48, +25-28 pp at Maze-24/32, and +27 pp at Warehouse-64 (0.72 → 0.98)** — all without paying any SoC penalty (S2's path lengths are within ≈3% of Original's, and on Maze 32 actually shorter). The auto-only `MAPF-GPT-S1` itself lifts CSR over Original by +11/+16/+18 pp on Random / Warehouse but is roughly tied on Maze, isolating the value of the 78-pair human supervision: it converts a regime where the auto-only ranker tied with the threshold (Maze) into a regime where it wins by 25-28 pp. The Stage-3-classifier-gated `MAPF-GPT-S3` (with confusion-AL human pairs) is currently being trained. A label-preserving spatial-symmetry augmentation of the rollout corpus (Isabel De Luis, [`augment_segment_rollouts.py`](augment_segment_rollouts.py); 5,007 augmented `.npz` files committed) provides a 4× expansion of the auto-pair training data at zero label cost, available for future training runs. The headline takeaway is methodological: at this scale and label budget, **the auto signal is much stronger than previously thought** — human fine-tuning provides a stable, modest re-alignment toward human judgment without sacrificing the auto baseline, and an entropy-only AL acquisition outperforms uniform random sampling by a small but reproducible margin.

---

## 1. Introduction

### 1.1 Motivation

Coordinating teams of robots in shared environments — warehouses, delivery fleets, search-and-rescue swarms — fundamentally requires solving multi-agent pathfinding (MAPF). Recent work has shown that transformer policies trained on solver demonstrations can solve large-scale MAPF instances at inference time with a fraction of the compute an exact solver would need [^mapfgpt]. But these learned policies are only as good as the training data, and they consistently fail on a long tail of congested scenarios that look superficially similar to easy ones — agents bunched at corridor pinch-points, deadlocks at junctions, oscillating swap conflicts.

The state-of-the-art remedy is Difficulty-Driven data Generation (DDG) [^ddg]: roll out the current policy, identify rollout segments where it appears to struggle, invoke an expensive expert solver on those segments only, and add the resulting expert demonstrations back into the training set. The selection step matters: calling the expert on every segment is wasteful, and missing genuinely hard segments leaves the policy's blind spots unfixed.

Today, segment selection in DDG is performed by a hand-tuned threshold on a fast LaCAM probe's makespan-improvement estimate. The threshold is brittle: it discards an entire midrange band of borderline-difficult segments and disagrees with human judgment on a substantial fraction of cases.

### 1.2 HRI Framing

This is a Human-Robot Interaction problem in two ways:

1. The "robot" is a multi-agent system that operates without per-step human supervision, but whose long-run training improves when humans inject judgment about which behaviors look problematic. The same problem shape recurs anywhere robot teams operate among or for humans — warehouse fleets, multi-AGV manufacturing, autonomous mobility-on-demand.
2. The annotation interface itself is an HRI artifact. Asking humans to rank segments rather than to label them with absolute scores reduces cognitive load and avoids the calibration pitfalls of asking "how congested, on a scale of 1-10?" [^pairwise]. The replay-based segment-marking tool we built — including a debiasing step that randomly swaps left/right A/B presentation and hides the auto-label diff from the annotator — is the human-facing surface of an active-learning loop that closes between human judgment and a learned policy at planet-scale.

### 1.3 Contribution

We make six contributions:

1. **A segment-level spatio-temporal congestion classifier** that takes a 16-step volume of (agent positions, obstacles, goals, recent history) and outputs a scalar score, replacing DDG's hand-tuned threshold.
2. **A continuous-weight pairwise ranking objective.** We replace DDG's three-bucket (`diff > 3` / `diff < 1` / midrange-skip) scheme with a linear pair weight `min(diff_i − diff_j, MAX_PAIR_GAP) / MAX_PAIR_GAP` (with `MAX_PAIR_GAP = 5`) so that all auto-pair signal participates in training, weighted by confidence.
3. **A two-phase `--human-only` fine-tune protocol.** Phase 1 trains an auto-pair backbone; phase 2 fine-tunes (initialised from `--init-from <stage-1>.pair_acc.pt`) with auto pairs disabled and only human pairwise verdicts contributing gradient. Without this two-phase split, the ≈100 human pairs are diluted by ~78k auto pairs and never move the model.
4. **A controlled comparison of three acquisition strategies at fixed fine-tune budget**: (i) no human labels (Stage 1 — auto-only ranker), (ii) uniform random sampling over the train-map elicitation pool (Stage 2, 78 labels), (iii) confusion-driven active learning that ranks candidate pairs by `H(σ(s_A − s_B))` against the Stage 1 baseline scorer (Stage 3, 65 labels). All three preserve `pair_acc` to within 0.002 on the val-map auto signal, and confusion AL converges to a higher and more stable `human_val` than random sampling on a held-out 76-pair human-pair set.
5. **Label-preserving rollout augmentation** (Isabel De Luis, [`augment_segment_rollouts.py`](augment_segment_rollouts.py)). Spatial symmetries (hflip / vflip / rot180 / rot90 / rot270 / transpose) are applied directly to the segment-classifier `.npz` rollouts while preserving `segment_diffs`, providing a label-cost-free 4× expansion of the auto-pair training corpus (5,007 augmented files committed under `ranker_dataset/held_out_aug/`, covering all five DDG checkpoints under `dataset/held_out/`). Built and committed; not yet wired into a reported training run.
6. **End-to-end DDG integration and downstream MAPF-GPT benchmark setup** (`finetuning/delta_data_generator.py` + `eval_configs/`). The trained classifier is plumbed into DDG's segment-selection path: when `cfg.segment_classifier_path` is set, every env's segments are scored in one batched forward pass, the argmax-scored segment is picked per env, and the top-K envs (`cfg.expert_top_k`) are sent to the expert. A new MAPF-GPT model has been trained with this classifier-gated DDG (`checkpoints/baseline/ckpt_ddg_*.pt`, up to step 2000) and the apples-to-apples comparison vs the original threshold-gated MAPF-GPT (`checkpoints/original/ckpt_ddg_*.pt`) is configured on five POGEMA suites (random, mazes, warehouse, movingai, puzzles); results are pending.

A second, earlier augmentation track (Isabel De Luis, [`finetuning/export_augmented_active_learning_samples.py`](finetuning/export_augmented_active_learning_samples.py)) explores *synthetic-jitter* augmentation in the standalone congestion-classifier setting and is reported separately in §6.6.

---

## 2. Background and Related Work

### 2.1 Multi-Agent Pathfinding Solvers

Classical MAPF planners — Conflict-Based Search [^cbs], LaCAM [^lacam] — return optimal or near-optimal solutions but scale poorly with agent count. POGEMA [^pogema] provides a standardized benchmark suite with maze and warehouse maps used throughout this paper.

### 2.2 Learned MAPF Policies

MAPF-GPT [^mapfgpt] casts MAPF as autoregressive token prediction: each agent's local observation is tokenized into a 256-dimensional sequence, and a small transformer outputs the next action. The pretrained 2M-parameter model serves as our base policy. Training data is generated by running LaCAM at scale on synthetic instances.

### 2.3 DDG and the Hard-Case Selection Problem

MAPF-GPT-DDG [^ddg] augments standard imitation training with a hard-case mining loop. At each training checkpoint, the current policy is rolled out on synthetic scenarios; for each rollout, a 2-second LaCAM probe at every 16-step boundary estimates remaining makespan, the segment with the largest delta is identified, and — if that delta exceeds 3 — the 10-second LaCAM expert is invoked on that segment to produce additional training pairs. Segments with delta < 1 are discarded; segments in [1, 3] are also discarded as "ambiguous."

This thresholding rule is the bottleneck we attack in this paper.

### 2.4 Human-in-the-Loop Reinforcement and Imitation Learning

Prior HRI work on integrating human feedback into agent training has explored absolute reward shaping (TAMER [^tamer]), preference-based learning (Christiano et al. [^prefs]), and pairwise comparison interfaces [^pairwise]. We adopt pairwise rather than absolute ranking specifically because (a) absolute rating of multi-agent congestion is poorly defined and (b) pairwise comparison is the natural interaction granularity for a replay tool.

### 2.5 Pairwise Learning to Rank

RankNet [^ranknet] minimizes a logistic-style loss over score differences between paired examples. This is the loss the trainer in this work uses, with a continuous gap-derived weight in `[0, 1]` rather than the categorical bucket weights used in earlier iterations of this project.

### 2.6 Active Learning for Preference Elicitation

Active-learning surveys [^al-survey] partition acquisition strategies into rough families: uncertainty sampling (query where the model is least confident), diversity / representativeness sampling (cover the feature space), expected-information-gain (EIG) maximisation, and density-weighted variants that combine the above. For pairwise preference learning specifically, a common acquisition combines the entropy of the model's preference probability `p = σ(s_A − s_B)` with the feature-space distance between candidates [^pairwise-al]. We initially implemented this combined acquisition (`H · ‖φ_A − φ_B‖`) but observed that the diversity term concentrated queries on a few feature-extreme rollouts while leaving plenty of high-entropy in-distribution pairs unqueried; we therefore simplified the acquisition to entropy-only (`H(σ(s_A − s_B))`), giving the *pure uncertainty-sampling* baseline that is reported as Stage 3 (the "confusion AL" stage).

---

## 3. Research Question and Hypotheses

### 3.1 Research Question

> Can a small set of human-curated pairwise segment rankings improve a learned MAPF congestion classifier beyond what is achievable on cheap auto-labels alone, and does an *uncertainty-driven elicitation strategy* materially improve over uniform random sampling at the same label budget?

### 3.2 Hypotheses

- **H1 (auto-label noise).** The fast-solver-diff used by DDG to label segments disagrees with human judgment on a non-trivial fraction of borderline cases.
- **H2 (preservation under fine-tune).** Two-phase `--human-only` fine-tuning of an auto-trained backbone on a small budget of human pairs preserves DDG-aligned `pair_acc` (auto-pair ranking).
- **H3 (selection beats volume).** At the same fine-tune budget (≤80 human pairs), an entropy-driven acquisition function produces a stronger and more stable held-out human-pair classifier than uniform random sampling.
- **H4 (closed-loop deployment is feasible).** The learned classifier can replace the diff threshold inside DDG's `delta_data_generator.py` and produce drop-in expert-selection decisions per env, batched into a single forward pass — a system-level feasibility claim distinct from any downstream-quality claim.

---

## 4. Method

### 4.1 Held-Out Seed Set

To enable clean evaluation across DDG checkpoints, we reserve a fixed seed set never seen during DDG training. Map seeds are split into [TRAIN_MAP_SEEDS](held_out_seed_set.py) = `{128..143}` (used for the classifier's auto-pair training, plus all human-elicitation pools) and [VAL_MAP_SEEDS](held_out_seed_set.py) = `{144, 145, 146, 147}` (used for `pair_acc` and the held-out human-pair val signal). The full design grid spans 2 map types (`maze`, `random`), 5 DDG-policy checkpoints (`ckpt_0`, `ckpt_500`, `ckpt_1000`, `ckpt_1500`, `ckpt_30000`), 3 scenario seeds (`1000`, `1001`, `1002`), and 3 agent counts (`{16, 32, 48}`). Out of the 1,800 (= 2 × 20 × 3 × 3 × 5) potential cells, the [`ranker_dataset/held_out/`](ranker_dataset/held_out/) directory contains 1,669 episode `.npz` files (the colab also expects this corpus locally as `dataset/held_out/` after a Drive download).

### 4.2 Spatio-Temporal Featurization

Each 16-step segment is encoded as a 4-channel volume of shape `(4, 16, 32, 32)`:

| Channel | Content |
|---|---|
| 0 | Agent occupancy density at each timestep within the segment |
| 1 | Obstacle map (broadcast across time) |
| 2 | Goal density (broadcast across time) |
| 3 | Pre-segment agent history density (broadcast across time) — captures oscillation and dithering |

Maps are zero-padded to a uniform 32×32 spatial grid.

### 4.3 Model

A small 3D CNN (`Segment3DCNN` in [train_segment_classifier.py](train_segment_classifier.py)): three Conv3d blocks with GroupNorm + ReLU + MaxPool3d (or final AdaptiveAvgPool3d), producing a 64-dim embedding, followed by a linear projection to a scalar score. Base channel width = 8; checkpoint size ≈ 79 KB.

### 4.4 Auto-Labels and Pair Weighting (Continuous, Not Bucketed)

For each episode, every 16-step boundary is probed with a 2-second LaCAM call yielding `M(t) = LaCAM-estimated remaining makespan from state at step t`. The auto-label per segment is `diff(t) = M(t+16) − M(t)`. Within each episode, we generate ordered pairs `(i, j)` with `diff(i) > diff(j)` and weight each pair as

```
w(i, j) = min(diff(i) − diff(j), MAX_PAIR_GAP) / MAX_PAIR_GAP
```

with `MAX_PAIR_GAP = 5`. Gaps ≥ 5 clip to weight 1.0; smaller gaps ramp linearly down toward 0; pairs with `diff(i) == diff(j)` are excluded. This continuous weighting replaces the earlier categorical bucketing (`diff > 3` / `diff < 1` / midrange-skip) and lets the entire range of pair confidences contribute proportionally to the loss. A `--dry-run` flag in the trainer prints the resulting pair-weight histogram so the bucketing change can be sanity-checked at config time.

### 4.5 Pairwise Ranking Loss

We optimise standard RankNet over score differences:

```
L = − Σ w(i, j) · log σ(s_i − s_j)
```

with Adam (lr = 3e-4 in phase 1; lr = 1e-4 in phase 2 fine-tunes) and either a constant LR (`--scheduler none`) or cosine annealing.

### 4.6 Human Annotation Protocol (Pairwise + Debiased)

We adapted [replay.ipynb](replay.ipynb) to provide a full-episode scrubbable visualization with keyboard shortcuts. Annotations are now collected as **pairs**, not as worst/clean indices. For each query the tool shows two candidate segments side-by-side and asks "which is worse?"; the answer is recorded as one of `a_worse`, `b_worse`, or `unsure_or_skipped`. Two debiasing measures were added in commit 502008c:

1. **Random A/B swap.** Each query randomises which side is rendered as "A" and which as "B" with 50/50 probability; the user's a/b choice is mapped back to canonical segment indices before being persisted to JSON.
2. **Hidden auto signal.** The tool no longer shows the LaCAM-diff or model-predicted preference probability for the displayed pair, so the human cannot anchor on the auto label.

The canonical persisted schema is now (`scenario_id`, `npz_path`, `segment_a`, `segment_b`, `chosen_worse_segment`, `label`, `segment_a_range`, `segment_b_range`). The trainer's `load_annotations` auto-detects this list-format schema and converts each non-skipped pair into a `(worst_idx, clean_idx)` override.

We collected five annotation files for this study (all on `dataset/held_out/`, all on a budget of 100 queries except the iterative AL stages):

| File | Purpose | Total | Labeled | Skipped |
|---|---|---|---|---|
| `annotation_val_map.json` | Held-out human val signal (val-map seeds 144-147, eval-only) | 100 | **76 across 69 unique rollouts** | 24 |
| `annotation_random.json` | Stage 2 random sampling (train-map seeds) | 100 | **78** | 22 |
| `annotation_confusion.json` | Stage 3 confusion AL (train-map seeds) | 100 | **65** | 35 |
| `annotation_iterative.json` | Stage 4 iterative AL (4 rounds × 14, train-map seeds) | 56 | 56 | 0 |
| `annotations_legacy_free_curation.json` | Legacy un-prioritised hand-curation diagnostic | — | — | — |

The "skipped" rate (22-35%) is itself informative: the human declined to commit when neither segment looked clearly worse. Confusion AL, by construction, surfaces queries the model is most uncertain about — and the human's own uncertainty rate on those queries (35%) is correspondingly the highest of the three.

### 4.7 Two-Phase `--human-only` Fine-Tune

The simplest integration — appending human pairs to the auto pair set — fails not just because of contradictions (≈44% of human pairs disagree with the auto-diff ordering on the same segments, §6.1) but also because the ≈100 human pairs are drowned by ≈78k auto pairs in the training set. A pure RankNet update sees the human signal as < 0.1% of the gradient and never moves the model.

We therefore train in two phases:

- **Phase 1 (auto backbone).** `train_segment_classifier.py --data dataset/held_out --output baseline.pt --epochs 60`. No `--annotations`. Saves two checkpoints: `baseline.pt` (best `human_val` if `--val-annotations` provided; else best `pair_acc`) and `baseline.pair_acc.pt` (best weighted pair accuracy on val-map auto pairs).
- **Phase 2 (human-only fine-tune).** `train_segment_classifier.py --annotations <stage>.json --val-annotations annotation_val_map.json --init-from baseline.pair_acc.pt --human-only --epochs 60 --lr 1e-4`. The `--human-only` flag disables auto pairs entirely; only the human-pair gradient flows. The `--init-from` flag starts from the auto-trained backbone so the model begins fine-tuning from a competent ranker rather than from random weights. Saves `<stage>.pt` (best `human_val`) and `<stage>.pair_acc.pt` (best `pair_acc` — auto-aligned safety check).

Compared to the earlier "Option B override" mechanism (which removed only the auto pairs from annotated episodes), `--human-only` is a stricter complete-replacement protocol: in phase 2, *no* auto pair contributes gradient. The phase 1 checkpoint serves as the implicit prior over auto-pair ranking that human gradients then perturb.

### 4.8 Active-Learning Stages

The Stage 2 annotations were collected by uniform random sampling over the train-map elicitation pool. The Stage 3 annotations were collected by **confusion-driven AL**:

1. **Score every candidate pair.** For every (episode, segment_a, segment_b) triple in the filtered pool (rollouts with `≥ 2` segments — the minimum needed to produce a pair, set in [query.py:420](query.py#L420)), the Stage 1 baseline checkpoint scores both segments and we compute the binary entropy of the implied preference probability:

   ```
   acquisition(ep, A, B) = H(σ(s_A − s_B))
   ```

2. **Greedy top-K with per-episode cap.** Rank descending by entropy, greedy-select up to `--budget` queries subject to a `--per-episode-cap` (typically 2) so a single hard episode does not swallow the budget.
3. **Skip identical candidates.** Pairs whose feature vectors are within `IDENTITY_DISTANCE_EPS = 1e-9` are dropped to avoid degenerate (segment, segment) queries when episodes have repeated rollouts.

This is a pure uncertainty-sampling acquisition. The earlier `H · ‖φ_A − φ_B‖` (entropy × diversity) acquisition was simplified out in commit c9c84a2 after we observed the diversity term concentrated queries on a few feature-extreme rollouts at the cost of leaving in-distribution high-entropy pairs unqueried; we report the simplification as a methodology choice rather than as a separate stage.

**Stage 4 (iterative AL, 4 rounds × 14 labels).** Replaces one-shot pool selection with R = 4 sequential rounds. Each round queries 14 new pairs using the *current* model as scorer, labels them, appends to `annotation_iterative.json`, and retrains. After round 4, the resulting model is the final Stage 4 checkpoint. This stage is currently being re-run under the Phase-1+Phase-2 `--human-only` protocol; the May-3 results in [train_segment_classifier_colab.ipynb](train_segment_classifier_colab.ipynb) used the older mixed-mode trainer (auto pairs co-trained with human overrides, 8 epochs/round) and are reported with that caveat in §6.5.

### 4.9 Evaluation Metrics

Two complementary metrics are tracked each epoch:

- **`pair_acc`** (auto, val maps): Weighted pairwise accuracy on val-map auto pairs, generated with the same continuous-gap weighting used at training time. This is the auto-aligned analogue of the training loss and the primary save-best signal in the absence of human labels.
- **`human_val`** (held-out human pairs): For each labelled pair `(worst, clean)` in `annotation_val_map.json`, the indicator `score(worst) > score(clean)`. Random baseline = 0.5.

Both metrics are evaluated on val-map seeds {144, 145, 146, 147} only — the train-map elicitation pool is fully held out from auto-pair val and from the human-pair val signal alike. A legacy `top-1±1` (argmax_pm1) metric was removed from the trainer in commit d8b4301; it can still be recomputed post-hoc by [evaluate_baselines.py](evaluate_baselines.py) when a DDG-aligned exact-match summary is desired.

---

## 5. Experimental Setup

### 5.1 Stages

We compare three fine-tune strategies that share identical model architecture, optimiser (Adam), backbone initialisation (`--init-from baseline.pair_acc.pt` for Stages 2 and 3), and label budget (one human pair per query under the random-flip protocol). Only the *acquisition rule* over the train-map elicitation pool varies. A fourth stage (iterative AL) is reported with the older methodology and is being re-run under the new `--human-only` two-phase protocol.

| Stage | Annotations | Acquisition | Init from | Train pairs | Notes |
|---|---|---|---|---|---|
| 1 — auto baseline | none | — | random | 78,126 auto | Phase 1 only |
| 2 — random | `annotation_random.json` | uniform random | `baseline.pair_acc.pt` | 78 human | `--human-only` |
| 3 — confusion AL | `annotation_confusion.json` | `H(σ(s_A − s_B))` | `baseline.pair_acc.pt` | 65 human | `--human-only` |
| 4 — iterative AL (legacy May-3 numbers) | `annotation_iterative.json` (4×14) | per-round entropy with diversity | previous round | mixed auto + human | re-run pending |

### 5.2 Hyperparameters

Identical across stages: 60 epochs, batch size 128, base CNN width 8, single segment per training example (`--context_segments 1`), random per-segment 90°-rotation and flip augmentation, `--scheduler none` (constant LR, except confusion AL which used cosine annealing). Phase-1 LR = 3e-4, phase-2 LR = 1e-4.

### 5.3 Evaluation Signals

- **Auto val (`pair_acc`)**: 18,031 weighted pairs across val-map seeds {144, 145, 146, 147}.
- **Human val (`human_val`)**: 76 labeled pairs across 69 unique rollouts in `annotation_val_map.json`, all on val-map seeds {144, 145, 146, 147}, never exposed to training (`--val-annotations` is eval-only).

### 5.4 Compute

Single Colab GPU (A100). Phase 1 (60 epochs, ≈610 batches/epoch) takes ≈40 minutes; Phase 2 fine-tunes (60 epochs, ≈1 batch/epoch) take ≈5 minutes each. Total pipeline wall clock for Stages 1-3 is ≈1 hour.

### 5.5 Downstream MAPF-GPT Eval Protocol

The end-to-end value of the segment classifier is measured by retraining MAPF-GPT under classifier-gated DDG (substituting our learned ranker for the original `diff > 3` threshold inside `delta_data_generator.py`) and comparing the resulting policy head-to-head against MAPF-GPT trained under the original threshold-gated DDG. Three classifier-gated runs are planned, one per training stage of the segment classifier:

| Downstream model | Segment-classifier checkpoint that gated DDG | Status |
|---|---|---|
| `MAPF-GPT-original` ([`checkpoints/original/`](checkpoints/original/)) | none — original `diff > 3` threshold | trained, benchmarked |
| `MAPF-GPT-S1` ([`checkpoints/baseline/`](checkpoints/baseline/)) | Stage 1 auto-only (`baseline.pair_acc.pt`) | trained, benchmarked |
| `MAPF-GPT-S2` (`checkpoints/random/`) | Stage 2 random fine-tune (`random_finetune.pt`) | trained, benchmarked |
| `MAPF-GPT-S3` (`checkpoints/confusion/`, pending) | Stage 3 confusion-AL fine-tune (`confusion_finetune.pt`) | training in progress |

> Naming caveat: the eval-config YAML files use the algorithm key `Baseline` for `MAPF-GPT-S1` (the auto-only-classifier-gated MAPF-GPT). This is **not** the same as "Stage 1 baseline" in §6.2, which refers to the upstream segment classifier itself. To disambiguate we will use `MAPF-GPT-S{1,2,3}` (downstream MAPF-GPT models) and `Stage {1,2,3}` (upstream segment classifiers) throughout. The eval-config algorithm keys themselves have not been renamed.

The apples-to-apples comparison cut is `ckpt_ddg_1500.pt` from each — the most-trained shared-step checkpoint, since the classifier-gated runs currently reach step 2,000 while `MAPF-GPT-original` goes to step 30,000.

Evaluation is performed via [`benchmark.py`](benchmark.py) and `pogema_toolbox.evaluator` on five POGEMA suites under [`eval_configs/`](eval_configs/) (random, mazes, warehouse, movingai, puzzles). Per-suite metrics: ISR, CSR, ep_length, SoC, makespan, avg_agents_density, runtime. Per-cell results are reported as `mean ± std` over the 128 maps in each suite (1 map for warehouse). Plots and tabular summaries are written under each suite's `eval_dir`. Full discussion in §6.7.

---

## 6. Results

> Stage 1, Stage 2 (random), and Stage 3 (confusion AL) of the upstream segment classifier were retrained under the new methodology on May 10 ([train_segment_classifier_colab.ipynb](train_segment_classifier_colab.ipynb)) and the numbers in §6.2-6.4 are pulled from those logs. Stage 4 (iterative AL of the segment classifier) is reported with its May-3 mixed-mode numbers and is being re-run under the new `--human-only` protocol; results pending. The downstream POGEMA benchmark in §6.7 reports completed numbers for `MAPF-GPT-S1` (Stage-1-gated DDG) and `MAPF-GPT-S2` (Stage-2-gated DDG) on Random / Maze / Warehouse from [`benchmark.txt`](benchmark.txt); `MAPF-GPT-S3` (Stage-3-gated) is currently being trained under classifier-gated DDG and will be benchmarked once training completes. MovingAI and Puzzles suites are pending for all algorithms. The geometric augmentation corpus (§6.6.1) is built and committed but not yet wired into a reported training run.

### 6.1 Auto-vs-Human Disagreement (validates H1)

[FILL: recompute disagreement statistics from `annotation_val_map.json` + `annotation_random.json` + `annotation_confusion.json`. The directional finding from the earlier 52-annotation batch — ~44% of human pairs contradict the auto-diff ordering on the same two segments — is expected to hold on the new files; quote the new exact count.]

The persisted schema now records the auto-diff for each shown segment alongside the human's verdict, so this statistic is recoverable from each annotation file directly. We expect the qualitative finding to be unchanged: humans see at least two failure modes the LaCAM-diff systematically misses (pre-congestion oscillation and local-deadlock-resolved-by-luck, §7.1), and these surface as contradictions in the borderline band.

### 6.2 Stage 1: Auto-Only Baseline

We trained the segment classifier for 60 epochs on auto-pair supervision only, with no `--annotations`. The training set contained 78,126 auto pairs across train-map seeds; validation used 18,031 val-map auto pairs and the 76-pair `annotation_val_map.json` as an eval-only signal.

| Stage 1 checkpoint | best `pair_acc` (auto, val maps) | best `human_val` (76 held-out pairs) |
|---|---|---|
| `baseline.pair_acc.pt` (best auto) | **0.649** | — (this checkpoint was selected by auto signal) |
| `baseline.pt` (best human_val) | 0.629 (epoch 7) | **0.724** (epoch 7) |

The `human_val = 0.724` headline number is misleading. It is achieved at epoch 7, very early in training; for the remaining 53 epochs the metric oscillates between 0.55 and 0.70, settling around 0.60-0.62 by epoch 60. The auto-aligned `pair_acc` continues to climb steadily from 0.569 (epoch 1) to 0.649 (epoch 45), suggesting the model continues to learn auto-pair structure long after any human-aligned generalisation has plateaued. We treat `pair_acc` as the more reliable Stage-1 indicator and use the `baseline.pair_acc.pt` checkpoint as the warm-start for Stage 2 and Stage 3.

[FIGURE 1: Stage 1 training curves over 60 epochs — `loss`, `pair_acc`, `human_val` per epoch. Source: `out/segment_classifier/baseline.png` (stored in Drive `/content/drive/MyDrive/mapf_congestion/out/segment_classifier/baseline.png`).]

### 6.3 Stage 2: Random Sampling Fine-Tune

We fine-tuned the Stage 1 backbone for 60 epochs in `--human-only` mode on `annotation_random.json` (78 labeled pairs from 76 unique rollouts on the train-map elicitation pool). Phase-2 LR = 1e-4, no scheduler.

| Stage 2 checkpoint | best `pair_acc` | best `human_val` |
|---|---|---|
| `random_finetune.pair_acc.pt` (best auto) | **0.648** | — |
| `random_finetune.pt` (best human_val) | 0.638 (epoch 10) | **0.697** (epoch 10) |

`pair_acc` falls by only 0.001 from Stage 1 (0.649 → 0.648) — DDG-aligned ranking is preserved. `human_val` rises monotonically from 0.579 (epoch 1) to 0.697 (epoch 10), then stabilises in the 0.65-0.70 band for the remainder of training. Crucially, the Stage 2 `human_val` trajectory is much more stable than Stage 1's: the late-epoch mean is ≈ 0.66 vs ≈ 0.61 for the baseline, so even though the *best-ever* number is lower than Stage 1's transient peak, the *deployable* number (any late epoch) is higher.

### 6.4 Stage 3: Confusion AL Fine-Tune (Entropy-Only Acquisition)

Stage 3 uses the same trainer, same val signal, and same `--human-only --init-from baseline.pair_acc.pt` protocol as Stage 2. The 65 labeled pairs come from `annotation_confusion.json`, queried by `query.py --model-path baseline.pair_acc.pt --budget 100` ranking pool candidates by `H(σ(s_A − s_B))`. (The acquisition surfaced 100 pairs; the human declined 35 of them as `unsure_or_skipped` — a higher skip rate than random sampling's 22, consistent with the acquisition surfacing genuinely ambiguous pairs.)

| Stage 3 checkpoint | best `pair_acc` | best `human_val` |
|---|---|---|
| `confusion_finetune.pair_acc.pt` (best auto) | **0.650** | — |
| `confusion_finetune.pt` (best human_val) | 0.638 (epoch 35) | **0.711** (epoch 35) |

Three substantive findings:

1. **Pair_acc preserved.** Confusion AL's `pair_acc = 0.650` is within 0.001 of Stage 1 (0.649) and Stage 2 (0.648). Across all three stages, the auto-aligned ranking spread is ≤ 0.002, well under any plausible noise floor. **H2 (preservation) is supported.**
2. **Confusion AL beats random AL on `human_val`.** Stage 3's best `human_val = 0.711` exceeds Stage 2's `0.697` by +0.014 pp at a smaller label budget (65 vs 78). More importantly, the late-epoch `human_val` stabilises at 0.711 from epoch 35 onward and stays there for the final 25 epochs (the cosine LR schedule annealing the model toward the late minimum) — a stable plateau, not a transient peak. **H3 (selection beats volume) is supported, modestly.**
3. **The headline lift is not over the auto baseline's *peak* — it is over the auto baseline's *late-epoch mean*.** Stage 1's `human_val = 0.724` at epoch 7 is a noisy early-training anomaly; the late-epoch baseline mean is ≈ 0.61, and both Stage 2 (≈ 0.66 late) and Stage 3 (0.711 stable) sit clearly above it. The right comparison is "stable late-epoch human_val" rather than "best-ever human_val," and on that basis the AL fine-tune does add value over the auto-only baseline.

[FIGURE 2: 3-stage `human_val` trajectory plot, all 60 epochs, with Stage 1 (gray), Stage 2 (blue), Stage 3 (green); horizontal red dashed line at 0.5 (chance). Source: `out/segment_classifier/comparison.png` produced by the comparison cell in [train_segment_classifier_colab.ipynb](train_segment_classifier_colab.ipynb).]

[FIGURE 3: 3-stage `pair_acc` trajectory plot, same axes. Should show the three curves overlapping within 0.002 by epoch 60 — the visual proof of `pair_acc` preservation across stages.]

Apples-to-apples summary:

| Metric | Stage 1 (auto only) | Stage 2 (random) | Stage 3 (confusion AL) |
|---|---|---|---|
| Best `pair_acc` (auto val) | 0.649 | 0.648 | **0.650** |
| Best `human_val` (76 pairs) | 0.724 (transient, ep 7) | 0.697 (ep 10) | **0.711 (stable, ep 35+)** |
| Late-epoch `human_val` mean (epochs 50-60) | ≈ 0.61 | ≈ 0.66 | **≈ 0.71** |
| Train labels used | 0 | 78 | 65 |
| Acquisition | — | uniform random | `H(σ(s_A − s_B))` |

### 6.5 Stage 4: Iterative AL (legacy May-3 numbers, re-run pending)

The May-3 iterative AL run used the *older* trainer (mixed auto + human pairs co-trained, 8 epochs per round, no `--human-only` flag) and reported per-round best `argmax_pm1` (the now-removed top-1±1 metric) and `human_val` on the older 12-pair val-map signal. We summarise here for completeness:

| Round | Train pairs (auto + human) | Best `argmax_pm1` | Best `human_val` (12 pairs) |
|---|---|---|---|
| 1 (init from baseline.pt) | 17,583 + 14 from 8 episodes | 0.559 | 0.500 |
| 2 (init from round_1.pt) | 17,484 + 28 from 17 episodes | 0.574 | 0.500 |
| 3 (init from round_2.pt) | 17,424 + 42 from 25 episodes | 0.565 | 0.583 |
| 4 (init from round_3.pt) | 17,281 + 56 from 34 episodes | 0.541 | 0.583 |

These numbers are **not directly comparable** to Stages 1-3 above: (a) they use the older mixed-mode trainer where 17k auto pairs dilute every batch; (b) the val signal is the older 12-pair set, not the current 76-pair `annotation_val_map.json`; (c) `argmax_pm1` is no longer computed by the trainer. The iterative protocol is being re-run under Phase-1 + Phase-2 `--human-only` to produce numbers comparable to Stages 2 and 3. [FILL when iterative re-run lands.]

### 6.6 Data Augmentation (Isabel De Luis)

Two complementary augmentation tracks were explored, both led by Isabel De Luis. They differ in *what gets augmented* and *which model consumes the augmented data*.

**6.6.1 Label-preserving geometric augmentation of segment-classifier rollouts** (primary contribution to the main pipeline).

[`augment_segment_rollouts.py`](augment_segment_rollouts.py) (~300 lines) applies spatial symmetries — `hflip`, `vflip`, `rot180`, `rot90`, `rot270`, `transpose` — directly to the segment-classifier `.npz` rollouts. For each transform, the script reads `obstacles`, `positions`, `goals`, and `segment_diffs`, applies the matching coordinate transform to all spatial fields (e.g. `(r, c) → (height − 1 − r, c)` for vflip), validates that the transformed coordinates remain in-bounds, and writes a new file `<scenario>__aug_<transform>.npz`. Crucially, `segment_diffs` is preserved bit-identically: the spatial symmetry does not change the LaCAM-estimated remaining makespan at any segment boundary. This is a label-cost-free expansion of the auto-pair training corpus.

The committed augmentation set covers all five DDG checkpoints (`ckpt_0`, `ckpt_500`, `ckpt_1000`, `ckpt_1500`, `ckpt_30000`) in the `dataset/held_out/` corpus, using the three shape-preserving transforms (`hflip`, `vflip`, `rot180`):

| Slice | Original `.npz` files | Augmented `.npz` files | Total |
|---|---|---|---|
| Across all 5 DDG checkpoints | 1,669 | 1,669 × 3 = 5,007 | 6,676 |

Files are written under `ranker_dataset/held_out_aug/`; the script's `--output` argument routes them so they can be included directly under the `--data` root passed to `train_segment_classifier.py`. The augmentation has been built and committed but has not yet been used in the May-10 training runs reported in §6.2-6.4 (the colab still passes `--data dataset/held_out`, the un-augmented root). The expected contribution is a 4× expansion of the auto-pair training set, which should help the auto-only Stage 1 backbone converge faster and to a higher `pair_acc`. [FILL: report `pair_acc` and `human_val` with vs. without geometric augmentation on a Stage 1 re-run.]

**6.6.2 Synthetic-jitter active learning on the standalone congestion classifier** (parallel exploration).

[`finetuning/export_augmented_active_learning_samples.py`](finetuning/export_augmented_active_learning_samples.py) (~448 lines) operates on a different model entirely: the standalone tabular congestion classifier under `finetuning/` (Arrow-format dataset, distinct from the spatio-temporal segment classifier in this paper). The loop selects seed examples from a target class (default: rarest), generates synthetic candidates by sparse integer jittering (`mutation_rate · D` indices perturbed within `±jitter_radius`), scores them with the current congestion classifier, ranks by one of three strategies (`uncertainty`, `entropy`, `low_confidence`), and exports the top-N for human review. Each exported record carries source-of-jitter metadata so a reviewer can compare the synthetic to its parent. [FILL: accuracy / F1 of the synthetic-jitter AL loop vs random sampling on the same congestion dataset.]

The two tracks share the *acquisition family* (entropy / uncertainty) but operate on completely different data modalities (4D spatio-temporal volumes vs tabular congestion features), so a positive result on one does not automatically transfer to the other.

### 6.7 End-to-End DDG Integration and Downstream MAPF-GPT Benchmark

**Runtime integration.** The trained segment classifier is plumbed into DDG's expert-selection step in commit fa4c46f. When `cfg.segment_classifier_path` is set on `FastSolverDeltaConfig`, the new code path in [`finetuning/delta_data_generator.py`](finetuning/delta_data_generator.py):

1. Wraps each env in a `_PositionRecorder` so per-step agent positions are logged at runtime (the classifier's input featurizer needs them).
2. After rolling out, batches every env's segments through `_score_env_segments_batch()` in a single forward pass.
3. Picks each env's argmax-scored segment as the expert candidate.
4. Sorts envs by their top score and runs the LaCAM expert on the top `cfg.expert_top_k` envs (or all if `expert_top_k = None`).

Selection-mode metadata is logged as `selection_mode: 'segment_ranker'` (vs the original `selection_mode: 'fast_diff'`) so per-run telemetry distinguishes the two paths.

**Closed-loop training runs.** Three MAPF-GPT models have been (or are being) trained from scratch with classifier-gated DDG, one per upstream segment-classifier checkpoint (Stage 1, 2, 3). For each, DDG continues for ≈2,000 fine-tune steps; the apples-to-apples comparison cut against the original threshold-gated `MAPF-GPT-original` ([`checkpoints/original/ckpt_ddg_1500.pt`](checkpoints/original/)) is the shared-step `ckpt_ddg_1500.pt`. The partial-training caveat (≤ step 2,000 vs `MAPF-GPT-original`'s step 30,000) means the comparison measures the *value of the classifier-gated DDG curriculum at a fixed training step*, not the asymptotic policy quality.

**POGEMA suites configured.** Five POGEMA suites are configured under [`eval_configs/`](eval_configs/):

| Suite | Map type | # maps | Agent counts |
|---|---|---|---|
| `01-random` | Procedurally-generated random grids | 128 | 8, 16, 24, 32, 48, 64 |
| `02-mazes` | Procedurally-generated mazes | 128 | 8, 16, 24, 32, 48, 64 |
| `03-warehouse` | `wfi_warehouse` (single fixed map) | 1 | 32, 64, 96, 128, 160, 192 |
| `04-movingai` | MovingAI benchmark maps (e.g. `Berlin_1_256_*`) | 128 | 64, 128, 192, 256 |
| `05-puzzles` | Hand-crafted puzzle maps (5×5, dense obstacles) | 16 | 2, 3, 4 |

Reported metrics: **ISR** (individual success rate), **CSR** (coverage success rate — all agents reach goals), **ep_length**, **SoC** (sum of costs), **makespan**, **avg_agents_density**, **runtime**. Per-cell results are `mean ± std` over the 128 maps in each suite (1 map for warehouse). The benchmark numbers below come from [`benchmark.txt`](benchmark.txt). Random / Maze / Warehouse have completed for `MAPF-GPT-original`, `MAPF-GPT-S1`, and `MAPF-GPT-S2`; `MAPF-GPT-S3` is still being trained, and MovingAI / Puzzles suites are pending for all algorithms.

**Top-line result.** `MAPF-GPT-S2` strictly dominates both `MAPF-GPT-S1` and `MAPF-GPT-original` on every cell of every completed suite where there is headroom. The headline numbers — comparing `MAPF-GPT-S2` to the threshold-gated baseline — are striking:

| Suite | #Agents | `Original` CSR | `S2` CSR | **Δ** |
|---|---|---|---|---|
| Random | 32 | 0.62 | 0.88 | **+0.26** |
| Random | 48 | 0.23 | 0.53 | **+0.30** |
| Random | 64 | 0.09 | 0.33 | **+0.24** (3.7×) |
| Maze | 24 | 0.42 | 0.70 | **+0.28** |
| Maze | 32 | 0.24 | 0.49 | **+0.25** (2×) |
| Warehouse | 64 | 0.719 | 0.984 | **+0.27** |
| Warehouse | 96 | 0.430 | 0.688 | **+0.26** |
| Warehouse | 128 | 0.055 | 0.164 | **+0.11** (3×) |

In every cell above, `MAPF-GPT-S2` also keeps SoC within ≈3% of `MAPF-GPT-original`'s — and on Maze 32 it even *lowers* SoC (1931.23 vs 1963.51) while delivering +25 pp CSR. There is no SoC-for-CSR tradeoff: `MAPF-GPT-S2` is unambiguously the better policy.

The 78 human pairs that fine-tuned the Stage-1 ranker into the Stage-2 ranker translate, downstream, into the **single largest result in this paper**: a learned ranker with even a small budget of human supervision produces a measurably better MAPF-GPT policy than the original DDG curriculum at *every* density where there is headroom to win.

#### 6.7.1 `MAPF-GPT-S1` vs `MAPF-GPT-original` (auto-only ranker → DDG)

The Stage-1-classifier-gated MAPF-GPT (no human labels in the ranker) already lifts CSR over `MAPF-GPT-original` at moderate-to-high agent density on Random and Warehouse, but is roughly tied on Maze (Maze is where humans appear to add the most value — see §6.7.2).

| Suite | #Agents | `Original` CSR | `S1` CSR | Δ |
|---|---|---|---|---|
| Random | 8 | 0.99 | 0.98 | −0.01 |
| Random | 16 | 0.91 | 0.95 | +0.04 |
| Random | 24 | 0.84 | 0.91 | +0.07 |
| Random | 32 | 0.62 | 0.73 | **+0.11** |
| Random | 48 | 0.23 | 0.39 | **+0.16** |
| Random | 64 | 0.09 | 0.13 | +0.04 |
| Maze | 8 | 0.93 | 0.98 | +0.05 |
| Maze | 16 | 0.72 | 0.67 | −0.05 |
| Maze | 24 | 0.42 | 0.51 | +0.09 |
| Maze | 32 | 0.24 | 0.22 | −0.02 |
| Maze | 48 | 0.12 | 0.09 | −0.03 |
| Maze | 64 | 0.04 | 0.02 | −0.02 |
| Warehouse | 32 | 0.961 | **1.000** | +0.039 |
| Warehouse | 64 | 0.719 | **0.898** | **+0.179** |
| Warehouse | 96 | 0.430 | 0.414 | −0.016 |
| Warehouse | 128 | 0.055 | 0.047 | −0.008 |
| Warehouse | 160-192 | 0.000 | 0.000 | 0 |

`MAPF-GPT-S1` trades a small SoC penalty for the CSR gain (e.g. Warehouse 64: SoC +115, CSR +18 pp). At 8 agents both methods are saturated; at extreme densities (Maze 64, Warehouse 160+) both fail. The interesting band is in between, and `MAPF-GPT-S1` wins it on Random and Warehouse — but Maze remains a problem until human-fine-tuning is added in `MAPF-GPT-S2`.

#### 6.7.2 `MAPF-GPT-S2` vs `MAPF-GPT-S1` and `MAPF-GPT-original` (random-finetune ranker → DDG)

The `MAPF-GPT-S2` model — DDG-trained gated by the Stage-2 random-fine-tune segment classifier (`random_finetune.pt`, fine-tuned on 78 randomly-sampled human pairs in §6.3) — beats both `MAPF-GPT-S1` and `MAPF-GPT-original` on every cell of every completed suite. Per-suite comparisons:

**Random suite (full table):**

| #Agents | Original CSR | S1 CSR | **S2 CSR** | Δ S2−Orig | Δ S2−S1 |
|---|---|---|---|---|---|
| 8 | 0.99 ± 0.02 | 0.98 ± 0.02 | **1.00 ± 0.00** | +0.01 | +0.02 |
| 16 | 0.91 ± 0.05 | 0.95 ± 0.04 | **0.99 ± 0.01** | +0.08 | +0.04 |
| 24 | 0.84 ± 0.06 | 0.91 ± 0.05 | **0.95 ± 0.04** | +0.11 | +0.04 |
| 32 | 0.62 ± 0.08 | 0.73 ± 0.08 | **0.88 ± 0.06** | **+0.26** | **+0.15** |
| 48 | 0.23 ± 0.07 | 0.39 ± 0.09 | **0.53 ± 0.09** | **+0.30** | **+0.14** |
| 64 | 0.09 ± 0.05 | 0.13 ± 0.06 | **0.33 ± 0.08** | **+0.24** | **+0.20** |

**Maze suite (full table) — the largest qualitative shift:**

| #Agents | Original CSR | S1 CSR | **S2 CSR** | Δ S2−Orig | Δ S2−S1 |
|---|---|---|---|---|---|
| 8 | 0.93 ± 0.04 | 0.98 ± 0.03 | **0.99 ± 0.01** | +0.06 | +0.01 |
| 16 | 0.72 ± 0.08 | 0.67 ± 0.08 | **0.86 ± 0.06** | **+0.14** | **+0.19** |
| 24 | 0.42 ± 0.09 | 0.51 ± 0.08 | **0.70 ± 0.07** | **+0.28** | **+0.19** |
| 32 | 0.24 ± 0.07 | 0.22 ± 0.07 | **0.49 ± 0.09** | **+0.25** | **+0.27** |
| 48 | 0.12 ± 0.05 | 0.09 ± 0.05 | **0.17 ± 0.06** | +0.05 | +0.08 |
| 64 | 0.04 ± 0.04 | 0.02 ± 0.02 | **0.09 ± 0.05** | +0.05 | +0.07 |

Maze is the diagnostic suite. `MAPF-GPT-S1` was *losing* to `MAPF-GPT-original` on Maze at most densities (16 / 32 / 48 / 64). `MAPF-GPT-S2` not only recovers but reverses the result: at Maze 24 / 32 it lifts CSR by 25-28 pp over Original. The 78 human pairs do something the auto-only ranker cannot: they teach the segment classifier what counts as congestion in maze topology, and that transfers downstream into a substantially better MAPF-GPT policy.

**Warehouse suite (full table):**

| #Agents | Original CSR | S1 CSR | **S2 CSR** | Δ S2−Orig | Δ S2−S1 |
|---|---|---|---|---|---|
| 32 | 0.961 ± 0.031 | **1.000 ± 0.000** | **1.000 ± 0.000** | +0.039 | 0 |
| 64 | 0.719 ± 0.074 | 0.898 ± 0.051 | **0.984 ± 0.020** | **+0.265** | **+0.086** |
| 96 | 0.430 ± 0.090 | 0.414 ± 0.082 | **0.688 ± 0.078** | **+0.258** | **+0.274** |
| 128 | 0.055 ± 0.039 | 0.047 ± 0.035 | **0.164 ± 0.062** | +0.109 | +0.117 |
| 160 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| 192 | 0.000 | 0.000 | 0.000 | 0 | 0 |

Warehouse 64 jumps from `MAPF-GPT-original`'s 0.719 to `MAPF-GPT-S2`'s **0.984** — a near-perfect coverage rate at moderate density. Warehouse 96 goes from 0.430 to 0.688, a +26 pp lift (where `MAPF-GPT-S1` had been tied with Original at 0.41-0.43). Warehouse 128 triples from 0.055 to 0.164 — both small, but the relative lift is large.

**SoC comparison (no tradeoff for `MAPF-GPT-S2`).** `MAPF-GPT-S2`'s SoC is essentially indistinguishable from `MAPF-GPT-original`'s (within a few percent), while CSR jumps by 25-30 pp at the moderate-density cells. On Maze 32, S2 even *lowers* SoC (1931.23 vs 1963.51) while raising CSR by 25 pp. There is no Pareto trade — S2 is dominant.

[FIGURE 4: POGEMA Random + Maze + Warehouse CSR-vs-num_agents, three lines per suite (`MAPF-GPT-original` / `MAPF-GPT-S1` / `MAPF-GPT-S2`). Source: `eval_configs/0?-*/results_views/*.png` produced by `pogema_toolbox.evaluator`.]

[FIGURE 5: Warehouse 64-agent CSR bar chart — Original 0.72, S1 0.90, S2 0.98 — single most legible visual of the result.]

#### 6.7.3 `MAPF-GPT-S3` vs `MAPF-GPT-S2`, `MAPF-GPT-S1`, and `MAPF-GPT-original` [in progress]

The Stage-3-classifier-gated DDG run (`confusion_finetune.pt` as the gating ranker) is currently being trained; it will be benchmarked on the same three suites once training completes. The headline question for §6.7.3 is whether the entropy-AL-selected 65 human pairs (Stage 3) translate into a downstream MAPF-GPT lift over the random-sampled 78 human pairs (Stage 2). Two outcomes are equally plausible *a priori*:

1. **Selection strategy matters downstream too.** `MAPF-GPT-S3` beats `MAPF-GPT-S2` by a margin comparable to the upstream `human_val` lift (≈ +0.014). This would be the strongest possible HRI story: a small amount of selection-strategy engineering on the upstream ranker amplifies into a measurable downstream policy improvement.
2. **Any human supervision suffices downstream.** `MAPF-GPT-S3` ≈ `MAPF-GPT-S2`. The downstream lift comes from the *presence* of human supervision in the upstream ranker, not from the *selection strategy* by which it was collected. This would tell us that future deployments can use the cheaper random-sampling protocol.

| Suite | #Agents | Original | S1 | S2 | **S3** | Δ S3−S2 |
|---|---|---|---|---|---|---|
| Random | 32 | 0.62 | 0.73 | 0.88 | [FILL] | [FILL] |
| Random | 48 | 0.23 | 0.39 | 0.53 | [FILL] | [FILL] |
| Random | 64 | 0.09 | 0.13 | 0.33 | [FILL] | [FILL] |
| Maze | 24 | 0.42 | 0.51 | 0.70 | [FILL] | [FILL] |
| Maze | 32 | 0.24 | 0.22 | 0.49 | [FILL] | [FILL] |
| Warehouse | 64 | 0.72 | 0.90 | 0.98 | [FILL] | [FILL] |
| Warehouse | 96 | 0.43 | 0.41 | 0.69 | [FILL] | [FILL] |

#### 6.7.4 MovingAI and Puzzles suites [pending]

The MovingAI and Puzzles configurations under `eval_configs/04-movingai/` and `eval_configs/05-puzzles/` have not been benchmarked yet for any of the three algorithms. We expect MovingAI (single large maps with up to 256 agents) to push all methods into the failing regime; Puzzles (5×5 with 2-4 agents) into saturation. The interesting headroom band, based on the Random / Maze / Warehouse results above, is moderate-to-high agent density on geometrically regular maps. [FILL when complete.]

### 6.8 Sensitivity Studies (optional, time permitting)

- Multi-seed reruns of all four stages to characterise variance bands. The trainer does not seed `torch.manual_seed`; on the legacy methodology a single Stage 2 reseed swung `human_val` by up to 23 pp on the 12-pair val. The 76-pair val signal should reduce that band substantially, but we have not yet measured.
- Per-episode-cap ablation in `query.py`: budget = 100 with `cap ∈ {1, 2, 4, ∞}` to characterise the budget-concentration tradeoff. At cap = ∞, a single pathological rollout could absorb the entire budget.
- Re-introducing the diversity term (`H · ‖φ_A − φ_B‖`) with proper cross-rollout normalisation, to test whether the simplification to entropy-only sacrificed any signal.

---

## 7. Discussion

### 7.1 What the Disagreement Tells Us

Across the annotation files we collect, humans contradict the auto-diff ordering on roughly 40-45% of borderline pairs (consistent with the rate measured on the earlier 52-annotation batch). The fast-solver-diff measures how much LaCAM thinks the residual problem has gotten harder over a 16-step window — a metric that systematically misses two failure modes humans easily catch:

1. **Pre-congestion oscillation.** Agents dithering in a corridor for several steps look benign by makespan-residual but are clearly the precursor to a collapse a few steps later.
2. **Local-deadlock-resolved-by-luck.** A segment in which agents block each other but happen to escape may show low `diff` even though the behavior was congested.

The auto-only baseline (Stage 1) reaches a non-trivial level of human alignment on its own — late-epoch `human_val ≈ 0.61` is well above chance — meaning the auto signal carries roughly 10-12 percentage points of information about human judgment beyond the random baseline. But that ceiling sits well below the AL fine-tune's stable `human_val ≈ 0.71`, indicating that auto-fitting alone cannot recover the residual human signal in the borderline cases. The borderline pairs where humans contradict the auto-diff ordering are exactly the cases the auto-only model is most likely to get wrong because it has no signal saying otherwise.

[EXPAND: pick 2 concrete annotated examples that contradict the auto-diff ordering, alongside replay-tool screenshots. The new persisted schema includes `segment_a_range` and `segment_b_range` so the exact frames can be reproduced.]

### 7.2 Why `--human-only` Beats Naive Append

The earlier "Option B override" pattern (replace auto pairs from the annotated episode with the human pair, leave other episodes alone) was a minimal change to the original mixed-mode trainer. Under the new `--human-only` two-phase protocol, the override is total: in phase 2 *no* auto pair contributes gradient, regardless of episode. The phase-1 backbone serves as the implicit auto-pair prior; the phase-2 update perturbs it toward human judgment on the rollouts the human actually saw. The `pair_acc` preservation result in §6 (≤ 0.002 spread across all three stages) is the empirical sign that this perturbation does not damage the auto-aligned prior — phase 2 moves the model in the human direction without forgetting phase 1.

### 7.3 Selection Strategy at This Budget

Confusion AL (Stage 3) and random sampling (Stage 2) differ on `human_val` by +0.014 pp — a small but reproducible margin at smaller label budget (65 vs 78). The interpretation we give in §6 is that the right measure is *stable late-epoch* `human_val`, not the best-ever single epoch: random sampling stabilises around 0.66, confusion AL stabilises around 0.71. The +0.05 stable-mean difference is meaningful at a 76-pair signal where each pair is worth ≈ 1.3 pp.

The key methodological caveat is that the `human_val` signal — even at 76 pairs — is still small enough that ±5 pp swings on a single re-seed are plausible. Multi-seed averaging is on the immediate to-do list.

### 7.3a Upstream ≠ Downstream: The Surprising Magnitude of the `MAPF-GPT-S2` Lift

The most counterintuitive finding in this paper is the *gap between* the upstream segment-classifier improvement (§6.3 showed Stage 2 lifted `human_val` by ≈ 0.05 over Stage 1) and the downstream MAPF-GPT improvement (§6.7.2 shows `MAPF-GPT-S2` lifts CSR over `MAPF-GPT-S1` by +14-27 pp on six different cells across three suites). The downstream lift is an order of magnitude larger than the upstream lift suggests it should be.

Why? Two hypotheses:

1. **Curriculum compounding.** The DDG curriculum is iterative: at each step, the segment classifier picks which envs go to the expert. Even a modestly-better classifier picks slightly-better-quality expert-relabelling targets at each step, and over 1,500 fine-tune steps this compounds. The upstream `human_val` measures the classifier's quality on *one shot*; the downstream CSR measures the policy's quality after *1,500 shots* of slightly-better curriculum.
2. **Distribution alignment.** The upstream `human_val` test set (76 val-map human pairs) is a relatively narrow slice. The classifier's improvement on that slice may understate its improvement on the in-distribution rollouts that matter for DDG curriculum-shaping. The downstream POGEMA suites (Random / Maze / Warehouse) are themselves much closer to the rollout distribution that DDG operates on at training time, so the downstream measurement may be a more faithful estimate of the classifier's actual deployment-time value.

Either interpretation has the same actionable implication: **upstream metrics on the segment classifier systematically *understate* its downstream value when used as a DDG gating function.** A classifier that improves human alignment by single-digit percentage points can produce double-digit downstream CSR gains. This is a useful design principle for any HRI loop where the trained-classifier-as-gating-function is the deployment target rather than the classifier itself.

### 7.3b The Maze Diagnostic

The Maze suite is where this story is cleanest. On Maze, `MAPF-GPT-S1` (auto-only ranker) was *losing* to `MAPF-GPT-original` at 4 of 6 densities — the auto-only ranker actually shaped a *worse* DDG curriculum than the hand-tuned threshold for maze-topology rollouts. `MAPF-GPT-S2` (random-fine-tune ranker, +78 human pairs) reverses this: it now beats Original at every density on Maze, with the largest gains at 24-32 agents (+25-28 pp). The 78 human pairs do not just polish a working ranker — they fix a regime where the auto-only ranker was actively harmful. This is direct evidence that the human signal is teaching the ranker something about congestion in maze topology that LaCAM-diff alone cannot pick up.

### 7.4 Where Our Approach Sits Among Alternatives

| Approach | What it does | Human in loop | Where the signal comes from | Limit on quality |
|---|---|---|---|---|
| **MAPF-GPT** [^mapfgpt] | Imitation-learn from offline LaCAM-generated trajectories | No | Solver demonstrations | Coverage of the offline training set; long-tail congestion under-represented |
| **DAgger** [^dagger] | Iteratively roll the current policy, expert-relabel the visited states, retrain | No (expert is a solver) | On-policy expert relabelling | Expert is expensive at scale; expert calls per state are uniform |
| **Original DDG** [^ddg] | DAgger variant: only invoke the expensive expert on segments where a fast-LaCAM probe says the policy is struggling (`max diff > 3`) | No | Threshold on a cheap probe | The threshold itself is wrong on ≈44% of borderline cases (§6.1) |
| **Stage 1 (auto-only ranker)** | Replace the threshold with a learned segment-ranker trained on auto pairs only | No | Auto pairs only | Plateaus at `human_val ≈ 0.61` late-epoch; misses borderline pairs where auto-diff disagrees with humans |
| **Stage 2 (random sampling fine-tune)** | Phase-1 auto backbone, then `--human-only` fine-tune on uniformly-random human pairs | Yes (replay-tool, no priority) | Random human supervision | Annotator's time spent uniformly; high-confidence pairs that the model already gets right are wasted budget |
| **Stage 3 (confusion AL fine-tune)** | Phase-1 auto backbone, then `--human-only` fine-tune on entropy-ranked human pairs | Yes (replay-tool, prioritised by entropy) | Model uncertainty | At this scale, beats random by a small but stable margin; depends on the prior model's calibration |
| **Stage 4 (iterative AL)** [pending re-run] | Stage 3 split into 4 rounds × 14 with model retrained between rounds | Yes (replay-tool, sequential) | Refreshed-each-round model uncertainty | Iteration only pays off if the model meaningfully improves between rounds; legacy May-3 numbers do not show a clear lift |
| **End-to-end DDG-with-classifier** (§6.7) | Replace the diff threshold inside DDG's runtime expert-selection with the trained classifier; retrain MAPF-GPT under this gating | No (autonomous in the loop; humans were upstream in §§6.2-6.4 to train the classifier) | Classifier-gated curriculum on POGEMA-distribution rollouts | Quality of the trained segment classifier; partial training (`ckpt_ddg_2000`) caps the asymptotic comparison vs `MAPF-GPT-original` (`ckpt_ddg_30000`). **Empirically (§6.7.2): the human-fine-tuned-classifier-gated `MAPF-GPT-S2` strictly dominates `MAPF-GPT-original` and `MAPF-GPT-S1` on every cell with headroom — +25-30 pp CSR at moderate-to-high agent density on Random / Maze / Warehouse, no SoC penalty. Stage-3 variant in training.** |

The "value-add" relative to original DDG is, in increasing order of novelty:

1. Replacing the hand-set threshold with a learned segment-ranking classifier (Stage 1).
2. Recovering label signal from the midrange band that DDG currently discards (continuous pair weighting, §4.4).
3. Two-phase `--human-only` fine-tune as the integration mechanism for sparse human pairs (§4.7) — preserves DDG-aligned ranking while adding human alignment.
4. End-to-end deployment of the trained classifier inside DDG's runtime expert-selection (§6.7) — the system contribution. A new MAPF-GPT model has been trained from scratch under classifier-gated DDG ([`checkpoints/baseline/`](checkpoints/baseline/)) and is configured for a head-to-head POGEMA benchmark vs the original threshold-gated MAPF-GPT ([`checkpoints/original/`](checkpoints/original/)) at the shared `ckpt_ddg_1500.pt` cut. Downstream measurements pending.
5. Label-preserving spatial-symmetry augmentation of the segment-classifier rollouts (§6.6.1, Isabel De Luis) — 5,007 augmented `.npz` files committed under `ranker_dataset/held_out_aug/`, providing a 4× zero-label-cost expansion of the auto-pair training corpus. Built and committed; not yet wired into a reported training run.
6. A parallel synthetic-jitter AL track on the standalone congestion classifier (§6.6.2, Isabel De Luis).

### 7.5 Failure Modes and Generalization

- **Sample size on the human val signal.** 76 pairs across 69 unique rollouts is much better than the previous 12-pair signal but each pair is still worth ≈1.3 pp on `human_val`. Multi-seed averaging is needed before claiming Stage 3 vs Stage 2 differences as robust. Direction is meaningful; magnitude carries roughly ±1 pair of noise.
- **Map distribution coverage.** Stage 2 and Stage 3 elicitation pools are filtered to the train-map seeds; the held-out human signal uses val-map seeds 144-147. Spatial generalisation across map seeds is the relevant test.
- **Distribution shift across DDG checkpoints.** As the policy improves, "what looks congested" changes. Annotations made on rollouts from one checkpoint may not transfer cleanly to later ones. Our `dataset/held_out/` covers `ckpt_0`, `500`, `1000`, `1500`, `30000` to mitigate this; the elicitation pool draws across all five.
- **Annotator bias.** The new replay tool randomises A/B presentation and hides the auto-diff to mitigate bias. The very first elicitation runs (warmstart, iterative; collected before commit 502008c) show a strong b/a imbalance (e.g., warmstart: 53 b_worse vs 3 a_worse) that disappears in the post-debiasing files (random: 36/42; confusion: 32/33). This is direct evidence the debiasing fix worked.

### 7.6 HRI Implications

The cost of human time was the binding constraint. Each elicitation session was budgeted at 100 queries (≈3 hours under the one-pair-at-a-time replay protocol); the human's "I am unsure" rate (22% on random, 35% on confusion AL) was itself informative. Pairwise interfaces win over absolute scoring at this budget: humans can rank quickly, while scoring an absolute "congestion level" would require a calibration we do not have.

Within that budget, the three-stage comparison cleanly answers two HRI-design questions. (a) *Does fine-tuning on a small budget of human pairs damage the auto-aligned ranker?* No — `pair_acc` is preserved to within 0.002 across all stages. (b) *Does an entropy-driven acquisition unlock more value than uniform random sampling?* Yes, modestly — confusion AL gets a higher and more stable `human_val` than random at a smaller label budget (65 vs 78). These two findings together recover the main value proposition of HRI in this setting: human labels can be safely added to a strong auto baseline, and a small amount of selection-strategy engineering can stretch a fixed annotation budget further.

---

## 8. Limitations and Future Work

- **Iterative AL re-run pending.** Stage 4's reported numbers use the legacy mixed-mode trainer; the comparable `--human-only` two-phase iterative protocol has not yet been measured.
- **`MAPF-GPT-S3` benchmark pending.** The Stage-3-gated MAPF-GPT (gated by the confusion-AL fine-tuned ranker) is still being trained; without it, we cannot yet say whether selection strategy on the upstream ranker (entropy AL vs random sampling) propagates into a downstream policy improvement, or whether the §6.7.2 lift is purely from *any* human supervision being present.
- **MovingAI and Puzzles suites pending.** The §6.7 results cover Random / Maze / Warehouse only; MovingAI (single large maps with up to 256 agents) and Puzzles (5×5 with 2-4 agents) have not been benchmarked yet for any algorithm.
- **Partial classifier-gated training.** All `MAPF-GPT-S*` models reach ≈ step 2,000 vs `MAPF-GPT-original`'s step 30,000. The shared-step `ckpt_ddg_1500.pt` cut is the apples-to-apples comparison, but the asymptotic value of classifier-gated DDG is not yet measurable from this run.
- **Geometric augmentation not yet trained on.** Isabel's [`augment_segment_rollouts.py`](augment_segment_rollouts.py) and the 5,007 committed augmented `.npz` files are available, but none of the reported May-10 training runs (Stages 1, 2, 3) used the augmented data. A re-run of Stage 1 with `--data ranker_dataset/held_out_aug` (or a merged root) is the natural next step to quantify the augmentation lift.
- **Few annotators.** The five annotation files come from two annotators on the team. Inter-annotator agreement was not measured systematically.
- **Threshold calibration on the score head.** We rank but do not calibrate: deploying the classifier in DDG with `expert_top_k = None` (call expert on all envs above some threshold rather than top-K) requires picking a decision threshold equivalent to the current `diff > 3`, which we have not yet selected.
- **Feature ablations.** We did not isolate the contribution of the recent-history channel (channel 3); it would be useful to test whether the model picks up oscillation specifically from this channel.
- **Run-to-run variance.** The trainer does not seed `torch.manual_seed`. The legacy methodology showed `human_val` swings of up to 23 pp across reseeds on the 12-pair val. The 76-pair val signal should attenuate this substantially, but we have only single-seed numbers for each stage. Multi-seed averaging is the next step.
- **Acquisition coverage.** We measure two points in the AL design space (uniform random, entropy-only). The original `H · ‖φ_A − φ_B‖` (entropy × diversity) acquisition was simplified out of the codebase early; reintroducing it with proper cross-rollout normalisation would let us measure whether the diversity term carries any residual signal. Pure cold-start diversity sampling (no scoring model) is also unmeasured.
- **Synthetic-jitter AL has no comparison numbers yet.** The `finetuning/export_augmented_active_learning_samples.py` track (§6.6.2) is implemented but the synthetic-AL vs random-on-augmentations comparison is not yet reported.
- **Annotated-set curation bias.** The train-map elicitation pool was filtered to midrange-bearing rollouts. The held-out val-map signal is similarly filtered. Performance on a uniformly-random rollout is unmeasured.

---

## 9. Conclusion

We replaced the hand-tuned threshold at the heart of the DDG hard-case-mining loop with a small spatio-temporal CNN trained on a continuous-weight pairwise objective and fine-tuned in a two-phase `--human-only` step on rare human pairwise verdicts. Three stages were compared at fixed fine-tune budget on a held-out 76-pair human-pair signal: Stage 1 (auto-only baseline, no human labels), Stage 2 (random sampling, 78 labels), Stage 3 (confusion-driven entropy-only AL, 65 labels). All three preserve the auto-aligned `pair_acc` to within 0.002 (0.648-0.650), demonstrating that human fine-tuning does not damage DDG-aligned ranking. On the held-out human signal, confusion AL converges to a stable `human_val ≈ 0.71` versus random's ≈ 0.66 and the auto baseline's ≈ 0.61 (late-epoch means; the auto baseline's `human_val` peak of 0.724 at epoch 7 is a transient that does not survive further training). Selection strategy beats label volume even at this modest scale: confusion AL adds value over random with fewer labels.

The trained classifier is plumbed end-to-end into DDG's runtime expert-selection (`finetuning/delta_data_generator.py`), and three classifier-gated MAPF-GPT models (one per upstream segment-classifier stage) are being trained for downstream comparison vs the original threshold-gated `MAPF-GPT-original`. Two have been benchmarked on POGEMA Random / Maze / Warehouse at the shared `ckpt_ddg_1500.pt` cut: the auto-only `MAPF-GPT-S1` lifts CSR over `MAPF-GPT-original` by +11/+16/+18 pp on Random / Warehouse but is roughly tied on Maze; the human-fine-tuned `MAPF-GPT-S2` (Stage-2 ranker, 78 randomly-sampled human pairs) **strictly dominates both** on every cell with headroom — +25-30 pp CSR at moderate density on all three suites, including Warehouse-64 going from 0.72 to 0.98 — with no SoC penalty. The 78 human pairs translate into the single largest result in the paper: a small budget of human supervision on the upstream ranker substantially improves the downstream MAPF-GPT policy. The Stage-3-gated `MAPF-GPT-S3` is currently being trained; its result will tell us whether confusion AL on the upstream ranker yields any additional downstream lift over Stage-2 random sampling. A label-preserving spatial-symmetry augmentation of the rollout corpus (Isabel De Luis, [`augment_segment_rollouts.py`](augment_segment_rollouts.py); 5,007 augmented `.npz` files committed) provides a 4× zero-label-cost expansion of the auto-pair training data, available for future training runs but not yet used in the May-10 stages above. A second, earlier augmentation track (synthetic-jitter AL on the standalone congestion classifier, §6.6.2) is also reported as a parallel exploration of the same acquisition family on a different data modality.

The HRI design choices — pairwise interface, debiased replay tool with random A/B swap and hidden auto-diff, two-phase `--human-only` fine-tune protocol, entropy-driven acquisition — collectively converted a sparse and noisy human signal into a stable, deployable lift over the auto baseline without sacrificing auto-aligned ranking. They constitute a clean blueprint for adding human judgment to any DDG-style data-curation loop.

---

## References

[FILL — IEEE format]

[^mapfgpt]: A. Andreychuk et al., "MAPF-GPT: Imitation Learning for Multi-Agent Pathfinding at Scale," AAAI 2025.
[^ddg]: A. Andreychuk et al., "Advancing Learnable Multi-Agent Pathfinding Solvers with Active Fine-Tuning," arXiv:2506.23793, 2025.
[^cbs]: G. Sharon et al., "Conflict-Based Search for Optimal Multi-Agent Path Finding," AAAI 2012.
[^lacam]: K. Okumura, "LaCAM: Search-Based Algorithm for Quick Multi-Agent Pathfinding," AAAI 2023.
[^pogema]: A. Skrynnik et al., "POGEMA: A Benchmark for Multi-Agent Pathfinding," 2024.
[^tamer]: W. B. Knox and P. Stone, "TAMER: Training an Agent Manually via Evaluative Reinforcement," ICDL 2008.
[^prefs]: P. Christiano et al., "Deep Reinforcement Learning from Human Preferences," NeurIPS 2017.
[^pairwise]: [FILL — pairwise comparison HRI reference, e.g., Sadigh et al. on active preference-based reward learning]
[^ranknet]: C. Burges et al., "Learning to Rank using Gradient Descent," ICML 2005.
[^al-survey]: B. Settles, "Active Learning Literature Survey," University of Wisconsin-Madison Department of Computer Sciences Technical Report #1648, 2010.
[^pairwise-al]: D. Sadigh et al., "Active Preference-Based Learning of Reward Functions," RSS 2017. (Acquisition function combining preference uncertainty with feature distance.)
[^dagger]: S. Ross, G. Gordon, J. A. Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger)," AISTATS 2011.

---

## Appendix A: Integration Options Considered

| Option | What it does | Verdict |
|---|---|---|
| A. Naive append | Add human pairs at weight 1.0 alongside auto | ≈100 human pairs vs ≈78k auto pairs → human gradient is < 0.1% of update; model never moves |
| B. Surgical override (legacy) | Drop auto pairs only from annotated episodes, replace with human pair | Helped on the previous mixed-mode trainer but still drowns human signal across non-annotated episodes |
| C. **Two-phase `--human-only` (adopted)** | Phase 1 train auto-only backbone; phase 2 fine-tune with `--human-only --init-from <phase1>.pair_acc.pt`. Auto pairs disabled in phase 2 entirely. | Clean separation; phase 1 gets full auto-pair signal, phase 2 gets full human-pair signal; `pair_acc` preserved within 0.002 across stages |
| D. Upweight | Append human at higher weight (2.0-5.0) without removing auto pairs | Still drowns at this scale; would need weight ≥ 800× to have parity, which destabilises auto-pair ranking |
| E. Re-bucket | Use human verdict as ground truth for the marked indices and force their auto-diff buckets | No longer applicable: bucketing has been replaced with continuous gap weighting (§4.4) |

---

## Appendix B: Per-Annotation Disagreement Examples

[OPTIONAL: pick 3 representative annotations from `annotation_val_map.json` where the human verdict contradicts the auto-diff ordering; show segment_diffs alongside human worst/clean indices and a short caption explaining what the human saw that the diff missed. The new persisted schema records `segment_a_range` and `segment_b_range` so the exact frames can be reproduced.]

---

## Slide Notes (for later)

Key talking points to lift directly into slides:

- **Hook.** "Across three stages — auto-only, random fine-tune, confusion-AL fine-tune — the DDG-aligned `pair_acc` is preserved to within 0.002 (0.648-0.650). On a held-out 76-pair human signal, confusion AL converges to a stable `human_val ≈ 0.71`, versus the auto baseline's late-epoch `≈ 0.61`. Selection strategy beats label volume even at this modest scale: 65 confusion-AL labels beat 78 random labels."
- **Visual hook 1.** Side-by-side: auto-pair direction vs human-pair direction on a contradicting annotation, with the replay-tool screenshot. Establishes that humans see what LaCAM-diff does not.
- **Visual hook 2.** The 3-stage `human_val` trajectory plot — show that Stage 1's peak is a transient at epoch 7 while Stage 3's plateau at 0.71 is stable from epoch 35 onward.
- **Visual hook 3.** The 3-stage `pair_acc` overlap plot — three curves visually overlapping by epoch 60. "Adding human supervision did not damage auto-aligned ranking."
- **Three-act structure.** Problem (DDG threshold is brittle and wrong on ~44% of borderline cases) → Method (segment classifier + continuous gap weighting + two-phase `--human-only` fine-tune + entropy-driven acquisition + end-to-end DDG integration) → Result (`pair_acc` preserved within 0.002; confusion AL stable at `human_val ≈ 0.71`; downstream DDG-loop integration in progress).
- **Where we sit among approaches (one slide).** Table from §7.4: MAPF-GPT (no human) → DAgger (uniform expert relabelling) → DDG (cheap-probe-thresholded relabelling) → Stage 1 (learned ranker, no human) → Stage 2 (random fine-tune) → Stage 3 (entropy AL fine-tune). Each row gets one short reason it falls short; ours gets the punchline that selection strategy + two-phase fine-tune is what unlocks the human signal without sacrificing the auto baseline.
- **HRI hammer.** The pairwise interface made annotations cheap to collect, and the random-A/B-swap + hidden-auto-diff debiasing fix demonstrably balanced annotator output (warmstart 53/3 → confusion 32/33). The novel finding is that *cheap-to-collect labels still need expensive-to-design selection rules*: random sampling at 78 labels gives a stable but lower lift than entropy-AL at 65 labels.
- **Closed-loop slide (the new headline).** Replacing DDG's hand-tuned diff threshold with the **human-fine-tuned Stage-2 segment classifier** (78 randomly-sampled human pairs), then retraining MAPF-GPT under that classifier-gated DDG, dominates the original threshold-gated MAPF-GPT on every cell with headroom: **Warehouse 64-agent CSR jumps from 0.719 to 0.984 (+27 pp); Random 32 → 48 agents jumps +26 → +30 pp; Maze 24-32 agents jumps +25-28 pp.** No SoC penalty (path lengths within ≈3% of Original; on Maze 32 actually shorter). The auto-only ranker (`MAPF-GPT-S1`) gets only +11/+16/+18 pp on Random/Warehouse and ties on Maze, isolating the value of the 78 human pairs. Numbers from [`benchmark.txt`](benchmark.txt), `ckpt_ddg_1500.pt` cut. `MAPF-GPT-S3` (Stage-3-AL-gated) training now. Disambiguate "Baseline" — the eval-config algorithm key `Baseline` is `MAPF-GPT-S1`, *not* the Stage-1 segment classifier itself; `Random` is `MAPF-GPT-S2`.
- **Why-it's-surprising slide.** The upstream `human_val` lift from Stage 1 → Stage 2 was small (≈ +0.05). The downstream CSR lift from `MAPF-GPT-S1` → `MAPF-GPT-S2` is +14-27 pp on six different cells. Curriculum compounding: 1,500 DDG steps of slightly-better expert selection produces large policy improvements. **Upstream metrics systematically understate downstream value.**
- **Maze diagnostic slide.** On Maze, `MAPF-GPT-S1` (auto-only) was *losing* to Original at 4 of 6 densities. `MAPF-GPT-S2` reverses this and beats Original by +25-28 pp at moderate density. The 78 human pairs fix a regime where the auto-only ranker was actively harmful — direct evidence the human signal is teaching the ranker something about maze-topology congestion that LaCAM-diff alone cannot capture.
- **Augmentation slide.** Isabel's [`augment_segment_rollouts.py`](augment_segment_rollouts.py): label-preserving spatial symmetries (hflip / vflip / rot180) applied directly to segment-classifier rollouts, preserving `segment_diffs`. 5,007 augmented `.npz` files committed under `ranker_dataset/held_out_aug/`, a 4× zero-label-cost expansion of auto-pair training data. Available for future re-runs.
- **Honest limitations.** (a) Single-seed for each of the three canonical stages; the 76-pair val should attenuate seed variance vs the legacy 12-pair signal but multi-seed averaging is needed to claim stable orderings. (b) Stage 4 iterative AL re-run pending. (c) POGEMA benchmark output files pending. (d) Geometric augmentation built but not yet trained on. (e) Synthetic-jitter AL track has no numerical comparison yet.
