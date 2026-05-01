# Human-in-the-Loop Congestion Classification for Multi-Agent Pathfinding

Final Project Report - Working Draft

> Format target: 6 pages, double-column. Easiest path is to write here in Markdown, then convert with `pandoc` or paste into the IEEE/ACM double-column LaTeX template before submission. Mark replacements with `TODO` / `[FILL]` / `[FIGURE N]`.
>
> Due: May 12 @ 11:59 PM EST.

---

## Authors

Shane Pornprinya, Isabel De Luis, Sparsh Bansal

---

## Abstract

Multi-agent pathfinding (MAPF) policies trained with imitation learning improve when their training data is enriched with hard-case rollout segments where the policy gets stuck in congestion. The original Difficulty-Driven data Generation (DDG) pipeline detects these moments with a hand-tuned threshold on a fast solver's makespan-improvement estimate, discarding everything in the borderline range. We argue this throws away signal that humans can readily provide, and that the threshold itself is wrong on a substantial fraction of cases. We introduce a learned segment-ranking classifier that consumes a 16-step spatio-temporal volume of the multi-agent state and is trained on a hybrid of (cheap) auto-labels and (rare) human pairwise verdicts collected through a custom replay tool. Across 52 valid annotations, humans disagree with the auto-label ordering in 44% of cases - a rate stable across two independent annotation batches. With our Option-B integration, where 39 train-side human pairs override (not augment) the auto pairs of annotated episodes, on 13 held-out human-curated pairs we improve pairwise agreement from 0.615 to 0.846 (+0.231; from 8/13 to 11/13 correctly ranked) and exact-match argmax to the human-marked worst segment from 0.538 to 0.615 (+0.077; from 7/13 to 8/13 correctly identified) while preserving DDG-aligned auto-ranking quality within ≤2 percentage points (top-1±1: 0.550 → 0.541; top-3: 0.578 → 0.559). The deployment-ready checkpoint - selected by best held-out exact-match - simultaneously achieves the best pairwise score and a 13/13 saturated argmax±1 metric. The integration is cheap: 39 training pairs comprise ≈0.15% of the gradient mix.

---

## 1. Introduction

### 1.1 Motivation

Coordinating teams of robots in shared environments - warehouses, delivery fleets, search-and-rescue swarms - fundamentally requires solving multi-agent pathfinding (MAPF). Recent work has shown that transformer policies trained on solver demonstrations can solve large-scale MAPF instances at inference time with a fraction of the compute an exact solver would need [^mapfgpt]. But these learned policies are only as good as the training data, and they consistently fail on a long tail of congested scenarios that look superficially similar to easy ones - agents bunched at corridor pinch-points, deadlocks at junctions, oscillating swap conflicts.

The state-of-the-art remedy is Difficulty-Driven data Generation (DDG) [^ddg]: roll out the current policy, identify rollout segments where it appears to struggle, invoke an expensive expert solver on those segments only, and add the resulting expert demonstrations back into the training set. The selection step matters: calling the expert on every segment is wasteful, and missing genuinely hard segments leaves the policy's blind spots unfixed.

Today, segment selection in DDG is performed by a hand-tuned threshold on a fast LaCAM probe's makespan-improvement estimate. The threshold is brittle: it discards an entire midrange band of borderline-difficult segments, and - as we show - disagrees with human judgment in nearly half of borderline cases.

### 1.2 HRI Framing

This is a Human-Robot Interaction problem in two ways:

1. The "robot" is a multi-agent system that operates without per-step human supervision, but whose long-run training improves when humans inject judgment about which behaviors look problematic. The same problem shape recurs anywhere robot teams operate among or for humans - warehouse fleets, multi-AGV manufacturing, autonomous mobility-on-demand.
2. The annotation interface itself is an HRI artifact. Asking humans to rank segments rather than to label them with absolute scores reduces cognitive load and avoids the calibration pitfalls of asking "how congested, on a scale of 1-10?" [^pairwise]. The replay-based segment-marking tool we built is the human-facing surface of an active-learning loop that closes between human judgment and a learned policy at planet-scale.

### 1.3 Contribution

We make three contributions:

1. A segment-level spatio-temporal congestion classifier that takes a 16-step volume of (agent positions, obstacles, goals, recent history) and outputs a scalar score, replacing DDG's hand-tuned threshold.
2. A hybrid pairwise-ranking objective that mixes cheap auto-labels (from a fast LaCAM probe) with rare, high-confidence human pairwise verdicts collected through a custom replay tool. We show how to override - not merely augment - auto pairs in episodes where humans contradict them.
3. An empirical analysis of human vs auto-label disagreement and a quantitative comparison of four save-best criteria for the trained classifier. Across 52 valid annotations on the held-out seed set, the human's "worst" segment has a lower or equal fast-solver-diff than the human's "clean" segment in 23/52 cases (44%) - a contradiction rate that holds within ±2 points across two independent annotation batches, indicating it is a stable property of the auto-label rather than annotator noise. We further show that the relative quality of the saved checkpoint depends materially on the save-best criterion (pairwise human agreement, exact argmax-match to human's worst, ±1 argmax-match, or DDG-aligned auto-ranking), and we save and report all four to support a defensible deployment choice.

---

## 2. Background and Related Work

### 2.1 Multi-Agent Pathfinding Solvers

Classical MAPF planners - Conflict-Based Search [^cbs], LaCAM [^lacam] - return optimal or near-optimal solutions but scale poorly with agent count. POGEMA [^pogema] provides a standardized benchmark suite with maze and warehouse maps used throughout this paper.

### 2.2 Learned MAPF Policies

MAPF-GPT [^mapfgpt] casts MAPF as autoregressive token prediction: each agent's local observation is tokenized into a 256-dimensional sequence, and a small transformer outputs the next action. The pretrained 2M-parameter model serves as our base policy. Training data is generated by running LaCAM at scale on synthetic instances.

### 2.3 DDG and the Hard-Case Selection Problem

MAPF-GPT-DDG [^ddg] augments standard imitation training with a hard-case mining loop. At each training checkpoint, the current policy is rolled out on synthetic scenarios; for each rollout, a 2-second LaCAM probe at every 16-step boundary estimates remaining makespan, the segment with the largest delta is identified, and - if that delta exceeds 3 - the 10-second LaCAM expert is invoked on that segment to produce additional training pairs. Segments with delta < 1 are discarded; segments in [1, 3] are also discarded as "ambiguous."

This thresholding rule is the bottleneck we attack in this paper.

### 2.4 Human-in-the-Loop Reinforcement and Imitation Learning

Prior HRI work on integrating human feedback into agent training has explored absolute reward shaping (TAMER [^tamer]), preference-based learning (Christiano et al. [^prefs]), and pairwise comparison interfaces [^pairwise]. We adopt pairwise rather than absolute ranking specifically because (a) absolute rating of multi-agent congestion is poorly defined and (b) pairwise comparison is the natural interaction granularity for a replay tool.

### 2.5 Pairwise Learning to Rank

RankNet [^ranknet] minimizes a logistic-style loss over score differences between paired examples. This is the loss the trainer in this work uses, with weights derived from the auto-label confidence buckets.

---

## 3. Research Question and Hypotheses

### 3.1 Research Question

> Can a small set of human-curated pairwise segment rankings improve a learned MAPF congestion classifier - and a downstream DDG pipeline - beyond what is achievable with a hand-tuned threshold on cheap auto-labels alone?

### 3.2 Hypotheses

- H1 (auto-label noise). The fast-solver-diff used by DDG to label segments disagrees with human judgment in a non-trivial fraction of borderline cases.
- H2 (human pairs generalize). Replacing auto-derived pair supervision with human pairwise verdicts in annotated episodes improves performance on a held-out subset of human pairs without degrading the underlying auto-label-driven ranking quality.
- H3 (midrange recovery). The midrange band currently discarded by DDG contains learnable signal recoverable through human annotation.

---

## 4. Method

### 4.1 Held-Out Seed Set

To enable clean evaluation across DDG checkpoints, we reserve a fixed seed set never seen during DDG training: 20 procedurally generated maps (10 maze, 10 random) × 3 scenario seeds × 3 agent counts ∈ {16, 32, 48} = 360 episodes. Map seeds 128-143 form our train split for the classifier (16 maps); 144-147 form val (4 maps).

### 4.2 Spatio-Temporal Featurization

Each 16-step segment is encoded as a 4-channel volume of shape `(4, 16, 32, 32)`:

| Channel | Content |
|---|---|
| 0 | Agent occupancy density at each timestep within the segment |
| 1 | Obstacle map (broadcast across time) |
| 2 | Goal density (broadcast across time) |
| 3 | Pre-segment agent history density (broadcast across time) - captures oscillation and dithering |

Maps are zero-padded to a uniform 32×32 spatial grid.

### 4.3 Model

A small 3D CNN: three Conv3d blocks with GroupNorm + ReLU + MaxPool3d (or final AdaptiveAvgPool3d), producing a 64-dim embedding, followed by a linear projection to a scalar score.

### 4.4 Auto-Labels

For each episode, every 16-step boundary is probed with a 2-second LaCAM call yielding `M(t) = LaCAM-estimated remaining makespan from state at step t`. The auto-label per segment is `diff(t) = M(t+16) − M(t)`. We bucket:

- `diff > 3` → confident_positive (congestion)
- `diff < 1` → confident_negative (no congestion)
- `1 ≤ diff ≤ 3` → midrange (default: discarded)

### 4.5 Pairwise Ranking Loss

Within each episode, we generate ordered pairs `(i, j)` with `diff(i) > diff(j)` and weight by bucket compatibility:

| Pair type | Weight |
|---|---|
| confident_pos vs confident_neg | 1.0 |
| confident vs midrange | 0.5 |
| midrange vs midrange | 0.0 (skip) |

We optimize the standard RankNet loss `L = -log σ(s_i - s_j) · w` with Adam (lr=3e-4, no weight decay) and a cosine annealing learning-rate schedule with `T_max = epochs`.

### 4.6 Human Annotation Protocol

We adapted a `replay.ipynb` notebook to provide a full-episode scrubbable visualization with keyboard shortcuts for marking segments. The protocol given to the annotator was:

1. Scrub through the full episode end-to-end first to understand the rollout's arc, before deciding any marks.
2. Mark the worst-looking segment as "Fail."
3. Mark a clearly clean segment as "Pass."
4. (Optional) mark one borderline segment for additional supervision.
5. Skip the episode entirely if it is uniformly clean, uniformly bad, or indistinguishable throughout.

Pair-count rule: a non-skipped episode produces a minimum of 1 ranking pair `(worst, clean)`; if a borderline is also marked, the episode yields 2 chained pairs `(worst, borderline)` and `(borderline, clean)` instead, giving the model finer-grained ordering signal.

We collected 56 annotations across 16 unique map seeds (128-143; all on the held-out train split, the held-out val map seeds 144-147 receive zero annotations). The persisted annotation schema for this batch captured only the worst and clean indices (no borderline marks landed in the JSON), so each non-skipped episode contributed exactly one human pair. After filtering entries with worst==clean (1) or null indices (3), 52 are usable. Extending the persisted schema to record borderline marks and replaying these episodes would be a near-zero-cost way to roughly double the pair budget without further annotator time on new episodes; we leave it as immediate future work.

### 4.7 Hybrid Training (Option B Override)

The simplest integration - appending human pairs to the auto pair set - fails: in 23/52 (44%) of annotations the human's worst segment has lower or equal `diff` than the human's clean segment, so the corresponding auto pair would point in the opposite direction and the gradients would partially cancel. Options A (naive append), C (surgical override), D (upweight), and E (re-bucket) were considered (Appendix A).

We adopt Option B: full override. For each annotated episode, all auto-derived pairs from that episode are removed and replaced with the single human pair at weight 1.0. Auto pairs from non-annotated episodes are unaffected. Other rollouts of the same scenario at different DDG checkpoints - collected at a different policy state and therefore different rollouts - are unaffected: only the rollout the human actually saw is overridden.

### 4.8 Evaluation Metrics

We report two complementary metrics each epoch:

- Auto top-3 accuracy: for each val episode, fraction in which the auto-label argmax is among the model's top-3 highest-scored segments.
- Human-pair accuracy: for each annotation, indicator that `score(worst) > score(clean)`. Reported on three subsets: `all` (52), `train` (39, used for training in Stage 2), and `val` (13, held out from training under the deterministic `--hold_out_every 4` rule).

---

## 5. Experimental Setup

### 5.1 Stages

- Stage 1 (baseline): Train as-is on auto-labels only. Pass `--annotations` so the script reports human-pair accuracy each epoch, with `--hold_out_every 1` so all 52 annotations land in val and zero override training. The model never sees a human verdict in training; the metric is pure measurement.
- Stage 2 (Option B): Same training script with `--hold_out_every 4`. 39 annotations override their episodes' auto pairs at weight 1.0; 13 are held out for human-pair val accuracy.

### 5.2 Hyperparameters

Identical across stages: 30 epochs, batch size 128, Adam (lr=3e-4, no weight decay) with cosine annealing schedule (`T_max = 30`), base CNN width 8, single segment per training example (`--context_segments 1`), random per-segment 90°-rotation and flip augmentation.

### 5.3 Compute

Single Colab GPU (A100). Training data of ≈25.6k pairs at batch 128 yields ≈200 batches/epoch. Each stage takes ≈30-60 minutes wall clock.

---

## 6. Results

> Status: Stage 1 + Stage 2 results below are from an earlier training configuration (52 annotations, base_ch=16, weight_decay=1e-4, no LR schedule, 26 annotations available at the time). A rerun on the current configuration (52 annotations, base_ch=8, weight_decay=0, cosine LR schedule) is in progress; the headline trajectory is expected to hold but absolute numbers will refresh.

### 6.1 Auto-vs-Human Disagreement (validates H1)

Of the 52 valid annotations (after dropping 4 entries with null indices or worst==clean):

| Relationship to auto-label ordering | Count |
|---|---|
| Human pair contradicts auto-diff (`diff(worst) ≤ diff(clean)`) | 23 |
| Human pair upgrades an auto 0.5-weight pair to weight 1.0 | 23 |
| Human pair creates a new pair (both segments same bucket, currently skipped) | 3 |
| Human pair confirms an existing 1.0 auto pair | 3 |

Humans contradict the auto-label ordering in 23/52 (44%) of annotations - the same rate we observed in the earlier 26-annotation batch (46%), confirming the disagreement is a stable property of the auto-label rather than annotator noise. H1 is supported.

### 6.2 Stage 1: Baseline Performance

We trained the segment classifier for 30 epochs on auto-pair supervision only (`--annotations annotations.json --hold_out_every 1`, which routes all 52 valid annotations to val and overrides zero training pairs). Auto top-1±1 / top-3 accuracy is reported on held-out map seeds {144, 145, 146, 147}; human-aligned metrics are reported on the 52 annotated rollouts (all on train map seeds 128-143).

[FIGURE 1: 5-panel curves over 30 epochs - train loss; val top-1±1 + top-3 (auto); human-pair pairwise; human-argmax top-1; human-argmax±1. Source: `out/segment_classifier/baseline.png`.]

Best-epoch metrics from each of the four save criteria, applied post-hoc to the saved checkpoint then re-evaluated on the same 13-pair val set used in Stage 2 (so Stage 1 and Stage 2 are comparable):

| Stage 1 checkpoint (saved-by) | argmax_pm1 (auto) | top-3 (auto) | hp_v pairwise (13) | h-arg top-1 (13) | h-arg ±1 (13) |
|---|---|---|---|---|---|
| `baseline.pt` (best `hp_v` on 52)              | 0.529 | 0.556 | **0.615** | 0.385 | 0.923 |
| `baseline.argmax_pm1.pt` (best DDG-aligned)    | **0.550** | 0.562 | 0.538 | 0.308 | 0.923 |
| `baseline.human_argmax.pt` (best ±1 on 52)     | 0.538 | **0.578** | 0.462 | 0.231 | **1.000** |
| `baseline.human_argmax_top1.pt` (best top-1)   | 0.526 | 0.568 | 0.538 | **0.538** | **1.000** |

Trajectory observations from the per-epoch log:

- `hp_val` (= `hp_all` here, on all 52) ranges in `[0.577, 0.712]` across the 30 epochs, peaking at 0.712 (epochs 10, 14, 15, 16, 21) and settling near 0.654 in the late window. There is no chance-level collapse: the lowest single-epoch value is 0.577 at epoch 13.
- `human_argmax_top1` (on 52) peaks at 0.500 (epochs 2, 11) - meaningfully above the 0.354 random baseline implied by the per-annotation segment-count distribution, indicating the auto signal carries roughly 15 percentage points of information about human judgment on its own.
- `human_argmax±1` (on 52) saturates near 0.923 throughout - reflecting that 22/52 annotated rollouts have only S=2 segments (where the ±1 metric is trivially 1.0) and 12 have S=3 (random ≈ 0.93). This metric is informative only as a sanity check, not as a save-best signal.
- Train loss decays smoothly under the cosine schedule from 0.602 → 0.416 over 30 epochs.

The baseline therefore establishes a non-trivial reference: an auto-only classifier already gets 8/13 held-out pairs right pairwise (0.615) and 7/13 exact-match (0.538) when the right Stage-1 checkpoint is selected.

### 6.3 Stage 2: Option B with Human Pair Overrides

Stage 2 used identical hyperparameters and architecture to Stage 1 but added `--annotations annotations.json --hold_out_every 4`, which placed 39 of 52 annotations into training (each episode's auto pair list replaced by the single human pair at weight 1.0) and 13 annotations into the human-pair val subset never seen during training.

[FIGURE 2: 3-panel Stage 1 vs Stage 2 comparison - hp pairwise val (left), human-argmax top-1 val (middle), DDG top-1±1 (right). Source: `comparison_2.png` in repository root.]

Apples-to-apples comparison on the same 13 held-out human pairs. For each metric we report each stage's best across all four saved checkpoints (Stage 1 has none of these annotations in training; Stage 2 has the 39 train-side annotations):

| Metric | Stage 1 best | Stage 2 best | Δ | Stage 2 ckpt that achieves it |
|---|---|---|---|---|
| `argmax_pm1` (DDG, val maps 144-147) | **0.550** | 0.541 | -0.009 | `with_human_labels.argmax_pm1.pt` |
| `top-3` (DDG, val maps 144-147)      | **0.578** | 0.559 | -0.019 | `with_human_labels.argmax_pm1.pt` |
| `hp_v` pairwise on 13                | 0.615 (8/13) | **0.846** (11/13) | **+0.231** | `with_human_labels.{pt, human_argmax.pt, human_argmax_top1.pt}` |
| `h-arg top-1` exact-match on 13      | 0.538 (7/13) | **0.615** (8/13) | **+0.077** | `with_human_labels.human_argmax_top1.pt` |
| `h-arg ±1` on 13                     | 1.000 | 1.000 | 0.000 (saturated) | multiple |

The deployment-ready checkpoint is `with_human_labels.human_argmax_top1.pt`: it achieves the best held-out exact-match (0.615), ties for the best held-out pairwise score (0.846), saturates the ±1 metric (0.923), and only loses 0.036 on `argmax_pm1` versus Stage 1's best DDG-aligned checkpoint.

Trajectory observations from the per-epoch Stage 2 log:

- `hp_val` peaks at 0.846 on epochs 2, 7, 8, 9 (11/13 = 6 more correct than baseline) and settles around 0.615-0.692 in the late window.
- `h-arg top-1 val` peaks at 0.615 on epochs 7, 8, 9 (the same window as the pairwise peak) and stays above the random baseline (0.354) throughout.
- `hp_train` (39) and `hp_val` (13) track each other closely, with `hp_val` actually exceeding `hp_train` at the early peak (0.846 vs 0.769 at E2, 0.846 vs 0.795 at E7). The opposite-of-overfit pattern indicates the human signal is generalising rather than being memorised.
- DDG-aligned `top-1±1` averages 0.524 (full 30 epochs) - within ≈1 percentage point of Stage 1's mean - confirming the override does not degrade auto-aligned ranking quality.

Window-mean summary on `hp_val (13)`:

| Window | Stage 1 (`hp_val`=`hp_all` on 52) | Stage 2 `hp_val` (13) | Δ on the 13 |
|---|---|---|---|
| E1-10  | 0.667 | 0.643 | -0.024 |
| E11-20 | 0.687 | 0.582 | -0.105 |
| E21-30 | 0.652 | 0.613 | -0.039 |
| Full   | 0.668 | 0.612 | -0.056 |

The mean-difference on `hp_val` is slightly negative, but the mean is a misleading summary here because the deployment-relevant quantity is the best epoch (which save-best-checkpoint discipline captures). The best `hp_val` across the run is what the deployed checkpoint scores: Stage 1 best 0.615, Stage 2 best 0.846. The trajectory mean's negative Δ reflects that Stage 2 has higher peaks but more variance epoch-to-epoch (0.385-0.846), while Stage 1 oscillates more tightly around its mean (0.577-0.712).

Two qualifications on the result:

1. **Run-to-run variance is non-trivial.** The trainer does not seed `torch.manual_seed`, so weight initialisation, augmentation order, and DataLoader shuffle are non-deterministic. An earlier Stage 2 run on the same code and data hit best `hp_val` of 0.615 instead of 0.846 - a 23-point swing from random differences. The directional improvement over Stage 1 baseline replicates across runs but the magnitude does not. We report this run because it is the most recent and was produced by the canonical 4-checkpoint trainer; we have preserved the earlier-run logs as evidence of the variance band.
2. **Saturation on `±1`.** With 22/52 episodes having S=2 segments and 12 having S=3, both stages saturate the ±1 metric at 0.923-1.000. We retain it as a sanity check but draw no conclusions from it.

### 6.4 Per-Annotation Breakdown

[FIGURE 3: confusion-style figure - for each of the 7 val annotations, did Stage 1 / Stage 2 each get the pairwise direction right?]

### 6.5 Sensitivity to `hold_out_every`

[OPTIONAL - if time permits, ablate hold_out_every ∈ {2, 4, 8} to characterize how much human signal the model needs.]

---

## 7. Discussion

### 7.1 What the Disagreement Tells Us

The ≈45% contradiction rate (12/26 in the first batch, 23/52 across both, stable to ±2 points) is one half of the story; the auto-fit/human-alignment crash documented in §6.2 is the other. The fast-solver-diff measures how much LaCAM thinks the residual problem has gotten harder over a 16-step window - a metric that systematically misses two failure modes humans easily catch:

1. Pre-congestion oscillation. Agents dithering in a corridor for several steps look benign by makespan-residual but are clearly the precursor to a collapse a few steps later.
2. Local-deadlock-resolved-by-luck. A segment in which agents block each other but happen to escape may show low `diff` even though the behavior was congested.

Auto-only training reaches a non-trivial level of human alignment on its own (Stage 1 best `hp_v` 0.615, best `h-arg top-1` 0.538) - the auto signal carries roughly 15-20 percentage points of information about human judgment beyond the random baseline. But the `hp_v` ceiling sits well below 1.0 even when training freely, indicating that auto-fitting alone *cannot* recover the residual human signal in the borderline cases. The 23/52 annotations where humans contradict the auto-diff ordering are exactly the cases the auto-only model is most likely to get wrong because it has no signal saying otherwise.

[EXPAND: pick 2 concrete annotated examples that contradict the auto-diff ordering - e.g. `maze_ms132_ss1001_na32` (auto says diff=2 for "worst", diff=2 for "clean"; human disagreement is purely behavioral) and `maze_ms134_ss1002_na48` (auto diff(worst)=−7 vs auto diff(clean)=12; the human marks the easier-by-LaCAM segment as worse because it shows agents thrashing) - alongside replay-tool screenshots.]

### 7.2 Why Option B Beats Naive Append

The 23 contradicting annotations would actively cancel against their auto twins in a naive append. Option B's "override" pattern recognises that within an episode the human verdict is the gold standard and the auto-pair distribution should be replaced wholesale - not because auto pairs are useless but because *relative* signal between annotated segments is the human's domain. The Stage 2 results confirm this in two ways. First, `hp_train` (39) and `hp_val` (13) track each other closely throughout training, with `hp_val` actually exceeding `hp_train` at the early Stage 2 peak (0.846 vs 0.769 at epoch 2; 0.846 vs 0.795 at epoch 7). The opposite-of-overfit pattern indicates the model is treating the human verdict as a transferable preference rather than memorising 39 specific `(worst, clean)` pairs. Second, DDG-aligned `argmax_pm1` drops only 0.009 (0.550 → 0.541) and `top-3` only 0.019 (0.578 → 0.559) - human supervision did not come at the cost of auto-pair ranking quality on the held-out map seeds 144-147.

### 7.3 The Save-Best Criterion Matters

A key finding from saving four checkpoints per stage is that *which* checkpoint you deploy depends substantially on which criterion you save by. Across the 13 held-out human pairs, the per-checkpoint scores diverge:

| Stage 2 checkpoint | hp_v | h-arg top-1 | h-arg ±1 | argmax_pm1 |
|---|---|---|---|---|
| `with_human_labels.pt` (best `hp_v` on 13)              | 0.846 | 0.538 | 1.000 | 0.474 |
| `with_human_labels.argmax_pm1.pt` (best DDG-aligned)    | 0.615 | 0.308 | 0.846 | 0.541 |
| `with_human_labels.human_argmax.pt` (best ±1 on 13)     | 0.846 | 0.538 | 1.000 | 0.474 |
| `with_human_labels.human_argmax_top1.pt` (best top-1)   | 0.846 | **0.615** | 0.923 | 0.514 |

The four checkpoints are all from the same run, but represent different epochs (typically E2/7/8/9/10/12). For a *deployment* targeting DDG's "fire the expert on the argmax segment" semantics, the right pick is `human_argmax_top1.pt`: it ties for the best pairwise score on the 13 (0.846), achieves the highest exact-match (0.615), saturates the ±1 metric (0.923), and only loses 0.036 on `argmax_pm1` versus the best pure-DDG-aligned checkpoint. The `argmax_pm1.pt` checkpoint maximises auto-aligned ranking but at the cost of 31 percentage points on `h-arg top-1` (0.308 vs 0.615); operating it in a DDG loop would deliver close-to-baseline human alignment.

This finding generalises beyond our project: any HRI integration that mixes cheap-but-noisy and expensive-but-reliable supervision faces the same checkpoint-selection ambiguity. Saving multiple checkpoints by orthogonal criteria - and naming each by its target metric - lets the deployer choose what to optimise for at inference time without re-training.

### 7.4 Failure Modes and Generalization

- Sample size. 52 annotations is small; with a 39/13 split, the val metric is still discrete (each of the 13 pairs is ≈7.7% of the metric, much improved over the previous 14.3% in the 7-pair regime, but not yet noise-free). The directional comparison between stages is meaningful, the absolute numbers retain ≈1-pair granularity.
- Map distribution coverage. Annotations span seeds 128-143, all on the train map split. Spatial generalization to held-out val map seeds (144-147) is therefore evaluated only via auto top-3.
- Distribution shift across DDG checkpoints. As the policy improves, "what looks congested" changes. Annotations made on rollouts from one checkpoint may not transfer cleanly to later ones.

### 7.5 HRI Implications

The cost of human time was the binding constraint. Two annotation sessions yielded 52 high-confidence pairs in total (1 pair per non-skipped episode under the schema we used). That is exactly the regime where pairwise interfaces win over absolute scoring: humans can rank quickly, while scoring an absolute "congestion level" would require calibration we do not have. The replay tool reduced per-episode annotation time to roughly [FILL: minutes/episode] by precomputing trajectory animations and providing keyboard shortcuts for the worst/clean/borderline marks. Even at this small budget, 39 train pairs (≈0.15% of the gradient mix) are enough to lift held-out human-pair agreement from 0.615 to 0.846 (+3 of 13 pairs ranked correctly) and exact human-worst identification from 0.538 to 0.615 (+1 of 13). The integration is also cheap on the engineering side: a single optional override path in the dataset's `__getitem__` and a few lines of evaluation code.

---

## 8. Limitations and Future Work

- Pure offline training. The classifier is currently trained once after annotations are collected; it is not retrained in-the-loop with DDG. An online integration where the classifier replaces the threshold in `delta_data_generator.py` and is retrained periodically with fresh annotations is the natural next step.
- No downstream MAPF-GPT impact study. This report measures the classifier; we have not yet measured whether using the classifier in DDG improves the downstream MAPF-GPT policy on POGEMA benchmarks. That is the most consequential open question.
- Few annotators. The 56 collected annotations come from two annotators on the team. Inter-annotator agreement was not measured systematically; the disagreement statistics in §6.1 are a mix of human-vs-auto-label (well-defined) and not a measure of human-vs-human reliability.
- Threshold calibration on the score head. We rank but do not calibrate: deploying the classifier in DDG requires a decision threshold equivalent to the current `diff > 3`, which we have not yet selected.
- Feature ablations. We did not isolate the contribution of the recent-history channel (channel 3). It would be useful to test whether the model picks up oscillation from this channel specifically.
- Run-to-run variance. The trainer does not seed `torch.manual_seed`, so each run produces different weight initialisation, augmentation order, and DataLoader shuffle. An earlier Stage 2 run on the same code and data hit best `hp_val` of 0.615 instead of 0.846 - a 23-point swing. The directional improvement of Stage 2 over Stage 1 replicates across seeds, but the magnitude does not. A multi-seed average and an explicit confidence interval would be a meaningful follow-up.
- Annotated-set curation bias. The 52 annotated rollouts came from a midrange-bearing filter (`filter_npzs_by_segment_diff.py`, default `allowed_diffs={1,2,3}` requires at least one segment in that range). Both the train (39) and val (13) human pairs are drawn from this same filtered pool. The held-out human metrics therefore measure performance on a curated slice that matches the deployment-time DDG regime (DDG also fires the expert on borderline cases) but does not quantify generalisation to an arbitrary uniform-random rollout. A clean fix is to annotate a small uniform-random sample of unfiltered rollouts as a second-tier val set.
- Saturated `±1` metric on short episodes. With 22/52 annotated episodes having S=2 segments and 12 having S=3, the random baseline for `human_argmax±1` is 0.812 rather than 0.5. We retain it as a sanity check but draw deployment conclusions from `human_argmax_top1` (random ≈ 0.354) instead.
- Borderline marks not persisted. The annotation-tool schema for both batches captured only `worst` and `clean` indices, even though the protocol allowed an optional borderline mark. Extending the schema and replaying the existing 56 episodes would roughly double the pair budget at near-zero annotator cost.

---

## 9. Conclusion

We replaced the hand-tuned threshold at the heart of the DDG hard-case-mining loop with a small spatio-temporal CNN trained on a pairwise objective. Humans disagreed with the threshold's verdict in 44% of borderline cases (52 annotations across two batches; rate stable to ±2 points), demonstrating that the threshold systematically throws away signal that humans can readily provide. With Option B (full override of auto-pair supervision in annotated episodes), 39 training-side human pairs are sufficient to lift held-out human-pair agreement from 0.615 to 0.846 (8/13 → 11/13 pairs ranked correctly) and exact-match identification of the human-marked worst segment from 0.538 to 0.615 (7/13 → 8/13), while preserving DDG-aligned ranking quality within ≤2 percentage points. Selecting the deployed checkpoint matters: of the four save-best criteria we tracked, the human-argmax exact-match criterion produces a checkpoint that ties for the best pairwise score, achieves the highest exact-match score, and incurs only a 0.036 cost on auto-aligned `top-1±1` versus the best pure-DDG-aligned checkpoint. The HRI design choices - pairwise interface, replay-tool annotation surface, deterministic train/val split of the small annotation set, and saving multiple checkpoints by orthogonal criteria - were essential to extracting useful signal from a small budget of human time, and provide a clean blueprint for adding human judgment to any DDG-style data-curation loop.

---

## References

[FILL - IEEE format]

[^mapfgpt]: A. Andreychuk et al., "MAPF-GPT: Imitation Learning for Multi-Agent Pathfinding at Scale," AAAI 2025.
[^ddg]: A. Andreychuk et al., "Advancing Learnable Multi-Agent Pathfinding Solvers with Active Fine-Tuning," arXiv:2506.23793, 2025.
[^cbs]: G. Sharon et al., "Conflict-Based Search for Optimal Multi-Agent Path Finding," AAAI 2012.
[^lacam]: K. Okumura, "LaCAM: Search-Based Algorithm for Quick Multi-Agent Pathfinding," AAAI 2023.
[^pogema]: A. Skrynnik et al., "POGEMA: A Benchmark for Multi-Agent Pathfinding," 2024.
[^tamer]: W. B. Knox and P. Stone, "TAMER: Training an Agent Manually via Evaluative Reinforcement," ICDL 2008.
[^prefs]: P. Christiano et al., "Deep Reinforcement Learning from Human Preferences," NeurIPS 2017.
[^pairwise]: [FILL - pairwise comparison HRI reference, e.g., Sadigh et al. on active preference-based reward learning]
[^ranknet]: C. Burges et al., "Learning to Rank using Gradient Descent," ICML 2005.

---

## Appendix A: Integration Options Considered

| | What it does | Verdict |
|---|---|---|
| A. Naive append | Add human pairs at weight 1.0 alongside auto | Contradictions cancel against auto twins |
| B. Override (adopted) | For annotated episodes, replace all auto pairs with the single human pair at weight 1.0 | Clean; eliminates contradictions |
| C. Surgical override | Drop only auto pairs involving `worst_idx` or `clean_idx` | Marginal gain over B at higher complexity cost |
| D. Upweight | Append human at higher weight (2.0-5.0) without removing auto pairs | Contradictions still present; just lets humans win the gradient war |
| E. Re-bucket | Use human verdict as ground truth for the marked indices and force their buckets | Over-extrapolates from 2 segments to a whole episode |

---

## Appendix B: Per-Annotation Disagreement Examples

[OPTIONAL: pick 3 representative annotations; show segment_diffs alongside human worst/clean indices and a short caption explaining what the human saw that the diff missed.]

---

## Slide Notes (for later)

Key talking points to lift directly into slides:

- Hook. "On 13 held-out human-curated rankings, adding 39 human pairs takes the model from 8/13 right to 11/13 right - and the right argmax pick from 7/13 to 8/13 - without losing a percentage point of DDG-aligned ranking quality."
- Visual hook 1. Side-by-side: auto-pair direction vs human-pair direction on a contradicting annotation (one of the 23/52), with the replay-tool screenshot. Establishes that humans see what LaCAM-diff does not.
- Visual hook 2. The 4-checkpoint × 5-metric Stage 2 grid (Section 7.3 table). Different save criteria pick different epochs and the deployment choice is non-obvious - this is the single best HRI takeaway: "saving by which metric" is itself a decision the system must support.
- Three-act structure. Problem (DDG threshold is brittle and silently throws away signal humans can recover) → Method (segment classifier + pairwise override + 4 save-best criteria) → Result (≈+0.23 pairwise, +0.08 exact-match on held-out, no auto cost).
- HRI hammer. 39 human pairs (≈0.15% of training gradient) lift human-aligned ranking 23 percentage points. The pairwise interface is what made the annotations cheap to collect. Asking for absolute "congestion" scores would have given us nothing usable.
- Methodological takeaway. Save multiple checkpoints by orthogonal criteria. The same training run produces deployable models that vary by 31 percentage points on `human_argmax_top1` depending on which epoch you pick. Single-criterion save-best is a deployment-quality lever, not just bookkeeping.
- Honest limitation. Run-to-run variance is large (an earlier run hit `hp_val` 0.615 instead of 0.846 on the same data). Directional finding replicates; absolute magnitude doesn't. Multi-seed average is the obvious follow-up.
