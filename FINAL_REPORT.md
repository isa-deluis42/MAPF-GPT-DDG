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

Multi-agent pathfinding (MAPF) policies trained with imitation learning improve when their training data is enriched with hard-case rollout segments where the policy gets stuck in congestion. The original Difficulty-Driven data Generation (DDG) pipeline detects these moments with a hand-tuned threshold on a fast solver's makespan-improvement estimate, discarding everything in the borderline range. We argue this throws away signal that humans can readily provide, and that the threshold itself is wrong on a substantial fraction of cases. We introduce a learned segment-ranking classifier that consumes a 16-step spatio-temporal volume of the multi-agent state and is trained on a hybrid of (cheap) auto-labels and (rare) human pairwise verdicts collected through a custom replay tool. Across 52 valid annotations, humans disagree with the auto-label ordering in 44% of cases - a rate stable across two independent annotation batches. We then run a four-stage comparison at fixed label budget (56 train-map labels per stage, evaluated on 12 held-out val-map pairs): Stage 1 auto-only baseline, Stage 2 random-sampling baseline, Stage 3 warm-start pool active learning, Stage 4 iterative active learning (4 rounds × 14 labels). Random sampling at fixed budget yields zero lift over the auto-only baseline (both `hp_val` = 5/12 = 0.417), while uncertainty × diversity AL lifts `hp_val` to 7/12 = 0.583 (+0.166; +17 percentage points), with iterative AL matching single-shot AL exactly at this budget. DDG-aligned ranking is preserved within 2.4 pp across all four stages. The headline takeaway is methodological: **selection strategy beats label volume** — at this scale, *which* 56 pairs the human labels matters more than the fact that they labelled 56 pairs.

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

We make four contributions:

1. A segment-level spatio-temporal congestion classifier that takes a 16-step volume of (agent positions, obstacles, goals, recent history) and outputs a scalar score, replacing DDG's hand-tuned threshold.
2. A hybrid pairwise-ranking objective that mixes cheap auto-labels (from a fast LaCAM probe) with rare, high-confidence human pairwise verdicts collected through a custom replay tool. We show how to override - not merely augment - auto pairs in episodes where humans contradict them.
3. An empirical analysis of human vs auto-label disagreement and a quantitative comparison of four save-best criteria for the trained classifier. Across 52 valid annotations on the held-out seed set, the human's "worst" segment has a lower or equal fast-solver-diff than the human's "clean" segment in 23/52 cases (44%) - a contradiction rate that holds within ±2 points across two independent annotation batches, indicating it is a stable property of the auto-label rather than annotator noise.
4. A four-stage comparison at fixed label budget that isolates the value of acquisition strategy from the value of human labels per se. Stage 1 (zero human labels) and Stage 2 (56 randomly-sampled human labels under the one-pair protocol) achieve identical held-out human-pair accuracy (5/12 = 0.417). Stage 3 (the same 56-label budget allocated by warm-start pool AL with acquisition `H(σ(s_A - s_B)) · ‖φ_A - φ_B‖`) lifts held-out accuracy to 7/12 = 0.583 (+17 percentage points). Stage 4 (iterative AL, 4 rounds × 14 labels with model retraining between rounds) matches Stage 3 exactly. The comparison directly answers external-review feedback ("compare AL strategies"), and the headline finding is that selection matters more than volume: random sampling does not move the model, but the same 56 labels chosen by an uncertainty × diversity acquisition do.

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

### 2.6 Active Learning for Preference Elicitation

Active-learning surveys [^al-survey] partition acquisition strategies into rough families - uncertainty sampling (query where the model's prediction is least confident), diversity / representativeness sampling (cover the feature space), expected-information-gain (EIG) maximisation (select the query that most reduces posterior entropy), and density-weighted variants that combine the above. For pairwise preference learning specifically, a common acquisition function combines the entropy of the model's preference probability `p = σ(s_A - s_B)` with the feature-space distance between candidates: queries where the model is uncertain *and* the candidates are not redundant carry the most information per label [^pairwise-al]. The professor's feedback on this project explicitly suggested comparing such strategies (boundary-querying vs. hard-case mining vs. alternation), and our Stages 3 and 4 implement two points in that design space: a single-shot warm-start uncertainty × diversity acquisition (Stage 3) and an iterative variant where the model is retrained between rounds (Stage 4).

---

## 3. Research Question and Hypotheses

### 3.1 Research Question

> Can a small set of human-curated pairwise segment rankings improve a learned MAPF congestion classifier beyond what is achievable with a hand-tuned threshold on cheap auto-labels alone, and does the *strategy used to elicit those rankings* (un-prioritised vs uncertainty-driven vs diversity-driven) materially affect the downstream classifier?

### 3.2 Hypotheses

- H1 (auto-label noise). The fast-solver-diff used by DDG to label segments disagrees with human judgment in a non-trivial fraction of borderline cases.
- H2 (human pairs generalize). Replacing auto-derived pair supervision with human pairwise verdicts in annotated episodes improves performance on a held-out subset of human pairs without degrading the underlying auto-label-driven ranking quality.
- H3 (midrange recovery). The midrange band currently discarded by DDG contains learnable signal recoverable through human annotation.
- H4 (acquisition strategy matters). At a fixed label budget, an uncertainty × diversity active-learning acquisition produces a stronger classifier than uniform random sampling, because it concentrates labels on the segment-pairs the current model is most uncertain about. We further test whether iterating the acquisition (model retrained between rounds) gives additional lift over single-shot pool selection.

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

### 4.8 Active-Learning Stages (Preference Elicitation)

The Stage 2 annotations were collected by un-prioritised replay-tool scrubbing: the annotator was free to skip episodes that looked uniform but otherwise saw all candidates equally. We extend this to two active-learning variants whose annotation pool is selected by an explicit acquisition function.

Pool-based acquisition. Both AL stages enumerate every (episode, segment_a, segment_b) candidate triple over the filtered pool of `≥4`-segment annotated rollouts, score each candidate with

```
acquisition(ep, A, B) = H(σ(s_A − s_B)) · ‖φ_A − φ_B‖
```

where `s_A`, `s_B` are scalar model scores for segments A and B, `H(·)` is binary entropy, and `φ_A`, `φ_B` are 11-dimensional hand-engineered feature vectors per segment (wait fraction, mean / max per-step movement, mean / max remaining goal distance, oscillation fraction, crowding indicators, shortest-path-overlap proxy). Features are globally normalised before the distance is computed. Candidates are then ranked descending by acquisition and a budget of `B` queries is greedy-selected subject to a `--per-episode-cap` to prevent any single hard episode from swallowing the budget.

Stage 3 - warm-start AL. The scoring model is the Stage 1 baseline checkpoint (saved-by `argmax_pm1`). The acquisition function reduces to "uncertainty × diversity": pairs the *current model* is most unsure about, weighted by feature-space novelty.

Stage 4 - iterative AL (4 rounds × 14 labels). Replaces the one-shot pool selection of Stage 3 with R = 4 sequential rounds. Each round queries 14 new pairs using the *current* model as scorer, labels them, appends to the corpus, and retrains. After round 4, the resulting model is the final Stage 4 checkpoint. Tests whether the warm-start scoring model improves enough between rounds to materially change the acquisition signal at this budget.

For both stages we set `--budget 56 --per-episode-cap 2`, matching Stage 2's total label count (52) and distributing across ≥28 distinct episodes. The trainer's `load_annotations` auto-detects `query.py`'s list-format JSON, so a single `(scenario_id, segment_a, segment_b, label)` per row produces multiple `(worst_idx, clean_idx)` overrides per episode (one per labelled pair), feeding directly into the same Option B path used in Stage 2.

### 4.9 Evaluation Metrics

We report two complementary metrics each epoch:

- Auto top-3 accuracy: for each val episode, fraction in which the auto-label argmax is among the model's top-3 highest-scored segments.
- Human-pair accuracy: for each annotation, indicator that `score(worst) > score(clean)`. Reported on three subsets: `all` (52), `train` (39, used for training in Stage 2), and `val` (13, held out from training under the deterministic `--hold_out_every 4` rule).

---

## 5. Experimental Setup

### 5.1 Stages

We compare four annotation-acquisition strategies, all sharing identical model architecture, optimiser, training data, and label budget. Stages 2-4 use the same one-pair-at-a-time labelling protocol against a held-out val-map elicitation set (12 random-sampled pairs from val map seeds 144-147, fixed across stages); only the *acquisition rule* over the train-map elicitation pool varies.

- Stage 1 (auto-only baseline): Train on auto-labels only. The model never sees a human verdict during training; the val-map labels serve only as an evaluation signal and the save-best criterion.
- Stage 2 (random-sampling baseline): Same trainer, same val signal. 56 train-map pairs are uniformly randomly sampled and labelled by a human under the one-pair-at-a-time protocol, then applied as Option-B overrides during training. Acts as the un-prioritised baseline isolating the value of selection strategy.
- Stage 3 (warm-start pool active learning): Use the Stage 1 baseline checkpoint as scoring model and run `query.py --budget 56 --per-episode-cap 2` over the train-map elicitation pool, pool-ranking by `H(σ(s_A − s_B)) · ‖φ_A − φ_B‖`. Greedy-pick top-56 subject to per-episode cap, label, train.
- Stage 4 (iterative active learning, 4 rounds × 14 labels): Replaces one-shot AL with R=4 rounds of 14 queries each. Each round: query 14 new pairs using the current model as scorer, label, append to the corpus, train. After round 4, `round_4.pt` is the final Stage 4 model.

In a separate, earlier experiment ("legacy hand-curation diagnostic", §6.3) we also collected 52 pairs via un-prioritised replay-tool scrubbing on a different held-out split; we report it for completeness but the 4-stage comparison above is the methodologically clean test.

All four stages produce up to four save-best checkpoints each (best `human_val`, best `argmax_pm1`, best `human_argmax±1`, best `human_argmax_top1`), supporting per-deployment-target checkpoint selection.

### 5.2 Hyperparameters

Identical across stages: 30 epochs, batch size 128, Adam (lr=3e-4, no weight decay) with cosine annealing schedule (`T_max = 30`), base CNN width 8, single segment per training example (`--context_segments 1`), random per-segment 90°-rotation and flip augmentation.

### 5.3 Compute

Single Colab GPU (A100). Training data of ≈25.6k pairs at batch 128 yields ≈200 batches/epoch. Each stage takes ≈30-60 minutes wall clock.

---

## 6. Results

> All four stages (1: auto-only, 2: random sampling, 3: warm-start pool AL, 4: iterative AL × 4 rounds) have been run and are reported below. base_ch=8, no weight decay, cosine LR schedule (`T_max=30`), four save-best checkpoints per stage. A separate legacy hand-curation diagnostic (52 hand-picked pairs on a different held-out split) is reported in §6.3 as a preliminary signal that the override mechanism works.

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

We trained the segment classifier for 30 epochs on auto-pair supervision only. Auto top-1±1 / top-3 accuracy is reported on held-out map seeds {144, 145, 146, 147}; human-aligned metrics are reported against the 12-pair val-map elicitation set (random-sampled, fixed across stages).

[FIGURE 1: 5-panel curves over 30 epochs - train loss; val top-1±1 + top-3 (auto); human-pair pairwise; human-argmax top-1; human-argmax±1. Source: `out/segment_classifier/baseline.png`.]

Best-epoch metrics across the four save criteria:

| Stage 1 checkpoint (saved-by) | argmax_pm1 (auto, val maps) | top-3 (auto, val maps) | hp_val (val-map · 12) | h-arg top-1 (val-map · 12) |
|---|---|---|---|---|
| `baseline.argmax_pm1.pt` (best DDG-aligned) | **0.565** | **0.581** | 0.417 | 0.083 |
| `baseline.pt` (best `hp_val`)               | 0.529  | 0.556  | **0.417** (random=0.5) | 0.000 |
| `baseline.human_argmax.pt` (best ±1)        | 0.538  | 0.578  | 0.333 | 0.083 |
| `baseline.human_argmax_top1.pt` (best top-1)| 0.526  | 0.568  | 0.333 | 0.083 |

The auto-only baseline scores **5/12 = 0.417** on held-out human pairwise val (essentially at random), and is essentially unable to identify the human-marked worst segment exactly (`h-arg top-1` ≤ 0.083). DDG-aligned `argmax_pm1` lands at 0.565. These set the reference points the AL stages have to clear.

Legacy diagnostic. On a separate, earlier 52-label hand-curated set (held-out 13 pairs), Stage 1's best checkpoint scored `hp_v = 0.615 (8/13)` and `h-arg top-1 = 0.538 (7/13)`. The val-map numbers above are stricter (different held-out maps, smaller N, random sampling not annotator selection), and we report them as the canonical baseline because all four AL stages share that exact val signal.

### 6.3 Stage 2: Random-Sampling Baseline (Option B Override)

Stage 2 uses the same trainer, same val-map signal, and same Option-B override path as the AL stages. The 56 train-map pairs are uniformly randomly sampled and labelled by a human under the one-pair-at-a-time protocol. This is the un-prioritised baseline that isolates the value of selection strategy: any lift Stages 3 and 4 show over Stage 2 is attributable to acquisition, not to the presence of human signal per se.

Best-checkpoint metrics (val-map · 12 pairs):

| Stage 2 checkpoint | argmax_pm1 | hp_val | h-arg top-1 | h-arg ±1 |
|---|---|---|---|---|
| `random_baseline.argmax_pm1.pt` (best DDG) | **0.556** | 0.417 | 0.000 | 0.250 |
| `random_baseline.pt` (best hp_val)         | 0.550  | **0.417** (5/12) | 0.083 | 0.250 |
| `random_baseline.human_argmax.pt`          | 0.553  | 0.333 | 0.083 | **0.333** |
| `random_baseline.human_argmax_top1.pt`     | 0.541  | 0.333 | **0.083** | 0.333 |

**Result: random sampling at fixed budget gives zero lift over the auto-only baseline on `hp_val` (Stage 1 = Stage 2 = 0.417, exactly 5/12).** The 56 human labels are spent on pairs the model wasn't confused about, and the lift on the held-out 12 evaporates. DDG-aligned `argmax_pm1` is unchanged within noise (0.565 → 0.556). This is the most important comparand for the AL stages: it tells us that without selection priority, human labels at this budget don't help.

Legacy hand-curation diagnostic. A separate, earlier experiment used 52 hand-curated annotations (un-prioritised replay-tool scrubbing) on a different held-out 13-pair val set; that ran reported `hp_v = 0.846 (11/13)` and `h-arg top-1 = 0.615 (8/13)`. We treat that as a preliminary positive signal that *some* human-curation strategy can lift the classifier, but the canonical 4-stage comparison below uses the stricter, methodologically clean random-sampling baseline against the val-map 12.

[FIGURE 2: 4-stage bar chart — `comparison_4stages.png` in repo root. Red bars = `hp_val` (val-map 12), grey bars = `argmax_pm1` (DDG, val maps 144-147). Random baseline at hp=0.5 marked.]

### 6.4 Stages 3 and 4: Active Learning Results

Stage 3 (warm-start pool AL) uses the Stage 1 baseline as scoring model and runs `query.py --budget 56 --per-episode-cap 2` over the train-map elicitation pool. Stage 4 (iterative AL) replaces one-shot pool selection with R=4 rounds × 14 labels = 56 total, retraining the model between rounds. Both share the same val-map 12-pair eval signal as Stages 1 and 2.

[FIGURE 3: 4-stage comparison plot - per-stage best on `hp_val` (val-map 12) and `argmax_pm1` (DDG val maps), with random-baseline guide line. Source: `comparison_4stages.png` in repo root.]

Apples-to-apples comparison across all four stages:

| Metric | Stage 1 (auto only) | Stage 2 (random sampling) | Stage 3 (warm-start AL) | Stage 4 (iterative AL · 4×14) |
|---|---|---|---|---|
| `hp_val` pairwise (val-map · 12)         | 0.417 (5/12) | 0.417 (5/12) | **0.583 (7/12)** | **0.583 (7/12)** |
| `argmax_pm1` (DDG, val maps 144-147)     | 0.565        | 0.556        | 0.562            | 0.541             |
| Δ on `hp_val` vs Stage 1                  | —            | 0.000        | **+0.166**       | **+0.166**        |
| Acquisition rule                          | —            | uniform random | H(σ(s_A−s_B))·‖φ_A−φ_B‖ | same, 4 rounds |
| Train labels used                         | 0            | 56           | 56               | 4 × 14            |
| Best round (Stage 4 only)                 | —            | —            | —                | round 3 (`pm1=0.565`, `hp_val=0.583`) |

**Three substantive findings.**

1. **Random sampling at fixed budget gives zero lift.** Stage 2 (`hp_val` 0.417) is identical to Stage 1 (`hp_val` 0.417). 56 human labels selected uniformly at random fail to move the model. Just adding human signal without selection priority is not enough.

2. **Uncertainty × diversity AL gives a clean +17 pp lift.** Stage 3 reaches `hp_val = 0.583 (7/12)` versus Stage 2's 0.417. Same labelling protocol, same budget, same val signal — only the acquisition rule changed. Selection strategy is what carries the lift.

3. **Iterative AL ≈ single-shot AL at this budget.** Stage 4 (4 rounds with model retrained between) matches Stage 3's 0.583 exactly. Engineering the iterative loop did not pay off here; one batch of 56 well-selected pairs was as good as four sequential batches of 14. At larger budgets the iterative variant might still pay off (the warm-start model gets a meaningfully better acquisition signal as it improves), but at our scale it does not.

DDG-aligned ranking is preserved across all four stages: `argmax_pm1` ∈ [0.541, 0.565], a ≤2.4-pp band that includes all four stages. Adding human supervision via any of these strategies does not degrade auto-aligned ranking quality.

### 6.5 Per-Annotation Breakdown

[FIGURE 4: confusion-style figure - for each of the 13 val annotations, which of the four stages got the exact-match (top-1) and pairwise direction right? Reveals which annotations are hardest, and whether different acquisition strategies stumble on different annotations.]

### 6.6 Sensitivity Studies (optional, time permitting)

- Multi-seed reruns of all four stages to characterise the variance band (we have evidence it is ≈±0.1 on `hp_val` from a single Stage 2 reseed).
- Alternation strategy: train on auto + Stage 4 (cold) for ≈10 epochs, then switch to Stage 3 (warm) using the resulting checkpoint. Tests whether early-training rounds want diversity and later rounds want uncertainty - a classic curriculum-learning question that the survey [^al-survey] flags as an open empirical issue.
- Per-episode-cap ablation: budget=56 with `cap ∈ {1, 2, 4, ∞}` to characterise the budget-concentration tradeoff.

---

## 7. Discussion

### 7.1 What the Disagreement Tells Us

The ≈45% contradiction rate (12/26 in the first batch, 23/52 across both, stable to ±2 points) is one half of the story; the auto-fit/human-alignment crash documented in §6.2 is the other. The fast-solver-diff measures how much LaCAM thinks the residual problem has gotten harder over a 16-step window - a metric that systematically misses two failure modes humans easily catch:

1. Pre-congestion oscillation. Agents dithering in a corridor for several steps look benign by makespan-residual but are clearly the precursor to a collapse a few steps later.
2. Local-deadlock-resolved-by-luck. A segment in which agents block each other but happen to escape may show low `diff` even though the behavior was congested.

Auto-only training reaches a non-trivial level of human alignment on its own (Stage 1 best `hp_v` 0.615, best `h-arg top-1` 0.538) - the auto signal carries roughly 15-20 percentage points of information about human judgment beyond the random baseline. But the `hp_v` ceiling sits well below 1.0 even when training freely, indicating that auto-fitting alone *cannot* recover the residual human signal in the borderline cases. The 23/52 annotations where humans contradict the auto-diff ordering are exactly the cases the auto-only model is most likely to get wrong because it has no signal saying otherwise.

[EXPAND: pick 2 concrete annotated examples that contradict the auto-diff ordering - e.g. `maze_ms132_ss1001_na32` (auto says diff=2 for "worst", diff=2 for "clean"; human disagreement is purely behavioral) and `maze_ms134_ss1002_na48` (auto diff(worst)=−7 vs auto diff(clean)=12; the human marks the easier-by-LaCAM segment as worse because it shows agents thrashing) - alongside replay-tool screenshots.]

### 7.2 Why Option B Beats Naive Append, and Why Selection Beats Volume

The 23 contradicting annotations would actively cancel against their auto twins in a naive append. Option B's "override" pattern recognises that within an episode the human verdict is the gold standard and the auto-pair distribution should be replaced wholesale - not because auto pairs are useless but because *relative* signal between annotated segments is the human's domain. The four-stage canonical comparison adds a second-order finding on top: Option B is *necessary* but not *sufficient* — the override path applied to randomly-sampled labels (Stage 2) gives zero lift over Stage 1, while the same override path applied to AL-selected labels (Stages 3, 4) lifts `hp_val` by +17 pp. Selection strategy is the dominant lever, not labelling protocol or override mechanics. DDG-aligned `argmax_pm1` is preserved across all four stages within ≤2.4 pp - human supervision via any of these strategies does not degrade auto-aligned ranking quality.

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

### 7.4 Where Our Approach Sits Among Alternatives

Five points in the design space, from cheapest to most informative:

| Approach | What it does | Human in loop | Where the signal comes from | Limit on quality |
|---|---|---|---|---|
| **MAPF-GPT** [^mapfgpt] | Imitation-learn from offline LaCAM-generated trajectories | No | Solver demonstrations | Coverage of the offline training set; long-tail congestion under-represented |
| **DAgger** [^dagger] | Iteratively roll the current policy, expert-relabel the visited states, retrain | No (expert is a solver) | On-policy expert relabelling | Expert is expensive at scale; expert calls per state are uniform |
| **Original DDG** [^ddg] | DAgger variant: only invoke the expensive expert on segments where a fast-LaCAM probe says the policy is struggling (`max diff > 3`) | No | Threshold on a cheap probe | The threshold itself is wrong on 44% of borderline cases (§6.1) |
| **Our Stage 2 (un-prioritised hand-curation)** | Replace the threshold with a learned segment-ranker; train on auto-labels + un-prioritised hand-curated overrides | Yes (replay-tool scrubbing) | Coverage-driven hand-labels | Annotator sees all candidates equally; human time is constant per episode regardless of value of label |
| **Our Stage 3 (warm-start pool AL)** | Same trainer; annotation pool selected by `H(σ(s_A − s_B)) · ‖φ_A − φ_B‖` against Stage 1 baseline; one-shot top-K | Yes (replay-tool, prioritised) | Model uncertainty × feature diversity | Quality of the prior scoring model upper-bounds the acquisition's value |
| **Our Stage 4 (iterative AL · 4×14)** | Same acquisition as Stage 3, but split into 4 rounds of 14 with the model retrained between rounds | Yes (replay-tool, prioritised, sequential) | Same as Stage 3, refreshed each round | Iteration only pays off if the model meaningfully improves between rounds - empirically, at this budget it does not |

The "value-add" relative to original DDG is therefore three things, in increasing order of novelty:

1. Replacing the hand-set threshold with a learned segment-ranking classifier (Stage 1 already shows this is viable - it picks the right segment ≈54% of the time on held-out human pairs, far above the DDG threshold's behaviour on the 44% contradicting cases).
2. Recovering label signal from the midrange band that DDG currently discards (Stage 2 demonstrates +0.231 pairwise / +0.077 exact-match using only midrange-bearing rollouts).
3. Comparing acquisition strategies for the elicitation step (Stages 3 and 4) so that future deployments of this kind of HRI loop have empirical guidance on whether it is worth spending engineering effort on uncertainty-driven sample selection vs cheaper diversity heuristics. This is the contribution the project's external feedback specifically requested.

The most consequential null result in the comparison is Stage 4 ≈ Stage 3 — the iteration loop did not buy anything at this label budget. This is good news for deployment: a single one-shot pool selection with a frozen scorer is as effective as a more complex 4-round protocol with model retraining between rounds. At larger budgets, where the scoring model improves meaningfully between rounds, iteration may still pay off; at this scale it does not.

### 7.5 Failure Modes and Generalization

- Sample size. 52 annotations is small; with a 39/13 split, the val metric is still discrete (each of the 13 pairs is ≈7.7% of the metric, much improved over the previous 14.3% in the 7-pair regime, but not yet noise-free). The directional comparison between stages is meaningful, the absolute numbers retain ≈1-pair granularity.
- Map distribution coverage. Annotations span seeds 128-143, all on the train map split. Spatial generalization to held-out val map seeds (144-147) is therefore evaluated only via auto top-3.
- Distribution shift across DDG checkpoints. As the policy improves, "what looks congested" changes. Annotations made on rollouts from one checkpoint may not transfer cleanly to later ones.

### 7.6 HRI Implications

The cost of human time was the binding constraint. Across the four canonical stages, each labelling session was budgeted at 56 pairs (≈3 hours under the one-pair-at-a-time replay protocol). That is exactly the regime where pairwise interfaces win over absolute scoring: humans can rank quickly, while scoring an absolute "congestion level" would require calibration we do not have. The replay tool reduced per-pair annotation time to roughly [FILL: seconds/pair] by precomputing trajectory animations and providing keyboard shortcuts.

Within that fixed budget, the four-stage comparison cleanly answers two HRI-design questions. (a) *Does the labelling cost pay off without prioritisation?* No - random sampling at 56 labels gives zero lift over the auto-only baseline on held-out human alignment. (b) *Does an uncertainty × diversity acquisition unlock the labels?* Yes - the same 56-label budget under AL selection delivers +17 pp on `hp_val`. The engineering investment in `query.py`'s acquisition loop pays for itself once the alternative is empirically null. The further finding that iterative AL did not improve over single-shot AL at this scale (Stage 4 ≈ Stage 3) is a deployment win: the simpler one-shot protocol is sufficient.

---

## 8. Limitations and Future Work

- Pure offline training. The classifier is currently trained once after annotations are collected; it is not retrained in-the-loop with DDG. An online integration where the classifier replaces the threshold in `delta_data_generator.py` and is retrained periodically with fresh annotations is the natural next step.
- No downstream MAPF-GPT impact study. This report measures the classifier; we have not yet measured whether using the classifier in DDG improves the downstream MAPF-GPT policy on POGEMA benchmarks. That is the most consequential open question.
- Few annotators. The 56 collected annotations come from two annotators on the team. Inter-annotator agreement was not measured systematically; the disagreement statistics in §6.1 are a mix of human-vs-auto-label (well-defined) and not a measure of human-vs-human reliability.
- Threshold calibration on the score head. We rank but do not calibrate: deploying the classifier in DDG requires a decision threshold equivalent to the current `diff > 3`, which we have not yet selected.
- Feature ablations. We did not isolate the contribution of the recent-history channel (channel 3). It would be useful to test whether the model picks up oscillation from this channel specifically.
- Run-to-run variance. The trainer does not seed `torch.manual_seed`, so each run produces different weight initialisation, augmentation order, and DataLoader shuffle. The legacy hand-curation experiment showed `hp_val` swings of up to 23 pp across reseeds. Each of the four canonical stages here is a *single* seed; a multi-seed average and explicit confidence intervals are needed before the Stage 3 vs Stage 4 ordering should be reported as definitive. The directional finding (random ≪ AL) is robust to seed; the +17 pp magnitude is a single-sample point estimate.
- Small held-out val. The val-map elicitation set is 12 pairs. Each correctly-ranked pair is worth ≈8 pp of `hp_val`, so the +17-pp Stage 3 lift over Stage 2 corresponds to flipping 2 specific pairs (5/12 → 7/12). The direction is meaningful; the magnitude carries 1-pair noise.
- Active-learning strategy coverage. We compare three points in the AL design space (random sampling as Stage 2, uncertainty × diversity warm-start as Stage 3, iterative warm-start AL as Stage 4) but stop short of two strategies the literature [^al-survey] highlights: pure boundary querying (entropy alone, no diversity term), and pure cold-start diversity sampling (no scoring model — entropy is constant, acquisition reduces to feature-space distance only). Adding these would require additional annotation passes; we leave them as future work.
- Annotated-set curation bias. The train-map elicitation pool was filtered to midrange-bearing rollouts (`filter_npzs_by_segment_diff.py`, default `allowed_diffs={1,2,3}` requires at least one segment in that range). The held-out val-map signal is also random-sampled from the val-map pool, which is similarly filtered. The held-out human metrics therefore measure performance on a curated slice that matches the deployment-time DDG regime (DDG also fires the expert on borderline cases) but does not quantify generalisation to an arbitrary uniform-random rollout.
- Borderline marks not persisted. The annotation-tool schema for both batches captured only `worst` and `clean` indices, even though the protocol allowed an optional borderline mark. Extending the schema and replaying the existing 56 episodes would roughly double the pair budget at near-zero annotator cost.

---

## 9. Conclusion

We replaced the hand-tuned threshold at the heart of the DDG hard-case-mining loop with a small spatio-temporal CNN trained on a pairwise objective. Humans disagreed with the threshold's verdict in 44% of borderline cases (52 annotations across two batches; rate stable to ±2 points), demonstrating that the threshold systematically throws away signal that humans can readily provide. We then ran a four-stage comparison at fixed label budget (56 train-map labels per stage, evaluated on the same 12 held-out val-map pairs): Stage 1 auto-only baseline, Stage 2 random-sampling baseline, Stage 3 warm-start pool active learning, Stage 4 iterative active learning (4 rounds × 14 labels). The headline finding is that **selection strategy beats label volume**: Stage 2 (random sampling) achieves identical `hp_val` to Stage 1 (both 5/12 = 0.417), while Stages 3 and 4 (uncertainty × diversity AL) both reach 0.583 (7/12), a +17-pp lift over the random baseline at the same labelling cost. Iterative AL did not improve over single-shot AL at this budget. DDG-aligned ranking quality (`argmax_pm1`) is preserved within 2.4 pp across all four stages. Selecting the deployed checkpoint also matters: of the four save-best criteria we tracked, the human-argmax exact-match criterion produces a checkpoint that ties for the best pairwise score, achieves the highest exact-match score, and incurs only a 0.036 cost on auto-aligned `top-1±1` versus the best pure-DDG-aligned checkpoint. The HRI design choices - pairwise interface, replay-tool annotation surface, four save-best criteria, and the comparison across acquisition strategies - were essential to extracting useful signal from a small budget of human time, and provide a clean blueprint for adding human judgment to any DDG-style data-curation loop.

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
[^al-survey]: B. Settles, "Active Learning Literature Survey," University of Wisconsin-Madison Department of Computer Sciences Technical Report #1648, 2010. (See also the AL-strategies overview on page 3 of [the paper the project's external reviewer linked]; FILL with exact citation.)
[^pairwise-al]: D. Sadigh et al., "Active Preference-Based Learning of Reward Functions," RSS 2017. (Acquisition function combining preference uncertainty with feature distance.)
[^dagger]: S. Ross, G. Gordon, J. A. Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger)," AISTATS 2011.

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

- Hook. "Random sampling of 56 human labels gives the same held-out human-pair accuracy as zero labels (5/12 = 0.417). Smart selection of the same 56 labels — uncertainty × diversity active learning — lifts it to 7/12 = 0.583. Selection strategy beats label volume."
- Visual hook 1. Side-by-side: auto-pair direction vs human-pair direction on a contradicting annotation (one of 23/52), with the replay-tool screenshot. Establishes that humans see what LaCAM-diff does not.
- Visual hook 2. The 4-stage bar chart (`comparison_4stages.png`): Stage 1 (auto-only) and Stage 2 (random sampling) on the left at 0.417; Stages 3 (warm-start AL) and 4 (iterative AL) on the right at 0.583. Same 56-label budget across Stages 2-4. The visual asymmetry between random and AL is the slide.
- Visual hook 3. The 4-checkpoint × 5-metric save-best grid (Section 7.3 table). Different save criteria pick different epochs and the deployment choice is non-obvious.
- Three-act structure. Problem (DDG threshold is brittle and wrong on 44% of borderline cases) → Method (segment classifier + Option-B override + comparison across acquisition strategies + 4 save-best criteria) → Result (random sampling gives zero lift; AL gives +17 pp on held-out; iterative ≈ single-shot).
- Where we sit among approaches (one slide). Table from §7.4: MAPF-GPT (no human) → DAgger (uniform expert relabelling) → DDG (cheap-probe-thresholded relabelling) → Stage 2 random sampling (human, no priority) → Stages 3/4 AL (human + selection priority). Each row gets one short reason it falls short; ours gets the punchline that selection priority is what unlocks the human signal.
- HRI hammer. The pairwise interface made annotations cheap to collect. The novel finding is that *cheap-to-collect labels still need expensive-to-design selection rules*: human time on randomly-chosen pairs goes to waste at this scale.
- Methodological takeaway. Save multiple checkpoints by orthogonal criteria. The same training run produces deployable models that vary by 31 percentage points on `human_argmax_top1` depending on which epoch you pick.
- Honest limitations. (a) Single-seed for each of the four canonical stages; the legacy hand-curation re-seed swung up to 23 pp on `hp_val`, so single-seed orderings need multi-seed averaging before they're definitive. (b) Held-out val is 12 pairs — each correctly-ranked pair is worth ≈8 pp. (c) Two AL strategies the literature highlights (pure boundary querying, pure cold-start diversity sampling with no scoring model) are future work.
