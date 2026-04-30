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

Multi-agent pathfinding (MAPF) policies trained with imitation learning improve when their training data is enriched with hard-case rollout segments where the policy gets stuck in congestion. The original Difficulty-Driven data Generation (DDG) pipeline detects these moments with a hand-tuned threshold on a fast solver's makespan-improvement estimate, discarding everything in the borderline range. We argue this throws away signal that humans can readily provide, and that the threshold itself is wrong on a substantial fraction of cases. We introduce a learned segment-ranking classifier that consumes a 16-step spatio-temporal volume of the multi-agent state and is trained on a hybrid of (cheap) auto-labels and (rare) human pairwise verdicts collected through a custom replay tool. Across 26 valid annotations, humans disagree with the auto-label ordering in 46% of cases. Pure auto-label training exhibits a striking pathology: the model's agreement with human verdicts peaks early at 80.8% (epoch 3) then degrades to chance (50%) by epoch 26, even as training loss decreases monotonically. With our Option-B integration, where human pairs override (not augment) the auto pairs of annotated episodes, the 26-pair human-agreement mean rises from 0.635 to 0.737 across 30 epochs (+0.102), the peak 0.808 is hit on 8 epochs versus 1 in the baseline, the deployed (best-checkpoint) model reaches 0.857 on a 7-pair held-out val set, and auto-aligned top-3 accuracy is preserved unchanged at 0.608. The headline result is qualitative: human supervision converts a one-epoch peak into a multi-epoch plateau, holding the model in the human-aligned ranking regime where pure auto-label training cannot stay.

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
3. An empirical analysis of human-auto-label disagreement and a striking auto-fit pathology. Across 26 valid annotations on the held-out seed set, the human's "worst" segment has a lower or equal fast-solver-diff than the human's "clean" segment in 12/26 cases. Furthermore, training a model purely on auto-labels causes its agreement with human verdicts to peak briefly (80.8% at epoch 3) and then collapse to chance (50%) by epoch 26 - even though training loss is monotonically decreasing the entire time. The auto-label is therefore not just a noisy approximation of human judgment but partially anti-aligned with it.

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

We optimize the standard RankNet loss `L = -log σ(s_i - s_j) · w` with Adam (lr=3e-4).

### 4.6 Human Annotation Protocol

We adapted a `replay.ipynb` notebook to provide a full-episode scrubbable visualization with keyboard shortcuts for marking segments. The protocol given to the annotator was:

1. Scrub through the full episode end-to-end first to understand the rollout's arc, before deciding any marks.
2. Mark the worst-looking segment as "Fail."
3. Mark a clearly clean segment as "Pass."
4. (Optional) mark one borderline segment for additional supervision.
5. Skip the episode entirely if it is uniformly clean, uniformly bad, or indistinguishable throughout.

Pair-count rule: a non-skipped episode produces a minimum of 1 ranking pair `(worst, clean)`; if a borderline is also marked, the episode yields 2 chained pairs `(worst, borderline)` and `(borderline, clean)` instead, giving the model finer-grained ordering signal.

We collected 28 annotations across 9 unique map seeds (all on the train split, none on val by chance). The persisted annotation schema for this batch captured only the worst and clean indices (no borderline marks landed in the JSON), so each non-skipped episode contributed exactly one human pair. After filtering entries with worst==clean or null indices, 26 are usable. Extending the persisted schema to record borderline marks and replaying these episodes would be a near-zero-cost way to roughly double the pair budget without further annotator time on new episodes; we leave it as immediate future work.

### 4.7 Hybrid Training (Option B Override)

The simplest integration - appending human pairs to the auto pair set - fails: in 12/26 annotations the human's worst segment has lower or equal `diff` than the human's clean segment, so the corresponding auto pair would point in the opposite direction and the gradients would partially cancel. Options A (naive append), C (surgical override), D (upweight), and E (re-bucket) were considered (Appendix A).

We adopt Option B: full override. For each annotated episode, all auto-derived pairs from that episode are removed and replaced with the single human pair at weight 1.0. Auto pairs from non-annotated episodes are unaffected. Other rollouts of the same scenario at different DDG checkpoints - collected at a different policy state and therefore different rollouts - are unaffected: only the rollout the human actually saw is overridden.

### 4.8 Evaluation Metrics

We report two complementary metrics each epoch:

- Auto top-3 accuracy: for each val episode, fraction in which the auto-label argmax is among the model's top-3 highest-scored segments.
- Human-pair accuracy: for each annotation, indicator that `score(worst) > score(clean)`. Reported on three subsets: `all` (26), `train` (19, used for training in Stage 2), and `val` (7, held out from training).

---

## 5. Experimental Setup

### 5.1 Stages

- Stage 1 (baseline): Train as-is on auto-labels only. Pass `--annotations` so the script reports human-pair accuracy each epoch, with `--hold_out_every 1` so all 26 annotations land in val and zero override training. The model never sees a human verdict in training; the metric is pure measurement.
- Stage 2 (Option B): Same training script with `--hold_out_every 4`. 19 annotations override their episodes' auto pairs at weight 1.0; 7 are held out for human-pair val accuracy.

### 5.2 Hyperparameters

Identical across stages: 30 epochs, batch size 128, Adam lr=3e-4, weight decay 1e-4, base CNN width 16, single segment per training example (`--context_segments 1`), random per-segment 90°-rotation and flip augmentation.

### 5.3 Compute

Single Colab GPU (A100). Training data of ≈26k pairs at batch 128 yields ≈205 batches/epoch. Each stage takes ≈30-60 minutes wall clock.

---

## 6. Results

> Status: Stage 1 and Stage 2 runs in progress. Numbers below are placeholders; refresh after both runs complete.

### 6.1 Auto-vs-Human Disagreement (validates H1)

Of the 26 valid annotations:

| Relationship to auto-label ordering | Count |
|---|---|
| Human pair contradicts auto-diff (`diff(worst) ≤ diff(clean)`) | 12 |
| Human pair upgrades an auto 0.5-weight pair to weight 1.0 | 12 |
| Human pair creates a new pair (both midrange, currently skipped) | 1 |
| Human pair confirms an existing 1.0 auto pair | 2 (1 dropped due to worst==clean) |

Humans contradict the auto-label ordering in 12/26 (46%) of annotations. H1 is supported.

### 6.2 Stage 1: Baseline Performance

We trained the segment classifier for 30 epochs on auto-pair supervision only (`--annotations annotations.json --hold_out_every 1`, which routes all 26 valid annotations to val and overrides zero training pairs). Auto top-3 accuracy is reported on the held-out map seeds {144, 145, 146, 147}; human-pair accuracy is reported on the 26 annotated rollouts (all on train map seeds 128-136).

[FIGURE 1: 3-panel curves over 30 epochs - train loss; val top-3 + top-1±1; human-pair accuracy. Source: `out/segment_classifier/baseline.png`.]

Best-epoch metrics (Stage 1):

| Metric | Value | Epoch |
|---|---|---|
| Best human-pair accuracy (all 26) | 0.808 | 3 |
| Best val top-3 accuracy | 0.608 | 17 |
| Train loss at end of training | 0.278 | 30 |

Key observation - the auto-fit/human-alignment crash. Human-pair accuracy traces a sharp inverted-U over training:

| Epoch | 1 | 2 | 3 | 4 | 8 | 16 | 26 | 30 |
|---|---|---|---|---|---|---|---|---|
| Human-pair acc | 0.577 | 0.769 | 0.808 | 0.692 | 0.577 | 0.538 | 0.500 | 0.500 |

Meanwhile, train loss drops monotonically from 0.580 → 0.278 over the same 30 epochs, and val top-3 hovers in `[0.556, 0.608]` with no clear trend. The model's auto-pair fitting therefore actively pushes it away from human judgment - by epoch 26 it has no better than chance agreement with the 26 annotators' verdicts on the same val rollouts, despite a halving of training loss.

This is meaningful for two reasons:

1. It validates H1 directly. The auto-label is not merely noisy - perfectly fitting it is anti-correlated with human judgment on the borderline cases that actually matter. The 12/26 contradicting annotations push the model into a regime where the auto pair-ordering points the wrong way.
2. It establishes a non-trivial baseline ceiling. Stage 2's job isn't to beat 0.808 in absolute terms; the right comparison is whether human supervision can sustain that level past epoch 3 instead of letting it collapse. A flat 0.80 across all 30 epochs in Stage 2 would be a clear win even without exceeding the absolute peak.

### 6.3 Stage 2: Option B with Human Pair Overrides

Stage 2 used identical hyperparameters and architecture to Stage 1 but added `--annotations annotations.json --hold_out_every 4`, which placed 19 of 26 annotations into training (each episode's auto pair list replaced by the single human pair at weight 1.0) and 7 annotations into the human-pair val subset never seen during training.

[FIGURE 2: 3-panel comparison - human-pair val/all (left), val top-1±1 (middle), val top-3 (right). Stage 1 vs Stage 2 overlaid. Source: `comparison_1.png` in repository root.]

Best-epoch metrics, Stage 1 vs Stage 2 (full 30 epochs):

| Metric | Stage 1 | Stage 2 | Δ |
|---|---|---|---|
| Best human-pair acc (all 26) | 0.808 (epoch 3 only) | 0.808 (8 epochs: 7, 12, 16, 17, 19, 20, 27, 28) | sustained, not just peaked |
| Best human-pair acc (val, 7) | [FILL via post-hoc eval of Stage-1 ckpt] | 0.857 (epochs 1, 3) | [FILL] |
| Best human-pair acc (train, 19) | n/a | 0.895 (epochs 27, 29) | - |
| Best val top-3 accuracy | 0.608 (epoch 17) | 0.608 (epoch 11) | 0.000 |
| Best val top-1±1 accuracy | [FILL via post-hoc eval] | 0.592 (epoch 10) | [FILL] |

Trajectory on `human-pair acc all` (the same 26-pair set scored under each model at the indicated epoch):

| Epoch | Stage 1 hp_all | Stage 2 hp_all | Stage 2 hp_val (7) | Stage 2 hp_train (19) |
|---|---|---|---|---|
|  1 | 0.577 | 0.731 | 0.857 | 0.684 |
|  3 | 0.808 | 0.769 | 0.857 | 0.737 |
|  7 | 0.692 | 0.808 | 0.714 | 0.842 |
| 12 | 0.577 | 0.808 | 0.714 | 0.842 |
| 17 | 0.692 | 0.808 | 0.714 | 0.842 |
| 20 | 0.654 | 0.808 | 0.714 | 0.842 |
| 24 | 0.577 | 0.538 | 0.286 | 0.632 |
| 27 | 0.654 | 0.808 | 0.571 | 0.895 |
| 28 | 0.615 | 0.808 | 0.714 | 0.842 |
| 30 | 0.500 | 0.692 | 0.429 | 0.789 |

Window-mean summary (full 30 epochs broken into thirds):

| Subset | Window | Stage 1 | Stage 2 | Δ |
|---|---|---|---|---|
| `hp_all` (26) | E1-10  | 0.677 | 0.723 | +0.046 |
| `hp_all` (26) | E11-20 | 0.635 | 0.781 | +0.146 |
| `hp_all` (26) | E21-30 | 0.592 | 0.708 | +0.116 |
| `hp_all` (26) | full mean | 0.635 | 0.737 | +0.102 |
| `hp_val` (7) | E1-10  | n/a | 0.657 | - |
| `hp_val` (7) | E11-20 | n/a | 0.671 | - |
| `hp_val` (7) | E21-30 | n/a | 0.514 | - |
| `hp_val` (7) | full mean | n/a | 0.614 | - |
| `hp_train` (19) | full mean | n/a | 0.782 | - |

Interpretation. Stage 2 sustains human-pair agreement above Stage 1 in every training window, with the gap *widening* in the middle of training (Δ +0.146 at epochs 11-20) and remaining substantial in the final third (Δ +0.116 at epochs 21-30). On the worst single epoch, Stage 2 hits 0.538 - still 4 points above Stage 1's worst (0.500, twice). On the saved checkpoint (epoch 1, deployed model), Stage 2 reaches 0.857 on the 7-pair val set against Stage 1's 0.808 best on any subset. This corresponds to scenario 1 of the three pre-registered in our planning: human supervision converts a transient peak into a sustained regime.

Two qualifications on the result:

1. Late-window val drift. `hp_val` (7) drops noticeably in epochs 21-30 (mean 0.514 vs 0.671 in the middle window). The corresponding `hp_train` window mean is 0.779 (with brief surges to 0.895 at epochs 27, 29), a mild signal that the model is starting to over-fit the 19 training pairs. Critically, `hp_all` does *not* drift the same way (0.708 in the late window) because most of the 26-set is still being trained to and continues to track. The save-best-by-`hp_val` logic captures the high-water mark at epoch 1, so the *deployed* checkpoint is from before any late drift.
2. Small-N val noise. With 7 val pairs each is worth 14.3% of the metric, so `hp_val` jumps in `[0.286, 0.857]` correspond to flipping 1-4 specific pairs. The trajectory shape (oscillation around a stable mean for 20 epochs, then a downshift in the last 10) is meaningful; specific epoch-to-epoch deltas are not.

### 6.4 Per-Annotation Breakdown

[FIGURE 3: confusion-style figure - for each of the 7 val annotations, did Stage 1 / Stage 2 each get the pairwise direction right?]

### 6.5 Sensitivity to `hold_out_every`

[OPTIONAL - if time permits, ablate hold_out_every ∈ {2, 4, 8} to characterize how much human signal the model needs.]

---

## 7. Discussion

### 7.1 What the Disagreement Tells Us

The 46% contradiction rate is one half of the story; the auto-fit/human-alignment crash documented in §6.2 is the other. The fast-solver-diff measures how much LaCAM thinks the residual problem has gotten harder over a 16-step window - a metric that systematically misses two failure modes humans easily catch:

1. Pre-congestion oscillation. Agents dithering in a corridor for several steps look benign by makespan-residual but are clearly the precursor to a collapse a few steps later.
2. Local-deadlock-resolved-by-luck. A segment in which agents block each other but happen to escape may show low `diff` even though the behavior was congested.

The Stage 1 baseline shows that perfectly fitting the auto-pair ordering is not just a noisy approximation of the human-judgment ordering - it is partially anti-aligned with it. The model briefly hits 80.8% human agreement at epoch 3 as a side effect of partially fitting the auto-pairs (the easy confident-vs-confident cases line up with humans), then is dragged back to chance by the ≈12 contradicting annotations bleeding into the gradient as the model fits the rest of the auto-pair distribution. This pathology - training loss down, human alignment down - is what motivates supplanting the auto signal with human verdicts on episodes where humans disagree, rather than augmenting.

[EXPAND: pick 2 concrete annotated examples - e.g. `maze_ms132_ss1001_na32` (auto says diff=2 for "worst", diff=2 for "clean"; human disagreement is purely behavioral) and `maze_ms134_ss1002_na48` (auto diff(worst)=−7 vs auto diff(clean)=12; the human marks the easier-by-LaCAM segment as worse because it shows agents thrashing) - alongside replay-tool screenshots.]

### 7.2 Why Option B Beats Naive Append

The 12 contradicting annotations would actively cancel against their auto twins in a naive append. Option B's "override" pattern recognizes that within an episode the human verdict is the gold standard and the auto-pair distribution should be replaced wholesale - not because auto pairs are useless but because relative signal between annotated segments is the human's domain.

The Stage 2 results confirm this in two ways. First, `hp_train` mostly hovers around 0.84 (16/19) for the bulk of training rather than running away to 1.0, indicating that the model is treating the human verdict as a transferable preference rather than memorising 19 specific `(worst, clean)` pairs - though it does briefly reach 0.895 (17/19) at epochs 27 and 29, the late mild over-fit discussed in §7.3. Second, the auto-aligned metrics (top-3, top-1±1) are essentially identical between stages - which is exactly the desired non-result: human supervision did not come at the cost of auto-pair ranking quality on the held-out map seeds. The override is therefore sufficient in the sense we asked it to be, while remaining cheap (one bit of plumbing in `SegmentPairDataset` and ≈3 hours of annotator time).

### 7.3 The Anti-Collapse Effect

The most striking comparative finding is qualitative, not numerical. Stage 1's `hp_all` traces a sharp inverted-U: 0.577 → 0.808 (epoch 3) → 0.500 (epoch 26). Stage 2's `hp_all` traces a noisy plateau in `[0.54, 0.81]` with peak 0.808 hit on eight different epochs (vs. once in Stage 1) and a late-window mean of 0.708 vs. Stage 1's 0.592 - a 0.116 absolute improvement that holds across every training window we measured. The intervention does not just match Stage 1's brief peak; it converts a one-epoch peak into an eight-epoch regime. A reader looking at the two `hp_all` curves overlaid (Figure 2 / `comparison_1.png`) sees a model that "tries to align with humans then drifts away" in Stage 1 versus one that "stays mostly anchored" in Stage 2 - despite both seeing identical auto-pair gradient flows from the 26100+ unannotated episodes.

We interpret this as the human-pair gradient acting as a regulariser on the auto-fit dynamics. The 19 train annotations contribute a tiny fraction of the gradient (≈19/26146 = 0.07% of pairs), but they are placed exactly on the cases that Stage 1 fits wrong, so they produce gradient pressure precisely where the auto signal is misleading the model. This is consistent with our view of the problem as one of label noise concentrated on borderline cases rather than uniform.

The regularisation effect is not unbounded. After epoch 21, `hp_val` (7) drifts down from a stable ~0.67 plateau to a ~0.51 mean, while `hp_train` (19) shows brief surges to 0.895 - mild over-fitting on the 19 specific train pairs. Two takeaways: (a) the save-best-checkpoint discipline is essential; the deployed model (epoch 1, `hp_val=0.857`) sidesteps the late drift entirely, and (b) for fielding this approach in a continuous DDG loop one would want either explicit early stopping on `hp_val` or a small upweight on the auto-pair loss in late epochs. Despite the late-window drift, Stage 2's `hp_all` in epochs 21-30 (0.708) is still substantially above Stage 1's same window (0.592), so the override-vs-collapse comparison holds across the entire training run, not just early.

### 7.4 Failure Modes and Generalization

- Sample size. 26 annotations is small; with a 19/7 split, the val metric is noisy. The directional comparison between stages is meaningful, the absolute number is not. The 0.857 best `hp_val` corresponds to 6/7, not a precision-3 number.
- Map distribution coverage. Annotators happened to mark only seeds 128-136 - all on the train map split. Spatial generalization to val map seeds (144-147) is therefore evaluated only via auto top-3, where Stage 2 matches Stage 1 (both 0.608).
- Distribution shift across DDG checkpoints. As the policy improves, "what looks congested" changes. Annotations made on rollouts from one checkpoint may not transfer cleanly to later ones.

### 7.5 HRI Implications

The cost of human time was the binding constraint. A 2-3-hour annotation session yielded 26 high-confidence pairs (1 pair per non-skipped episode under the schema we used). That is exactly the regime where pairwise interfaces win over absolute scoring: humans can rank quickly, scoring an absolute "congestion level" would require calibration we do not have. The replay tool reduced per-episode annotation time to roughly [FILL: minutes/episode] by precomputing trajectory animations and providing keyboard shortcuts for the worst/clean/borderline marks. Even at this small budget, the human signal is large enough to flip the model's training trajectory away from the auto-fit collapse: 26 pairs (0.07% of the gradient mix) regularise the model into the human-aligned regime that pure auto-label training cannot stay in.

---

## 8. Limitations and Future Work

- Pure offline training. The classifier is currently trained once after annotations are collected; it is not retrained in-the-loop with DDG. An online integration where the classifier replaces the threshold in `delta_data_generator.py` and is retrained periodically with fresh annotations is the natural next step.
- No downstream MAPF-GPT impact study. This report measures the classifier; we have not yet measured whether using the classifier in DDG improves the downstream MAPF-GPT policy on POGEMA benchmarks. That is the most consequential open question.
- Single annotator. All 28 annotations were collected by one person. Inter-annotator agreement is unknown.
- Threshold calibration on the score head. We rank but do not calibrate: deploying the classifier in DDG requires a decision threshold equivalent to the current `diff > 3`, which we have not yet selected.
- Feature ablations. We did not isolate the contribution of the recent-history channel (channel 3). It would be useful to test whether the model picks up oscillation from this channel specifically.
- Late-epoch over-fit on the 19 train annotations. `hp_train` briefly reaches 0.895 at epochs 27, 29 while `hp_val` mean drops to 0.514 in epochs 21-30. The save-best logic captures the early peak so the deployed checkpoint is unaffected, but for a continuous DDG loop one would want explicit early stopping or a small auto-pair upweight in late epochs.
- Borderline marks not persisted. The annotation-tool schema for this batch captured only `worst` and `clean` indices. Extending the schema to also persist the optional borderline mark and replaying the existing 28 episodes would roughly double the pair budget at near-zero annotator cost.

---

## 9. Conclusion

We replaced the hand-tuned threshold at the heart of the DDG hard-case-mining loop with a small spatio-temporal CNN trained on a pairwise objective. Humans disagreed with the threshold's verdict in 46% of borderline cases, demonstrating that the threshold throws away signal that humans can readily provide. Pure auto-label training exhibits a previously unreported pathology: human-judgment alignment peaks early at 0.808 then collapses to chance even as training loss continues to decrease. With Option B (full override of auto-pair supervision in annotated episodes), 26 human pairs are sufficient to keep the model anchored in the human-aligned ranking regime: 26-pair agreement mean rises by 0.102 across 30 epochs, the 0.808 peak is reached on 8 epochs (vs 1 in the baseline), and the deployed early-checkpoint reaches 0.857 on the 7-pair held-out subset. Auto-aligned ranking quality (top-3) is preserved unchanged. The HRI design choices - pairwise interface, replay-tool annotation surface, deterministic train/val split of the small annotation set - were essential to extracting useful signal from a small budget of human time, and provide a clean blueprint for adding human judgment to any DDG-style data-curation loop.

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

- Hook (revised). "Auto-label training fits perfectly, then forgets what humans were teaching it." The Stage-1 human-pair-acc curve dropping from 0.808 → 0.500 while train loss keeps falling is a one-slide visual that tells the whole motivation.
- Visual hook 1. Side-by-side: auto-pair direction vs human-pair direction on a contradicting annotation, with the replay-tool screenshot.
- Visual hook 2. Stage 1 vs Stage 2 hp_all curves overlaid. Stage 1 peaks at 0.808 then collapses to 0.5; Stage 2 plateaus around 0.78 and hits 0.808 on six different epochs. The "peak vs plateau" contrast is the single best one-slide visual.
- Three-act structure. Problem (DDG threshold is brittle, and fitting it actively hurts) → Method (segment classifier + pairwise human override) → Result (Stage 2 converts a transient peak into a stable regime; auto-aligned metrics unchanged).
- HRI hammer. 26 human pairs (0.07% of training gradient) are enough to flip the model's late-epoch trajectory from collapse to plateau. The pairwise interface is what made it possible to collect them quickly. Asking for absolute scores would have given us nothing usable.
- Surprise finding to highlight. Pure auto-label training does not just plateau in human alignment - it actively drifts away from it. Loss going down, alignment going down. This reframes "human in the loop" as a *regulariser*, not just an *augmentation*.
- Honest limitation. Single annotator, no downstream policy-quality measurement yet.
