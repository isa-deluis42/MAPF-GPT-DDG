# Human-in-the-Loop Congestion Classification for Multi-Agent Pathfinding

**Final Project Report — Working Draft**

> **Format target:** 6 pages, double-column. Easiest path is to write here in Markdown, then convert with `pandoc` or paste into the IEEE/ACM double-column LaTeX template before submission. Mark replacements with `TODO` / `[FILL]` / `[FIGURE N]`.
>
> **Due:** May 12 @ 11:59 PM EST.

---

## Authors

[FILL: names]

---

## Abstract

[FILL — write last, ~150 words]

Multi-agent pathfinding (MAPF) policies trained with imitation learning improve when their training data is enriched with examples of *hard cases* — moments during a rollout where the policy gets stuck in congestion. The original Difficulty-Driven data Generation (DDG) pipeline detects these moments with a hand-tuned threshold on a fast solver's makespan-improvement estimate, discarding everything in the borderline range. We argue this throws away signal that humans can readily provide, and that the threshold itself is wrong on a substantial fraction of cases. We introduce a learned segment-ranking classifier that consumes a 16-step spatio-temporal volume of the multi-agent state and is trained on a hybrid of (cheap) auto-labels and (rare) human pairwise verdicts. We collect 28 episode-level annotations through a custom replay tool, find that humans disagree with the auto-label ordering in ≈43% of cases, and show [RESULT: Stage-2 human-pair val accuracy of X compared to Stage-1 baseline Y, while preserving auto top-3 accuracy at Z].

---

## 1. Introduction

### 1.1 Motivation

Coordinating teams of robots in shared environments — warehouses, delivery fleets, search-and-rescue swarms — fundamentally requires solving multi-agent pathfinding (MAPF). Recent work has shown that **transformer policies trained on solver demonstrations** can solve large-scale MAPF instances at inference time with a fraction of the compute an exact solver would need [^mapfgpt]. But these learned policies are only as good as the training data, and they consistently fail on a long tail of *congested* scenarios that look superficially similar to easy ones — agents bunched at corridor pinch-points, deadlocks at junctions, oscillating swap conflicts.

The state-of-the-art remedy is **Difficulty-Driven data Generation (DDG)** [^ddg]: roll out the current policy, identify rollout segments where it appears to struggle, invoke an expensive expert solver on those segments only, and add the resulting expert demonstrations back into the training set. The selection step matters: calling the expert on every segment is wasteful, and missing genuinely hard segments leaves the policy's blind spots unfixed.

Today, segment selection in DDG is performed by a **hand-tuned threshold** on a fast LaCAM probe's makespan-improvement estimate. The threshold is brittle: it discards an entire midrange band of borderline-difficult segments, and — as we show — disagrees with human judgment in nearly half of borderline cases.

### 1.2 HRI Framing

This is a Human-Robot Interaction problem in two ways:

1. **The "robot" is a multi-agent system that operates without per-step human supervision**, but whose long-run training improves when humans inject judgment about which behaviors look problematic. The same problem shape recurs anywhere robot teams operate among or for humans — warehouse fleets, multi-AGV manufacturing, autonomous mobility-on-demand.
2. **The annotation interface itself is an HRI artifact.** Asking humans to rank segments rather than to label them with absolute scores reduces cognitive load and avoids the calibration pitfalls of asking "how congested, on a scale of 1–10?" [^pairwise]. The replay-based segment-marking tool we built is the human-facing surface of an active-learning loop that closes between human judgment and a learned policy at planet-scale.

### 1.3 Contribution

We make three contributions:

1. **A segment-level spatio-temporal congestion classifier** that takes a 16-step volume of (agent positions, obstacles, goals, recent history) and outputs a scalar score, replacing DDG's hand-tuned threshold.
2. **A hybrid pairwise-ranking objective** that mixes cheap auto-labels (from a fast LaCAM probe) with rare, high-confidence human pairwise verdicts collected through a custom replay tool. We show how to *override* — not merely augment — auto pairs in episodes where humans contradict them.
3. **An empirical analysis of human–auto-label disagreement.** Across 26 valid annotations on the held-out seed set, the human's "worst" segment has a *lower or equal* fast-solver-diff than the human's "clean" segment in 12/26 cases, demonstrating that the auto-label is a noisy proxy.

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

Prior HRI work on integrating human feedback into agent training has explored absolute reward shaping (TAMER [^tamer]), preference-based learning (Christiano et al. [^prefs]), and pairwise comparison interfaces [^pairwise]. We adopt **pairwise** rather than absolute ranking specifically because (a) absolute rating of multi-agent congestion is poorly defined and (b) pairwise comparison is the natural interaction granularity for a replay tool.

### 2.5 Pairwise Learning to Rank

RankNet [^ranknet] minimizes a logistic-style loss over score differences between paired examples. This is the loss the trainer in this work uses, with weights derived from the auto-label confidence buckets.

---

## 3. Research Question and Hypotheses

### 3.1 Research Question

> *Can a small set of human-curated pairwise segment rankings improve a learned MAPF congestion classifier — and a downstream DDG pipeline — beyond what is achievable with a hand-tuned threshold on cheap auto-labels alone?*

### 3.2 Hypotheses

- **H1 (auto-label noise).** The fast-solver-diff used by DDG to label segments disagrees with human judgment in a non-trivial fraction of borderline cases.
- **H2 (human pairs generalize).** Replacing auto-derived pair supervision with human pairwise verdicts in annotated episodes improves performance on a held-out subset of human pairs *without* degrading the underlying auto-label-driven ranking quality.
- **H3 (midrange recovery).** The midrange band currently discarded by DDG contains learnable signal recoverable through human annotation.

---

## 4. Method

### 4.1 Held-Out Seed Set

To enable clean evaluation across DDG checkpoints, we reserve a fixed seed set never seen during DDG training: 20 procedurally generated maps (10 maze, 10 random) × 3 scenario seeds × 3 agent counts ∈ {16, 32, 48} = 360 episodes. Map seeds 128–143 form our train split for the classifier (16 maps); 144–147 form val (4 maps).

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

We adapted a `replay.ipynb` notebook to provide a full-episode scrubbable visualization. Per episode the annotator marks:

- A **worst congestion failure** segment ("Fail")
- A **clearly clean** segment ("Pass")
- (optional) A borderline segment

Each annotated episode therefore contributes one human-curated pair `(worst, clean, weight=1.0)`. Annotators were instructed to skip episodes that were uniformly clean, uniformly bad, or indistinguishable. We collected 28 annotations across 9 unique map seeds (all on the train split, none on val by chance); after filtering entries with worst==clean or null indices, 26 are usable.

### 4.7 Hybrid Training (Option B Override)

The simplest integration — appending human pairs to the auto pair set — fails: in 12/26 annotations the human's worst segment has *lower or equal* `diff` than the human's clean segment, so the corresponding auto pair would point in the *opposite* direction and the gradients would partially cancel. Options A (naive append), C (surgical override), D (upweight), and E (re-bucket) were considered (Appendix A).

We adopt **Option B: full override**. For each annotated episode, *all* auto-derived pairs from that episode are removed and replaced with the single human pair at weight 1.0. Auto pairs from non-annotated episodes are unaffected. Other rollouts of the same scenario at different DDG checkpoints — collected at a different policy state and therefore *different rollouts* — are unaffected: only the rollout the human actually saw is overridden.

### 4.8 Evaluation Metrics

We report two complementary metrics each epoch:

- **Auto top-3 accuracy:** for each val episode, fraction in which the auto-label argmax is among the model's top-3 highest-scored segments.
- **Human-pair accuracy:** for each annotation, indicator that `score(worst) > score(clean)`. Reported on three subsets: `all` (26), `train` (19, used for training in Stage 2), and `val` (7, held out from training).

---

## 5. Experimental Setup

### 5.1 Stages

- **Stage 1 (baseline):** Train as-is on auto-labels only. Pass `--annotations` so the script reports human-pair accuracy each epoch, with `--hold_out_every 1` so all 26 annotations land in val and zero override training. The model never sees a human verdict in training; the metric is pure measurement.
- **Stage 2 (Option B):** Same training script with `--hold_out_every 4`. 19 annotations override their episodes' auto pairs at weight 1.0; 7 are held out for human-pair val accuracy.

### 5.2 Hyperparameters

Identical across stages: 30 epochs, batch size 128, Adam lr=3e-4, weight decay 1e-4, base CNN width 16, single segment per training example (`--context_segments 1`), random per-segment 90°-rotation and flip augmentation.

### 5.3 Compute

Single Colab GPU (A100). Training data of ≈26k pairs at batch 128 yields ≈205 batches/epoch. Each stage takes ≈30–60 minutes wall clock.

---

## 6. Results

> **Status:** Stage 1 and Stage 2 runs in progress. Numbers below are placeholders; refresh after both runs complete.

### 6.1 Auto-vs-Human Disagreement (validates H1)

Of the 26 valid annotations:

| Relationship to auto-label ordering | Count |
|---|---|
| Human pair *contradicts* auto-diff (`diff(worst) ≤ diff(clean)`) | 12 |
| Human pair *upgrades* an auto 0.5-weight pair to weight 1.0 | 12 |
| Human pair *creates* a new pair (both midrange, currently skipped) | 1 |
| Human pair *confirms* an existing 1.0 auto pair | 2 (1 dropped due to worst==clean) |

Humans contradict the auto-label ordering in **12/26 (46%)** of annotations. **H1 is supported.**

### 6.2 Stage 1: Baseline Performance

[FIGURE 1: Loss + top-3 + human-pair-acc curves over 30 epochs]

| Metric | Value |
|---|---|
| Best val top-3 accuracy | [FILL] |
| Best human-pair accuracy (all 26) | [FILL] |

> *Partial run early data:* through epoch 3, human-pair accuracy climbed `0.577 → 0.769 → 0.808`. The fact that it can climb *above* the theoretical 14/26 ≈ 54% ceiling implied by pure auto-label-fitting indicates the model is picking up spatial features that correlate with human judgment beyond the literal `diff` ordering.

### 6.3 Stage 2: Option B with Human Pair Overrides

[FIGURE 2: Stage 1 vs Stage 2 — human-pair val accuracy (left), val top-3 accuracy (right)]

| Metric | Stage 1 | Stage 2 | Δ |
|---|---|---|---|
| Best val top-3 accuracy | [FILL] | [FILL] | [FILL] |
| Best human-pair acc (val, 7) | [FILL] | [FILL] | [FILL] |
| Best human-pair acc (all, 26) | [FILL] | [FILL] | [FILL] |
| Best human-pair acc (train, 19) | n/a | [FILL] | — |

[INTERPRETATION TODO based on numbers]

### 6.4 Per-Annotation Breakdown

[FIGURE 3: confusion-style figure — for each of the 7 val annotations, did Stage 1 / Stage 2 each get the pairwise direction right?]

### 6.5 Sensitivity to `hold_out_every`

[OPTIONAL — if time permits, ablate hold_out_every ∈ {2, 4, 8} to characterize how much human signal the model needs.]

---

## 7. Discussion

### 7.1 What the Disagreement Tells Us

The 46% contradiction rate is the headline finding of this work. The fast-solver-diff measures *how much LaCAM thinks the residual problem has gotten harder* over a 16-step window — a metric that systematically misses two failure modes humans easily catch:

1. **Pre-congestion oscillation.** Agents dithering in a corridor for several steps look benign by makespan-residual but are clearly the precursor to a collapse a few steps later.
2. **Local-deadlock-resolved-by-luck.** A segment in which agents block each other but happen to escape may show low `diff` even though the *behavior* was congested.

[EXPAND: pick 2 concrete annotated examples, ideally with screenshots from the visualizer]

### 7.2 Why Option B Beats Naive Append

The 12 contradicting annotations would actively cancel against their auto twins in a naive append. Option B's "override" pattern recognizes that within an episode the human verdict is the gold standard and the auto-pair distribution should be replaced wholesale — not because auto pairs are useless but because *relative* signal between annotated segments is the human's domain.

### 7.3 Failure Modes and Generalization

- **Sample size.** 26 annotations is small; with a 19/7 split, the val metric is noisy. The directional comparison between stages is meaningful, the absolute number is not.
- **Map distribution coverage.** Annotators happened to mark only seeds 128–136 — *all* on the train map split. Spatial generalization to val map seeds (144–147) is therefore evaluated only via auto top-3.
- **Distribution shift across DDG checkpoints.** As the policy improves, "what looks congested" changes. Annotations made on rollouts from one checkpoint may not transfer cleanly to later ones.

### 7.4 HRI Implications

The cost of human time was the binding constraint. A 2–3-hour annotation session yielded ≈30 high-confidence pairs. That is exactly the regime where pairwise interfaces win over absolute scoring: humans can rank quickly, scoring an absolute "congestion level" would require calibration we do not have. The replay tool reduced per-episode annotation time to roughly [FILL: minutes/episode] by precomputing trajectory animations and providing keyboard shortcuts for marking segments.

---

## 8. Limitations and Future Work

- **Pure offline training.** The classifier is currently trained once after annotations are collected; it is not retrained in-the-loop with DDG. An online integration where the classifier replaces the threshold in `delta_data_generator.py` and is retrained periodically with fresh annotations is the natural next step.
- **No downstream MAPF-GPT impact study.** This report measures the classifier; we have not yet measured whether *using* the classifier in DDG improves the downstream MAPF-GPT policy on POGEMA benchmarks. That is the most consequential open question.
- **Single annotator.** All 28 annotations were collected by one person. Inter-annotator agreement is unknown.
- **Threshold calibration on the score head.** We rank but do not calibrate: deploying the classifier in DDG requires a decision threshold equivalent to the current `diff > 3`, which we have not yet selected.
- **Feature ablations.** We did not isolate the contribution of the recent-history channel (channel 3). It would be useful to test whether the model picks up oscillation from this channel specifically.

---

## 9. Conclusion

We replaced the hand-tuned threshold at the heart of the DDG hard-case-mining loop with a small spatio-temporal CNN trained on a pairwise objective. Humans disagreed with the threshold's verdict in 46% of borderline cases, demonstrating that the threshold throws away signal that humans can readily provide. With Option B — full override of auto-pair supervision in annotated episodes — we [SUMMARY OF RESULT, written after numbers come in]. The HRI design choices (pairwise interface, replay-tool annotation surface, deterministic train/val split of the small annotation set) were essential to extracting useful signal from a small budget of human time.

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

---

## Appendix A: Integration Options Considered

| | What it does | Verdict |
|---|---|---|
| **A. Naive append** | Add human pairs at weight 1.0 alongside auto | Contradictions cancel against auto twins |
| **B. Override** *(adopted)* | For annotated episodes, replace all auto pairs with the single human pair at weight 1.0 | Clean; eliminates contradictions |
| **C. Surgical override** | Drop only auto pairs *involving* `worst_idx` or `clean_idx` | Marginal gain over B at higher complexity cost |
| **D. Upweight** | Append human at higher weight (2.0–5.0) without removing auto pairs | Contradictions still present; just lets humans win the gradient war |
| **E. Re-bucket** | Use human verdict as ground truth for the marked indices and force their buckets | Over-extrapolates from 2 segments to a whole episode |

---

## Appendix B: Per-Annotation Disagreement Examples

[OPTIONAL: pick 3 representative annotations; show segment_diffs alongside human worst/clean indices and a short caption explaining what the human saw that the diff missed.]

---

## Slide Notes (for later)

Key talking points to lift directly into slides:

- **Hook.** "The standard DDG pipeline silently throws away half its borderline cases. Humans recover them."
- **Visual hook.** Side-by-side: auto-pair direction vs human-pair direction on a contradicting annotation.
- **Three-act structure.** Problem (DDG threshold is brittle) → Method (segment classifier + pairwise human override) → Result (X% improvement on val without losing top-3).
- **HRI hammer.** The pairwise interface is what made 28 annotations enough to matter. Asking for absolute scores would have given us nothing usable.
- **Honest limitation.** Single annotator, no downstream policy-quality measurement *yet*.
