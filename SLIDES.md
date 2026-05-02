---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Human-in-the-Loop Congestion Classification · HRI Final · 2026-05-03'
style: |
  section { font-size: 26px; padding: 40px 60px; }
  h1 { font-size: 38px; }
  h2 { font-size: 32px; }
  table { font-size: 0.85em; }
  ul, ol { line-height: 1.4; }
  .small { font-size: 0.78em; color: #555; }
  .hl   { background: #fffbcc; padding: 1px 5px; border-radius: 3px; }
  .num  { font-weight: bold; color: #c0392b; }
  .speaker { color: #999; font-size: 0.7em; margin-top: 8px; }
---

<!-- ─────────────────────────────────────────────────────────────────── -->
<!-- TIMING TARGET: 4:00 total, 1:00 Q&A.                                  -->
<!-- 8 content slides @ ~30s each. Speakers labelled per slide.            -->
<!-- ─────────────────────────────────────────────────────────────────── -->

# Human-in-the-Loop Congestion Classification for Multi-Agent Pathfinding

### Replacing DDG's hand-tuned threshold with learned, human-aligned segment ranking

Shane Pornprinya · Isabel De Luis · Sparsh Bansal

<div class="speaker">[ALL THREE briefly introduce themselves — 10s ]</div>

---

# Motivation: where MAPF policies break

- Multi-agent pathfinding (MAPF) coordinates teams of robots in shared spaces — warehouses, delivery, search-and-rescue.
- State-of-the-art: MAPF-GPT, a transformer policy imitation-trained on solver demos. Cheap at inference, brittle on the long tail.
- Failure mode: <span class="hl">**congestion**</span> — agents bunched at corridor pinch-points, deadlocks, swap conflicts.
- The HRI question: <span class="hl">**there is currently no human in this loop.**</span> The policy is trained on solver demos and refined by a hand-tuned threshold. Where does human judgement help?

[FIGURE: small diagram — robots-in-warehouse cartoon → MAPF-GPT box → "fails on tail"]

<div class="speaker">SPARSH · 30s · sets the HRI hook</div>

---

# Problem statement & research question

**The current fix (DDG):** at every 16-step segment, run a 2s LaCAM probe; if `max_diff > 3` → invoke 10s expert solver. Otherwise discard.

```
if max_diff > 3:    invoke expert    # confident hard case
elif max_diff > 1:  discard          # midrange — DDG throws away
else:               discard          # confident easy case
```

> **RQ.** Can a small set of human-curated pairwise rankings improve a learned MAPF congestion classifier beyond what is achievable with the threshold — and does the **strategy used to elicit those rankings** matter?

<div class="speaker">SPARSH · 25s · framing the question</div>

---

# Why it's hard

1. **The auto-label is *anti-aligned* with humans on hard cases.** 23/52 (<span class="num">44%</span>) of human-marked `(worst, clean)` pairs have `diff(worst) ≤ diff(clean)` — the auto-pair points the wrong way.
2. **Naive integration fails.** Appending human pairs alongside auto pairs makes the contradicting twins cancel each other in the gradient. Net signal lost.
3. **Human time is the binding constraint.** ~5 min/episode × 56 budget. Every annotation has to count — making selection strategy itself part of the problem.

<div class="speaker">SHANE · 30s · the three open challenges</div>

---

# Key insight & method

**Insight 1 — Override, don't augment.** For each annotated episode, *replace* the auto pair-list with the human pair at weight 1.0. 39 train pairs = 0.15% of gradient — but placed exactly on the cases auto fits wrong.

**Insight 2 — Strategy of elicitation matters.** Compare un-prioritised hand-curation vs uncertainty × diversity active learning vs pure-diversity AL, all at the same label budget.

**Method:** small 3D CNN over a 16-step (4-channel × T × H × W) spatial-temporal volume → RankNet pairwise loss; pool-based active query with acquisition `H(σ(s_A − s_B)) · ‖φ_A − φ_B‖`.

[FIGURE: 3D-CNN box + replay-tool screenshot side-by-side]

<div class="speaker">SHANE · 35s · technical core</div>

---

# Results: hand-curation lifts every gold-label metric

On the **same 13 held-out human pairs**, comparing best checkpoint per stage:

| Metric | Stage 1 (auto only) | Stage 2 (+39 human pairs) | Δ |
|---|---|---|---|
| **Pairwise** (hp_v on 13)        | 0.615 (8/13) | <span class="num">**0.846 (11/13)**</span> | **+0.231** |
| **Exact-match** (h-arg top-1)    | 0.538 (7/13) | <span class="num">**0.615 (8/13)**</span>  | **+0.077** |
| DDG-aligned `argmax_pm1` (auto val maps) | 0.550 | 0.541 | −0.009 |
| DDG-aligned `top-3` (auto val maps)      | 0.578 | 0.559 | −0.019 |

[FIGURE: `comparison_2.png` — 3 panels overlaid (hp_val, h-arg top-1, auto pm1) for both stages]

→ +3 of 13 pairs ranked correctly, +1 of 13 worst-segments identified exactly, **no measurable cost on auto-aligned ranking**.

<div class="speaker">ISABEL · 45s · headline numbers</div>

---

# Where we sit & the active-learning extension

| Approach | Human in loop? | Selection signal | Limit |
|---|---|---|---|
| MAPF-GPT | no | offline solver demos | long-tail miss |
| DAgger / DDG | no | hand-set probe threshold | wrong on 44% borderline |
| **Our Stage 2** | yes | un-prioritised replay scrubbing | constant time/episode |
| **Stage 3 — warm-start AL** | yes | uncertainty × diversity | needs prior model |
| **Stage 4 — cold-start AL**  | yes | pure feature diversity | no info-gain signal |

- Stages 3 and 4 use the **same 56-label budget** as Stage 2; only the *acquisition* differs.
- Comparison directly addresses external review feedback ("multiple AL strategies").
- <span class="hl">[Stages 3 & 4 results: numbers pending — comparison framework itself is the contribution.]</span>

<div class="speaker">ISABEL · 30s · the empirical-depth pitch</div>

---

# Conclusion & limitations

**Walk-aways.**
- 39 human pairs (0.15% of gradient) → +23 pp pairwise / +8 pp exact-match human alignment, no DDG cost.
- The **save-best criterion** is itself a deployment choice: 4 saved checkpoints differ by 31 pp on the top-1 metric.
- The **acquisition strategy** comparison (Stages 1–4 at fixed budget) is the contribution that adds depth.

**Honest limitations.**
- Run-to-run variance is large (`hp_val` swung 0.615 ↔ 0.846 across two reseeds). Multi-seed averaging is the obvious follow-up.
- Stages 3 & 4 single-seed at submission time.
- No downstream MAPF-GPT impact study yet — does using this classifier in DDG actually train a better policy on POGEMA? Most consequential open question.

<div class="speaker">ISABEL · 25s · close + invite Q&A</div>

---

<!-- BACKUP / Q&A SLIDES (do not present unless asked) -->

# Backup A — auto-label disagreement breakdown

| Relationship to auto-diff ordering | Count |
|---|---|
| Human pair **contradicts** auto-diff (`diff(worst) ≤ diff(clean)`) | 23 |
| Human pair **upgrades** an auto 0.5-weight pair to 1.0           | 23 |
| Human pair **creates** a new pair (both segments same bucket)    | 3  |
| Human pair **confirms** an existing 1.0 auto pair                | 3  |

44% contradiction rate stable to ±2 points across two annotation batches → not annotator noise.

---

# Backup B — save-best criterion grid (Stage 2)

| Stage 2 ckpt (saved by) | hp_v | h-arg top-1 | h-arg ±1 | argmax_pm1 |
|---|---|---|---|---|
| best `hp_v`             | 0.846 | 0.538 | 1.000 | 0.474 |
| best `argmax_pm1`       | 0.615 | 0.308 | 0.846 | **0.541** |
| best `h-arg ±1`         | 0.846 | 0.538 | 1.000 | 0.474 |
| **best `h-arg top-1`**  | **0.846** | **0.615** | 0.923 | 0.514 |

Deploy the bottom row: best deployment-shape metric, ties for best pairwise, only −0.036 on auto top-1±1.
