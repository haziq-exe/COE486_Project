# Spatial grounding in a frozen LLaVA-1.5-7B

Training-free interventions on CV-Bench spatial reasoning, plus one small trained module.

On CV-Bench Relation, LLaVA-1.5-7B answers at 64.8%. The coordinates recovered from the
model's own attention answer the same 650 questions at 84.5%, with 100% localisation
coverage. The model is looking in the right place and still getting the answer wrong, so
perception is not the bottleneck. This repo measures that ~20-point gap and tests six ways
of closing it.

The main finding is that what you deliver matters less than how. Handing the model the same
coordinates as raw numbers makes it worse (-8.5 points). Handing them over as a relational
sentence gains +12.0. The comparison has to happen in language, not in the model's head.

## Results

All numbers are CV-Bench, LLaVA-1.5-7B, greedy decoding, paired over identical item sets.
95% CIs are percentile bootstrap (5000 resamples); p is exact McNemar.

### Table 1: perception vs behaviour (Relation, n=650)

| | |
|---|---|
| Vanilla accuracy | 0.648 [0.611, 0.685] |
| Coordinate oracle (same attention coords, oracle comparison) | 0.845 |
| Gap the model leaves on the table | +0.197 |
| Localisation coverage | 1.00 |
| Oracle under swapped coords (sanity: should be ~1 - oracle) | 0.155 |

### Table 2: same coordinates, different delivery (Relation, n=650)

| Condition | Acc | Delta vs vanilla | p | Swap acc | Real - swap |
|---|---|---|---|---|---|
| `vanilla` | 0.648 | | | | |
| `coords_xy` (raw x, y in the prompt) | 0.563 | -0.085 [-0.129, -0.042] | 0.0001 | 0.546 | +0.017 |
| `steer_abs` (absolute latent steering) | 0.669 | +0.022 [-0.014, +0.057] | 0.265 | | |
| `steer_rel` (relative latent steering) | 0.685 | +0.037 [+0.006, +0.068] | 0.026 | 0.568 | +0.117 |
| `nl_absolute` (each object's region in words) | 0.686 | +0.039 [+0.006, +0.072] | 0.026 | | |
| `brief_2d` (the relation as a sentence) | 0.768 | +0.120 [+0.083, +0.159] | <0.0001 | 0.219 | +0.549 |

The swap control is the one that matters. Feed `brief_2d` the wrong relation and accuracy
collapses to 0.219, so the model is reading the sentence rather than being nudged by its
presence.

### Table 3: un-gated injection damages 3D tasks

Delta accuracy vs vanilla. Depth and Distance are not extra benchmarks; they test whether a
2D intervention breaks tasks it cannot help.

| Condition | Relation | Depth | Distance |
|---|---|---|---|
| `nl_absolute` | +0.039 | -0.073 | -0.040 |
| `coords_xy` | -0.085 | -0.240 | -0.017 |

`brief_2d` states only what 2D attention supports, but the un-gated version still costs
-0.220 on Depth (0.783 to 0.563). A 2D spatial brief actively misleads a 3D question.

### Table 4: routing fixes it, for free

Inject the sentence only when the asked relation is decidable from image-plane coordinates
(choices are a subset of left/right/above/below/top/bottom); otherwise answer vanilla. This
costs zero extra generations, since it is assembled from the same cache.

| Task | n | Router fires | Vanilla | Always-inject | Delta | Routed | Delta |
|---|---|---|---|---|---|---|---|
| Relation | 650 | 100% | 0.648 | 0.768 | +0.120 | 0.768 | +0.120 |
| Depth | 300 | 0% | 0.783 | 0.563 | -0.220 | 0.783 | 0.000 |
| Distance | 300 | 0% | 0.510 | 0.490 | -0.020 | 0.510 | 0.000 |
| Suite (pooled) | 1250 | | 0.647 | 0.652 | +0.005 | 0.710 | +0.062 [+0.042, +0.083] |

Always-on injection nets +0.005 across the suite: the Relation gain is almost exactly
cancelled by the Depth damage. Routing keeps the gain at zero collateral, +0.062 pooled.

### Table 5: the learned relation token (held-out test, n=131)

A small module (about 18M params, against 7B frozen) reads the two coordinates and writes
one token into the residual stream at layer 14. It is trained by context distillation from
the `brief_2d` teacher, with counterfactual coordinate twins, and scored only on the
held-out split.

| Condition | Acc | Delta vs vanilla | p | Swap acc | Real - swap |
|---|---|---|---|---|---|
| `vanilla` | 0.672 | | | | |
| `brief_2d` (the teacher) | 0.725 | +0.053 [-0.031, +0.137] | 0.281 | 0.252 | +0.473 |
| `abstractor` | 0.771 | +0.099 [+0.008, +0.191] | 0.047 | 0.298 | +0.473 |

The student beats its teacher on this split, with a CI excluding zero and real-minus-swap of
+0.47, so the gain flips when the relation flips. That means the relation is being used, not
just the perturbation. Distillation reaches 0.846 val accuracy against 0.692 for plain CE.

## Repo contents

```
cv-project-final.ipynb    the whole thing, run top to bottom
README.md
```

The notebook is organised so that Part F is the paper; everything before it exists to make
Part F run.

| Part | Contents | GPU |
|---|---|---|
| A | Setup, config | no |
| B | Pure-logic core: scorer, statistics, 34 in-notebook unit tests | no |
| C | LLaVA-1.5-7B load, plus the 576-token geometry assertion | yes |
| D | CV-Bench (the only dataset): Relation, Depth, Distance views | yes |
| E | Method: attention to coords, text deliveries, latent steering, abstractor | yes |
| F | Results: Tables 1-5, Figures 1-4, all written to RES_DIR | yes |

## Method

```
attention -> coordinates -> delivery -> generate -> score
```

Every condition reads the same coordinates via one shared config (`RES_LOC`), and every
steering condition uses one shared config (`RES_STEER`), so conditions differ only in the
delivery mechanism.

Why the coordinates are trustworthy: visual token i is exactly patch i of the CLIP 24x24
grid (row-major, CLS dropped), so a 24x24 attention map indexes the 576 visual tokens with
no learned alignment. Part C asserts this against the live processor and refuses to continue
if it fails.

Two details matter for localisation quality:

* Relative attention. Subtract the baseline looking pattern (mean attention-to-image over
  generic text positions) to remove position bias and attention sinks. Without it the maps
  mostly show where LLaVA always looks.
* Denoise before smooth. Zero the bottom quantile of attention mass first, so the 3x3 filter
  cleans rather than smears the noise floor.

## Running it

Built for 2x T4 (Kaggle), ~16 GB each; the model is sharded with `device_map="auto"`.

1. Open the notebook and Run All. It is designed for one top-to-bottom pass.
2. Part B's self-checks gate everything downstream. If they fail, stop.
3. Outputs go to `RES_DIR`, which defaults to `/kaggle/working/results_v1` if present and
   `./results_v1` otherwise. Override with the `RES_DIR` env var.

Resume-safety: every prediction is appended to `RES_DIR/predictions.jsonl`, keyed by
(task, item, condition, swap). Re-running any cell skips what is already on disk, so a dead
session costs nothing. Kill it, restart, re-run.

### Compute budget

| Section | Time on 2x T4 |
|---|---|
| §2 Table 1 | 40-60 min |
| §3 all conditions x all tasks | 4-6 h (the long one) |
| §7 abstractor training | 2-3 h, plus ~25 min teacher cache, once |
| §8 Table 5 | 30 min |
| §4, §5, §6, §9 | minutes |

If pressed for time, cut `N_3D` to 200 and drop `steer_rel` from `REL_SWAPS`, both in §0/§3.

### Outputs

```
results_v1/
├── predictions.jsonl              every generation, resume-safe
├── loc_cache.json                 cached coordinates, reused by every condition
├── relation_split.json            the fixed 70/10/20 split
├── table1_oracle.json
├── table2_relation.json / .tex
├── table3_crosstask.json
├── table4_routed.json
├── table5_abstractor_test.json
├── grad_probe.json                §6 verdict
├── abs_train_log_{ce_noscaler,kl_ce}.json
├── abstractor_relation_kl_ce.pt
└── figs/
    ├── fig1_main_relation.png     bars + CIs, with the oracle ceiling
    ├── fig2_crosstask_delta.png   the degradation heatmap
    ├── fig3_training_curves.png   CE vs distillation
    └── fig4_qualitative.png       vanilla-wrong to brief-fixed, with attention maps
```

## Experimental hygiene

* Swap controls throughout. Every relation-carrying condition is also run with the two
  candidates' coordinates swapped. A delivery only counts if real-minus-swap is large, which
  is what separates "the model used the relation" from "the model was nudged by extra text".
* Full-coverage conditions. When localisation fails, a condition falls back to the cached
  vanilla answer (flagged and counted), so every condition covers the identical item set and
  paired bootstrap and McNemar stay valid. Observed fallback rate: 0.0.
* One localisation config and one steering config, shared by every condition.
* Fixed 70/10/20 Relation split. The trained abstractor is only ever scored on test.
* The scorer is adversarial to itself. Hedging ("either A or B") scores wrong rather than
  getting first-mention luck. Whole-token matching, so `art` never matches `heart`.
  Unparseable output abstains and is never silently correct. Number words match digits and
  spatial synonyms are canonicalised. All 34 assertions run before any model loads.
* The leaky ablation is kept on purpose. `brief_ref` adds an image-plane proximity claim,
  which does leak into Distance. It is reported rather than hidden, to quantify 2D-to-3D
  leakage.
* Deterministic: greedy decoding, every RNG seeded (SEED = 1234).

## Ablations

| Question | Where |
|---|---|
| Is it the relation, or just extra text? | swap controls (Tables 2, 5) |
| Does 2D leak into 3D? | `brief_ref` (Table 3) |
| Is routing doing the work? | router on/off (Table 4) |
| Does the objective matter? | CE vs KL-distillation, scaler on/off (Fig 3) |
| Does augmentation matter? | counterfactual coordinate twins (§7) |
| Relation or coordinate in latent space? | `steer_rel` vs `steer_abs` (Table 2) |

## Caveats

* One model (LLaVA-1.5-7B), one benchmark (CV-Bench). The routing result is a selectivity
  claim on three tasks, not a general one.
* The router is a hand-written rule over the answer choices, not learned. It fires at 100%
  on Relation and 0% on Depth/Distance by construction. The gain is real, but the gating is
  trivially correct here and would need learning on messier task mixes.
* Table 5's n=131 is small. `brief_2d`'s delta on that split does not reach significance
  even though it clearly does on the full 650.
* §6's gradient probe records no underflow signature; gradients flow at both loss scales. The
  distillation win in §7 is therefore about the objective, not fp16 numerics.

## Citation

```bibtex
@misc{spatial-grounding-llava,
  title  = {Spatial grounding in a frozen LLaVA-1.5-7B},
  note   = {Course project / preprint},
  year   = {2026}
}
```
