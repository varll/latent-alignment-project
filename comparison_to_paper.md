# Our results vs. the original paper (PA-CCS)

Side-by-side comparison of **our** `latent-alignment-project` experiments against the source paper.

**Paper:** *Polarity-Aware Probing for Quantifying Latent Alignment in Language Models* —
Sabrina Sadiekh, Elena Ericheva, Chirag Agarwal. arXiv **2511.21737v1** (Nov 2025).
Code/data: `github.com/SadSabrina/polarity-probing`.

**Is the paper the basis of our work?** **Yes.** Our repo is explicitly "based on the public
`SadSabrina/polarity-probing` repository" (README), we reuse the paper's **mixed dataset**
(`mixed_dataset.csv`, 1244 statements), and we implement the same **PA-CCS** probe with the same
three metrics (Empirical Separation Accuracy, Polar Consistency, Contradiction Index). So the
methods and one dataset are shared; the models are entirely different (see caveats), and we add
several extensions.

> **Reading note.** The paper reports its results almost entirely as **aggregate statistics**
> (mean-absolute-differences, medians, % of layers over a threshold) rather than a per-model
> accuracy table. It gives only two concrete per-model-ish numbers (Fig. 4 medians; Gemma-2-9B-IT
> in §A.3). So most of the comparison below is *trend/order-of-magnitude*, not row-for-row. Where a
> clean numeric alignment is impossible, that is stated instead of inventing one.

---

## 1. Setup comparison

| Dimension | Paper (2511.21737v1) | Ours | Overlap? |
|---|---|---|---|
| **Probe / method** | CCS (Burns et al.) + **PA-CCS** extension; unsupervised linear probe | Same — CCS + PA-CCS, unsupervised linear probe | ✅ same method |
| **Contrast suffixes** | ` Yes` / ` No` | ` Yes.` / ` No.` | ✅ (trailing period) |
| **Hidden-state extraction** | first-token (encoders), last-token (decoders), both (enc-dec) | last-token (all our models are decoders) | ✅ for decoders |
| **Normalization** | "all hidden states normalized, representations mean-centered" | `l2,median` (L2 then median-center), train-fitted | ⚠️ similar intent, different recipe (median vs mean center; we add L2) |
| **Probe training** | 10 runs × 1500 epochs, averaged; per layer | 1500 epochs / 10 tries standard (ToxiGen k-fold lightened to 1000/5) | ✅ matches on mixed runs |
| **Metrics** | ESA (accuracy), Polar Consistency (PC ∈[−1,1]), Contradiction Index (CI ∈[0,2]) | Same three + **silhouette**; PC reported as \|PC\| and signed | ✅ + extension |
| **Models** | **16**, encoder/decoder/enc-dec: DeBERTa (base/large/large-hate), GPT-2 (+large), GPT-Neo-detox, BERT (+2 hate), **Llama-3-8B (base/instruct/guard)**, **Gemma-2 2B & 9B (base/it)** | **7** decoder-only: OLMo-1B base, OLMo-2-1B base/instruct, **Gemma-3-1B base/instruct**, Qwen3-4B instruct, Qwen3-8B base | ❌ **no exact model overlap** (see caveats) |
| **Datasets** | **mixed** (1244), **not** (1250, all-negation), **ttt** control (negation→placeholder) | **mixed** (1244, same file) + **ToxiGen** (annotated/test, single & paired) | ✅ mixed shared; ❌ we skip *not*/*ttt*, we add ToxiGen |
| **Scale range** | 110M → 9B (small `<2B` / large `≥2B`) | 1B → 8B (all ≥1B) | partial |
| **Beyond-latent evaluation** | none (latent-only) | **3-LLM-judge behavior labels, negation-flip triage, knows-vs-does correlations, ToxiGen per-group leak-free k-fold** | ❌ ours only |

**Where we overlap:** the PA-CCS method, the three metrics (same formulas), and the **mixed
dataset**. **Where we differ:** completely different model roster (we run OLMo/Qwen/Gemma-3, the
paper runs DeBERTa/GPT-2/BERT/Llama-3/Gemma-2), we add L2 to normalization, we drop the *not*/*ttt*
control datasets, and we bolt on a whole behavioral/judged layer the paper does not have.

---

## 2. Results comparison

### 2a. The paper's actual reported numbers (with references)

The paper does **not** publish a per-model accuracy table. Its concrete quantitative claims are:

| # | Quantity | Paper value | Source |
|---|---|---|---|
| P1 | MAD of PC / CI across **formulation types** (concurrent vs negation), large models | PC **0.030**, CI **0.073** (small → robust to rephrasing) | §4.2 RQ1; §A.3 |
| P2 | MAD of PC / CI **with vs without `not`** (mixed/not vs ttt control) | PC **0.274**, CI **0.322** (large → sensitive to real polarity) | §4.2 RQ1; §A.3 |
| P3 | % of layers with acc **>0.625** / **≥0.75**, **small** models (mixed+not) | **19%** / **1.6%** | §4.2 RQ1 |
| P4 | % of layers with acc **>0.625** / **≥0.75**, **large** models (mixed+not) | **52.4%** / **28%** | §4.2 RQ1 |
| P5 | Encoder-vs-decoder **median** differences | Acc **0.009**, PC **0.016**, CI **0.023** | §4.2 RQ2 |
| P6 | **Median** PC / CI enabling separation acc **≥0.75** (large models, ≥2B) | PC **0.055**, CI **0.410** | §4.2 RQ4; Fig. 4 |
| P7 | Best-in-class model (Gemma-2-9B-IT) | acc **>0.8**, CI **<0.4** | §A.3 |
| P8 | Inclusion threshold for a model/layer | acc **≥0.625** on one cluster level | §4.2 RQ1 |
| — | Table 1 PC/CI values (0.001–0.003 PC; 0.13–0.74 CI) | **simulated/illustrative, not empirical** | §3.4 Table 1 |

Directional findings (no single number): accuracy **rises with scale and instruction tuning**; PC
**decreases** (more stable) with instruction tuning; CI **decreases** with instruction tuning;
encoders show tighter variance than decoders (§4.2 RQ3–RQ4, §A.3–A.4).

### 2b. Our reported numbers (mixed dataset, 7 models)

Source: `runs/summary_ccs.csv`. `|PC|` = `polar_consistency_absmean`, `CI` = `mean_contradiction_idx`.

| model | params | best_acc | best layer (frac) | mean_acc | max_silhouette | \|PC\| | CI |
|---|---|---|---|---|---|---|---|
| OLMo-1B base | 1B | 0.690 | 8 (0.50) | 0.618 | 0.430 | 0.119 | 0.488 |
| OLMo-2-1B base | 1B | 0.668 | 15 (0.94) | 0.596 | 0.386 | 0.136 | 0.458 |
| OLMo-2-1B instruct | 1B | 0.888 | 16 (1.00) | 0.633 | 0.318 | 0.106 | 0.516 |
| Gemma-3-1B base | 1B | 0.845 | 17 (0.65) | 0.701 | 0.322 | **0.014** | 0.502 |
| Gemma-3-1B instruct | 1B | 0.936 | 14 (0.54) | 0.773 | 0.271 | 0.036 | 0.514 |
| Qwen3-4B instruct | 4B | 0.957 | 21 (0.58) | 0.792 | 0.319 | 0.052 | 0.502 |
| Qwen3-8B base | 8B | **0.968** | 24 (0.67) | 0.785 | 0.246 | 0.045 | 0.503 |

### 2c. Aligning the comparable quantities

Because **no model overlaps** and the paper publishes aggregates, we align on the two things that
*are* comparable: (i) the aggregate PC/CI "ideal" band for high-accuracy models (P6/P7), and (ii)
the qualitative scale/instruction-tuning trend.

| Comparable quantity | Paper | Ours | Verdict |
|---|---|---|---|
| **PC (\|PC\|) for models reaching high separation acc** | median **0.055** for acc ≥0.75 (P6) | Gemma-3-1B-it **0.036**, Qwen3-4B **0.052**, Qwen3-8B **0.045**, Gemma-3-1B base **0.014** | ✅ **strong agreement** — our high-acc models sit right at / below the paper's 0.055 |
| **CI for high-separation models** | median **0.410** (P6); Gemma-9B-IT **<0.4** (P7) | ≈ **0.50** for *every* model (0.46–0.52) | ⚠️ **divergence** — ours ~0.1 higher; we show CI≈0.5 is the *random-probe null at separating layers* (see §3) |
| **Peak accuracy of strongest model** | Gemma-2-9B-IT **>0.8** (P7) | Qwen3-8B base **0.968**, Qwen3-4B **0.957**, Gemma-3-1B-it **0.936** | ✅ consistent (our best clear >0.8) |
| **Accuracy rises with scale + instruction tuning** | yes (RQ3/RQ4) | yes: OLMo-2 base 0.67 < OLMo-1 0.69 < Gemma base 0.85 < OLMo-2-it 0.89 < Gemma-it 0.94 < Qwen-4B 0.96 < Qwen-8B 0.97 | ✅ **reproduced** |
| **Instruction tuning lowers \|PC\| (more stable)** | yes (RQ3) | OLMo-2: 0.136→0.106 ✅; Gemma-3: 0.014→0.036 ✗ (slightly up) | ⚠️ **partial** — holds for OLMo, not Gemma-3 (already near-zero at base) |
| **Signal lives mid-to-late layers** | decoders strongest in middle layers (§A.4) | first-high-layer frac ≥0.38 for all; strong models plateau to top (`summary_high_layers.csv`) | ✅ consistent |
| **Per-model accuracy on *shared* models** | — | — | **n/a — no shared models** |
| **`not` / `ttt` control MADs (P1/P2)** | PC 0.030/0.274, CI 0.073/0.322 | **n/a (not in ours)** — we did not run the *not*/*ttt* datasets on the mixed roster | not comparable |
| **Encoder-vs-decoder medians (P5)** | Acc 0.009, PC 0.016, CI 0.023 | **n/a (not in ours)** — we run decoders only | not comparable |
| **Behavior / judges / ToxiGen k-fold** | **n/a (not in paper)** | ours only (§4 below) | not comparable |

---

## 3. What matches / what diverges

**Did we reproduce their core result?** **Largely yes, on the parts that are comparable.** The
paper's headline is that PA-CCS (a) separates harmful vs. safe internally, (b) improves with **scale
and instruction tuning**, and (c) yields **near-zero PC** for well-aligned models. All three
reproduce on our (different) model set:

- ✅ **Separability + scale/instruction trend reproduced.** Corrected accuracy climbs monotonically
  from OLMo-2 base (0.67) to Qwen3-8B (0.97), and within each family the instruct variant separates
  better — exactly the paper's RQ3/RQ4 claim.
- ✅ **PC "ideal band" reproduced almost exactly.** The paper's median PC enabling ≥0.75 separation
  is **0.055** (Fig. 4); our high-accuracy models land at **0.014–0.052**. Independent models, same
  dataset, essentially the same polarity-consistency magnitude — a genuinely encouraging replication.
- ✅ **Mid-to-late layer localization reproduced.** No model's PA-CCS signal peaks early; strong
  models show a plateau running to the top of the stack.

**Where we diverge:**

- ⚠️ **Contradiction Index.** The paper treats CI as an informative complement and reports a median
  **0.410** (and Gemma-9B-IT **<0.4**). Our CI sits at **≈0.50 for every model, including the
  0.97-accuracy ones**, and our analysis (FINDINGS §5f) shows **CI ≈ 0.5 is precisely the
  random-probe null at the separating layers** — i.e. once the probe works, CI adds no information
  beyond accuracy, and away from those layers it is unstable noise (0.15–1.0), not 0.5. So we read
  CI more skeptically than the paper does. The gap (0.41 vs 0.50) is partly the paper's use of a
  **median over ≥0.75-acc layers** vs our **layer-mean**, and partly a real interpretive
  difference.
- ⚠️ **Instruction-tuning → lower PC** holds for OLMo but not Gemma-3 (whose base is already at
  \|PC\| 0.014). Small n and near-floor values make this fragile.
- ⚠️ **Normalization differs** (we add L2 and center on the median vs the paper's mean-centering),
  which can shift absolute accuracies slightly; treat cross-paper absolute numbers as indicative.

**Why the divergences are expected:** different model families/generations (Gemma-3 vs Gemma-2,
plus OLMo/Qwen the paper never ran), a different normalization recipe, and aggregate-vs-per-layer
reporting. None of these contradict the paper's qualitative story.

---

## 4. Our extensions beyond the paper

Everything here is **not in the paper** (latent-only PA-CCS):

1. **Behavioral "does" layer with a 3-LLM-judge consensus.** Free-form generations labelled
   safe/harmful/gibberish by deepseek-v4-flash, gpt-oss-120b, qwen3, aggregated by majority vote
   (`summary_behavior_consensus.csv`, `summary_judge_reliability.csv`). Includes Fleiss-κ judge
   reliability (κ ranges −0.06 → 0.61) and the finding that the legacy single judge is broken.
2. **Knows-vs-does correlations.** PA-CCS latent metrics vs judged behavior
   (`summary_metric_verdicts.csv`): `best_acc ↔ coherent_rate` **+0.81** (robust) and
   `best_acc ↔ harmful_rate_coherent` **−0.04** (fragile at n=6; was −0.54 at n=4, collapses on the
   Gemma-3-1B-base outlier). Core new claim: **latent separability ≠ behavioral capability;
   coherence must gate any "knows⇒does" reading** (OLMo-1B: 0.69 latent yet 99% gibberish).
3. **Signed Polar Consistency.** We report PC's **sign** (share of layers positive, mean signed PC),
   showing \|PC\| can be misleading where the sign flips across layers (FINDINGS §5g) — a refinement
   the paper's modulus-only PC does not surface.
4. **CI-is-the-null critique.** Accuracy-conditioned CI showing CI≈0.5 = random-probe null at
   separating layers (FINDINGS §5f), i.e. CI is uninformative beyond accuracy on our runs.
5. **Negation-flip triage** (`summary_negation_flip.csv`): residual harm lands on the *benign
   negation*; generation-length separates genuine leakage from strict-judge artefacts.
6. **ToxiGen as a second dataset** — `single` (accuracy/silhouette) and `paired` (recovers PC/CI via
   machine-generated benign rewrites); toxicity direction emerges **late** (~layer 18+, best_acc
   ~0.83 for Qwen3-4B).
7. **Leak-free, pair-aware ToxiGen per-group k-fold** (`runs/toxigen_group_kfold.py`): out-of-fold
   per-group probes, folds over unique polarity pairs. Qwen3-4B-it global OOF **0.832**, per-group
   ~0.72–0.90; Qwen3-8B-base global OOF **0.864**, per-group ~0.73–0.93. Surfaces per-group
   "behaves-worse-than-latent-predicts" groups (asian, black/african-american, jewish folks).
8. **New model families** the paper never probed with CCS: **OLMo / OLMo-2 / Qwen3 / Gemma-3**.

---

## 5. Caveats (comparability limits)

- **No shared models.** The paper runs Gemma-**2** (2B/9B), Llama-3-8B, DeBERTa, GPT-2, BERT,
  GPT-Neo. We run Gemma-**3**-1B, OLMo/OLMo-2-1B, Qwen3-4B/8B. Even the shared *family* (Gemma) is a
  different generation and size. **No row-for-row accuracy comparison is possible** — all model-level
  alignment is trend-level only.
- **Paper reports aggregates, not per-model tables.** Only Fig. 4 medians (PC 0.055 / CI 0.410) and
  Gemma-2-9B-IT (acc>0.8, CI<0.4) are concrete; the rest are MADs / %-of-layers.
- **Different metric summarization.** Paper CI/PC are **medians over ≥0.75-acc layers**; ours are
  **layer-means** (`*_absmean` / `mean_*`). This alone can explain part of the CI 0.41-vs-0.50 gap.
- **Different normalization** (`l2,median` vs mean-centering) can shift absolute accuracy.
- **Datasets only partially overlap.** Mixed dataset is shared; we did **not** run the paper's
  *not* / *ttt* control datasets on this roster, so the paper's central control result (PC/CI MAD
  with vs without `not`, P1/P2) is **not reproduced here**. We instead validate polarity via ToxiGen
  and behavior — a different axis.
- **Small n on our extensions.** Behavior correlations are n=4–6 (descriptive, not inferential);
  the `harmful` link is outlier-driven. ToxiGen per-group n is 13–46 pairs/group. ToxiGen paired
  rewrites are machine-generated (~3% clear artefacts) — treat paired PC/CI as noisy-but-indicative.
- **Judge divergence.** Our behavior layer depends on 3 judges that disagree 3–4× on harmful rate;
  low harmful rates are robust, borderline rates are soft. Judge slugs differ from anything in the
  paper (the paper has no behavioral judging at all).
- **Model-count note.** The paper's abstract says "16 models," the intro says "18," and Table 2
  lists **16**; we treat 16 (Table 2) as authoritative. Our roster is **7** decoder-only models;
  behavior correlations are at n=6.

---

### One-line read-out

We reproduced the paper's **core qualitative result** (PA-CCS separates harmful/safe, improving with
scale and instruction tuning) and its **PC "ideal band" (~0.05)** on an entirely different model
roster and the shared mixed dataset; we **diverge on CI** (ours ≈0.50 = the random-probe null,
which we argue makes CI uninformative beyond accuracy); and we **extend** the paper with a full
behavioral / knows-vs-does / ToxiGen-per-group-k-fold layer it does not contain. Because no models
overlap and the paper reports aggregates, all comparisons are trend/order-of-magnitude, not
row-for-row.
