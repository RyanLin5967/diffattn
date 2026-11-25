# Differential Attention vs. a QK-norm baseline: a matched, pre-registered small-scale replication

*Workshop draft. Every number below comes from the released per-step JSON logs via
`analysis/analyze.py` (`analysis/results.json`).*

## Abstract
We test whether Differential Attention (DiffAttn) beats a standard GPT, and whether QK-norm recovers
its gain, with three matched arms (standard, QK-norm, DiffAttn), an equal learning-rate sweep per arm,
five seeds per arm, and a pre-registered primary metric and tests. At the scale free compute allowed
(51M parameters, 118M training tokens per run, 15 main runs on Kaggle T4s), **the standard baseline had
the lowest held-out loss**: 4.079 ± 0.010 vs 4.168 ± 0.024 for DiffAttn (difference 0.089, 95% CI
[0.059, 0.119], Welch p = 5e-4) and 4.223 ± 0.008 for QK-norm (p = 9e-9). DiffAttn beat QK-norm by
0.055 (p = 0.005). DiffAttn gave no gain for QK-norm to recover, so the recovery question is undefined at
this scale. DiffAttn did reproduce the paper's mechanism signature: on a key-retrieval probe its
final-position attention put 3.2x more of its (scale-free) mass on the answer span than the baselines did.
These results hold for learning rates chosen by the pre-registered single-seed, quarter-budget sweep;
they do not show that DiffAttn or QK-norm are worse when each is tuned at full budget, and the
std-DiffAttn gap was still narrowing at the end of training.

## 1 Introduction
Differential Attention (DiffAttn; Ye et al., 2024) replaces softmax attention with the difference of two
softmax maps, (A1 - lambda A2) V, and reports lower language-modeling loss than a standard Transformer.
Many recent baselines include QK-norm (RMSNorm on queries and keys), which also controls attention-logit
growth. If part of DiffAttn's gain comes from taming attention logits, a QK-norm baseline may recover it.
We test this with three matched arms, an equal-effort learning-rate sweep for each, five seeds per arm,
and a pre-registered primary metric and tests.

## 2 Setup
- **Arms.** `std`: GPT with RoPE, SwiGLU, RMSNorm, tied embeddings, no biases. `qknorm`: `std` + per-head
  RMSNorm on q and k. `diff`: `std` with DiffAttn exactly as in the reference implementation (lambda =
  exp(lq1.lk1) - exp(lq2.lk2) + lambda_init, lambda_init = 0.8 - 0.6 exp(-0.3 l), per-head RMSNorm,
  (1 - lambda_init) output scale, half the heads so all projection matrices match the baseline's shapes).
  The `diff` arm has no QK-norm, as in the paper.
- **Model.** d=512, 8 layers, 8 heads (4 differential heads), SwiGLU hidden 1408, context 1024, GPT-2 BPE;
  ~51M parameters including the tied embedding (all arms within 0.01%).
- **Data.** FineWeb-Edu sample-10BT, first parquet shard; the first 10M tokens of documents are a
  held-out validation set. Seed s fixes init and a global shuffle of 1024-token windows; the same seed gives
  the identical batch sequence in every arm.
- **Training.** AdamW (0.9, 0.95, wd 0.1 on matrices), 2% warmup then cosine to 10% of peak, batch 65,536
  tokens, fp16 with dynamic loss scaling, grad clip 1.0, Kaggle T4 GPUs.
- **Tuning.** Each arm: lr in {1e-3, 2e-3, 4e-3} at 25% of the main budget (seed 100), extended by one
  factor of 2 if the best value is on the edge; best lr per arm used for 5 seeds (0-4) at full budget.
- **Pre-registration** (`docs/prereg.md`, committed before any GPU run): primary metric = final val loss;
  two-sided Welch t-test and exact paired sign-flip test, alpha 0.05; secondary = zero-shot HellaSwag,
  ARC-Easy, PIQA, LAMBADA and an attention-noise probe.

## 3 Results
**Tuning (seed 100, 450 steps = 29.5M tokens).** Lower learning rates won in every arm, so the
pre-registered edge rule added 5e-4 to every grid. Chosen: std 1e-3, qknorm 5e-4, diff 1e-3. QK-norm's
grid was non-monotonic (5e-4: 5.457, 1e-3: 5.475, 2e-3: 5.667, 4e-3: 5.519) and its choice sits on the
edge of the extended grid; see Limitations.

**Main runs.** All 15 runs finished 1,800 steps (117.96M tokens); none diverged; no fp16 overflow steps
were skipped in any run. Parameters: std 51,454,464; qknorm 51,455,488; diff 51,457,536. Model
FLOPs/token: 3.59e8 (std, qknorm), 3.84e8 (diff, +7%). Measured T4 MFU (fp16 dense peak 65 TFLOP/s,
seed-0 runs, last log window): 24.5% std, 24.3% qknorm, 22.2% diff.

Table 1. Final validation loss (10M held-out tokens), 5 seeds per arm.

| arm | mean ± sd | per seed (0-4) |
|---|---|---|
| std | **4.0787 ± 0.0096** | 4.0891, 4.0681, 4.0841, 4.0837, 4.0687 |
| qknorm | 4.2231 ± 0.0081 | 4.2311, 4.2247, 4.2094, 4.2247, 4.2255 |
| diff | 4.1677 ± 0.0242 | 4.1496, 4.1872, 4.1999, 4.1556, 4.1463 |

| comparison | difference | 95% CI (Welch) | Welch p | paired sign-flip p | MDE (80% power) |
|---|---|---|---|---|---|
| std − diff | −0.089 | [−0.119, −0.059] | 5.0e-4 | 0.0625 (floor) | 0.037 |
| std − qknorm | −0.144 | [−0.157, −0.131] | 8.7e-9 | 0.0625 (floor) | 0.018 |
| qknorm − diff | +0.055 | [+0.026, +0.085] | 5.0e-3 | 0.0625 (floor) | 0.037 |

The exact paired sign-flip test was pre-registered with the note that 5 pairs cannot go below p =
0.0625; every comparison reached that floor (all 5 seeds agree in sign). The DiffAttn gain over the
standard baseline is G = −0.089 (paired seed bootstrap 95% interval [−0.109, −0.069]; G > 0 in 0% of
resamples), so the pre-registered recovery ratio R = (L_std − L_qknorm)/G is not meaningful here.

Figure 1 (`analysis/curves.png`): validation loss with seed bands, and lambda per layer for DiffAttn.
On the 2M-token val prefix, QK-norm is lowest early (6.6M and 13.1M tokens), std is lowest from 19.7M
tokens on, and DiffAttn passes QK-norm at ~46M tokens; the final order std < diff < qknorm holds from
45.9M tokens to the end. Learned lambda ends between 0.15
(layer 0) and 0.68 (layer 5).

Table 2. Zero-shot accuracy, mean over 5 seeds (%, sd in parentheses). All are near chance at this scale.

| arm | HellaSwag (acc_norm) | ARC-Easy (acc_norm) | PIQA (acc) | LAMBADA (acc) |
|---|---|---|---|---|
| std | 25.0 (0.2) | 32.5 (0.3) | 56.2 (0.3) | 7.0 (1.3) |
| qknorm | 24.9 (0.2) | 31.1 (0.9) | 56.0 (0.6) | 3.2 (0.3) |
| diff | 25.0 (0.1) | 31.9 (0.4) | 55.8 (0.4) | 4.8 (0.7) |

## 4 Mechanism check: attention noise
200 fixed prompts (~900 tokens of held-out text with one "The secret number of NAME is NNNN." sentence
at a random depth, ending in the matching question). For the final position we take the head-averaged
effective attention map per layer (A1 − lambda A2 for DiffAttn) and compute the answer share
|answer| / (|answer| + |noise|), where noise is all context except the first token, the needle sentence
and the query; this is scale-free because DiffAttn rows sum to 1 − lambda, not 1.

| arm | answer share (mean over layers, seeds) | last-layer share | retrieval acc |
|---|---|---|---|
| std | 0.0125 | 0.038 | 2.1% |
| qknorm | 0.0132 | 0.039 | 2.6% |
| diff | **0.0396** | **0.126** | 0.8% |

DiffAttn concentrates 3.2x more of its attention on the answer span than either baseline (3.3x in the
last layer), qualitatively matching the paper's noise-cancellation claim, but none of the models can
actually retrieve the number at this scale (all ≤ 2.6%), so the mechanism did not translate into
better retrieval or lower loss here.

## 5 Limitations
- One scale (~51M parameters) and 118M tokens per run (~2.3 tokens/parameter, far from compute-optimal),
  set by free compute; the study was specified at 350M parameters and 10B tokens. The original paper's
  gains are reported at much larger scale and budget; ours may not transfer either way.
- Learning rates were tuned with one seed at 25% of the budget. All arms preferred low LRs at that
  horizon, and QK-norm's choice (5e-4) sits on the edge of the extended grid, so QK-norm in particular may
  be under-tuned for the full budget; QK-norm is usually paired with higher LRs. The QK-norm result is the
  least robust conclusion here.
- Sweep sensitivity is as large as the effects. At 450 steps a factor-2 LR change moved loss by +0.13 to
  +0.30 for std and DiffAttn (+0.017 for QK-norm), and QK-norm's single-seed grid is non-monotonic by 0.15,
  which bounds sweep noise at roughly the size of the main-run gaps. A parabola fit in log2(LR) through
  each arm's three best sweep points puts the interpolated optimum within 0.014 of the chosen LR at 450
  steps, but the full-budget (1800-step) optimum was never measured.
- The gaps are budget-dependent: std − diff shrank from 0.114 (step 1000) to 0.089 (step 1800); std −
  qknorm grew from 0.136 to 0.144. Longer training could change the DiffAttn comparison.
- QK-norm uses unit-RMS q, k with learnable gains initialised at 1 and the default 1/sqrt(64) scale, which
  bounds early attention logits (|logit| <= 8); this may handicap QK-norm at low LR (untested).
- Seed 4 ran in a different Kaggle session for diff than for std and qknorm (same GPU type, torch version
  and code); seeds 0-3 were split by session with all three arms of a seed in the same session.
- One dataset (FineWeb-Edu), one tokenizer, one context length.
- Five seeds: the exact paired sign-flip test cannot reach p < 0.05 (minimum two-sided p = 0.0625).
- fp16 on T4 rather than bf16; DiffAttn costs 1.5x attention FLOPs per token, so the comparison is
  matched in parameters and tokens, not compute.
- The secondary metrics are near chance at this scale and are reported without significance claims.

## 6 Code and logs
All code, plans, per-step JSON logs and the analysis script: https://github.com/RyanLin5967/diffattn.
