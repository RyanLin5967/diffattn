# Does QK-norm close the gap to Differential Attention? A matched, pre-registered small-scale replication

*Workshop draft. Every number in this paper comes from logged JSON in the released runs; sections marked
[RESULTS] are filled only from `analysis/results.json`.*

## Abstract
[RESULTS]

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
[RESULTS] Table 1: final val loss mean +/- sd per arm, differences with 95% CI, both p-values, MDE.
[RESULTS] Gain G = L(std) - L(diff); QK-norm recovery R = (L(std) - L(qknorm)) / G with bootstrap CI.
[RESULTS] Figure 1: val-loss curves with seed bands; lambda per layer over training.
[RESULTS] Table 2: zero-shot accuracies.

## 4 Mechanism check: attention noise
[RESULTS] Answer share of the final position's attention on a key-retrieval prompt, per layer and arm.

## 5 Limitations
- One scale (~51M parameters) and a short token budget set by free compute, far below the 350M/10B-token
  design this study was originally specified at; effects at this scale may not transfer.
- One dataset (FineWeb-Edu), one tokenizer, one context length.
- Five seeds: the exact paired sign-flip test cannot reach p < 0.05 (minimum two-sided p = 0.0625).
- fp16 on T4 rather than bf16; DiffAttn costs 1.5x attention FLOPs per token, so the comparison is
  matched in parameters and tokens, not compute.
- The secondary metrics are near chance at this scale and are reported without significance claims.

## 6 Code and logs
All code, plans, per-step JSON logs and the analysis script: https://github.com/RyanLin5967/diffattn.
