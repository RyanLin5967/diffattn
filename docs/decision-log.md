# Decision log (append-only; newest last)

Goal: test whether Differential Attention (Ye et al. 2024) still beats a standard GPT once the baseline uses QK-norm, with a matched, pre-registered, multi-seed protocol.

## 2026-11-24 — scale and compute
- Compute budget for this project is $0: all GPU work runs on Kaggle's free T4 x2 sessions and data
  prep on a Kaggle CPU session. `modal_app.py` is an unused alternative entrypoint; nothing runs it.
- The spec's setup (350M params x 10B tokens x 15 runs) is ~270 H100-hours by the 6ND estimate
  (unverified, no pilot at that scale), far outside a $0 budget.
- Scale-down that keeps the protocol intact: ~50M-param GPT (d=512, 8 layers, 8 heads, SwiGLU 1408,
  GPT-2 BPE, seq 1024), all three arms, 3-LR sweep per arm, 5 seeds per arm at full budget,
  pre-registered tests, unit tests, zero-shot evals, attention-noise probe. The per-run token budget
  is set from the measured T4 pilot so the whole study fits the free Kaggle GPU quota.
- Data: one FineWeb-Edu sample-10BT parquet file (000_00000), GPT-2 BPE; first 10M tokens of documents
  held out as val (disjoint from train). The pool is larger than any run's budget, so each seed sees a
  different random subset of windows; the same seed sees the identical batch sequence in every arm.

## 2026-11-24 — DiffAttn arm has no QK-norm
- Spec says the DiffAttn arm is "identical [to the baseline], except attention is DiffAttn per the
  paper". The paper's DiffAttn has no QK-norm, and the question is whether QK-norm alone recovers
  DiffAttn's gain over the standard baseline, so the DiffAttn arm uses the paper's attention (no
  QK-norm). Everything else (RoPE, SwiGLU, RMSNorm, tied embeddings, no biases, init, optimizer,
  schedule, data order, batch, tokens) is identical across arms.
- Param matching: DiffAttn uses n_head/2 heads with head_dim d/n_head per q/k half, so the q,k,v,o
  matmuls have exactly the baseline's shapes. Extra params per layer: 4 lambda vectors (4*64) + subln
  (128); QK-norm arm: 2*64. Arms differ by <0.01% in parameter count.
- DiffAttn computes both softmax maps in ONE fused SDPA call over 4H heads ([q1,q1,q2,q2] x
  [k1,k1,k2,k2] x [va,vb,va,vb]) because fused kernels need equal q/k and v head dims. Model
  FLOPs/token (reported): attention 1.5x baseline; hardware FLOPs actually executed: attention 2x.

## 2026-11-24 — validation cadence
- Spec: val loss on a fixed 20M-token set every 100 steps. At this scale that costs a large fraction
  of training compute. Implemented: full fixed val set (10M tokens, not 20M: one parquet file's first
  documents) at the final step (primary metric) and every `val_full_every` steps; a fixed 2M-token
  prefix of the same val set at other log steps.

## 2026-11-24 — statistics note
- The paired-by-seed sign-flip permutation test with n=5 pairs has only 2^5 = 32 sign patterns; the
  smallest attainable two-sided p is 2/32 = 0.0625 > 0.05, so it can never reject at alpha 0.05 with
  5 seeds. Kept (the spec requires it) and reported with this caveat; Welch's t-test is the test that
  can reject.

## 2026-11-24 — bugs caught before any GPU time
- CPU smoke: `F.cross_entropy(... targets.view(-1))` crashed on the non-contiguous target slice -> reshape.
- Unit tests: RMSNorm/lambda forced float32, breaking float64 gradcheck -> promote to >= float32.
- Mutation check: 4 hand-made DiffAttn-core mutations (lambda scale, swapped a2 halves, swapped v
  halves, non-causal) were each caught by >= 2 unit tests.
- CPU smoke: a 20-step run paused (checkpoint + PAUSED) and resumed to 40 steps reproduced the
  uninterrupted 40-step run bit-for-bit (train loss, val loss, lr, lambdas at every log step).
- CPU smoke: evals.py crashed when the needle prompt (900 tokens) exceeded a 64-token smoke model's
  context -> probe length = min(900, 0.88 * seq_len); the real runs (seq 1024) use 900.
- The first attempt at the analysis fixes silently did not apply (a preceding `grep -c` returning 0
  short-circuited the `&&` chain) and the old analyze.py was committed. Caught by a synthetic test:
  15 fake runs with a planted effect (G=0.05, R=0.8, seed sd 0.005) and one DIVERGED std run. After
  re-applying: G=0.055 [0.050, 0.062], R=0.73 [0.62, 0.81] (covers 0.8), diverged run excluded and
  listed, paired test on the 4 common seeds (16 sign patterns).
