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
- Stage A logic test (fake lane runner, planted sweep optima, simulated OOM at micro_bs 16 for std):
  micro_bs fallback to 8 worked, budget filled exactly 8.0 h for the slower lane, the edge rule
  extended std down (5e-4) and diff up (8e-3), and LR selection returned the planted optima. The CPU
  dry run with real tiny training completed the pilot stage (all three arms reached step 60, rc 0).

## 2026-11-24 — stage A result (raw outputs in runs/stageA/)
- Pilot on T4 (fp16, micro_bs 16 fit for all arms): std 45.3k tok/s (MFU 25%), qknorm 40.8k (23%),
  diff 38.3k (23%). Fixed overhead per run 518 s + evals 360 s.
- Pre-registered formula -> main_steps 1800 (117.96M tokens/run), sweep 450 steps; predicted slower
  lane 7.7 h for all 15 main runs on one session.
- Sweep (seed 100, 450 steps, final full val loss): no fp16 overflow skips in any run, loss scale
  constant. Lower LRs won in every arm, so the pre-registered edge rule added 5e-4 for all three arms.
  Best: std 1e-3 (5.308), qknorm 5e-4 (5.457), diff 1e-3 (5.718). qknorm's grid is non-monotonic
  (1e-3 5.475, 2e-3 5.667, 4e-3 5.519): single-seed sweep noise, reported as a limitation; the
  edge rule allows one extension only and qknorm's best (5e-4) is again on the edge.
- Main runs split into two Kaggle sessions (8 + 7 runs) to halve wall-clock; same steps and LRs.
  Kaggle allows 2 concurrent GPU sessions per account and another project's kernels held both slots,
  so the pushes are queued and retried until a slot frees. The weekly free GPU quota is shared with
  that project (remaining quota is not visible through the API).

## 2026-11-24 — main results (raw: runs/main/, analysis: analysis/results.json)
- All 15 main runs finished 1800 steps, none diverged, no fp16 skipped steps, all evals rc 0.
- std 4.0787±0.0096 < diff 4.1677±0.0242 < qknorm 4.2231±0.0081 (final full val). DiffAttn gain over
  std is negative (-0.089, Welch p=5e-4), so the pre-registered recovery ratio is not meaningful.
- The spec's target resume line ("QK-norm recovers X% of DiffAttn's gain") is not supported at this
  scale. Supported line (measured): "Ran a matched, pre-registered 3-arm x 5-seed replication of
  Differential Attention on a 51M-param GPT (118M tokens/run, free Kaggle T4s): at this scale the
  standard baseline beat DiffAttn by 0.089 val loss (p=5e-4) and QK-norm by 0.144, while DiffAttn
  still tripled attention share on the answer span in a retrieval probe."
- Figure fix: the first curves.png joined two different val sets (2M prefix and full 10M) in one line,
  producing fake upticks; now plotted separately.
- Fresh-context verification (own script, not analyze.py): all loss means/sds, Welch diffs/CIs/p,
  sign-flip p, matching, needle shares CONFIRMED. Narrowed the claim: results hold for sweep-chosen LRs
  only; sweep LR sensitivity (+0.13 to +0.30 per 2x for std/diff) and single-seed sweep noise (~0.15 for
  qknorm) are as large as the effects; std-diff gap shrank 0.114 -> 0.089 from step 1000 to 1800.
- Resume line, narrowed accordingly: "Ran a matched, pre-registered 3-arm x 5-seed replication of
  Differential Attention (51M-param GPT, 118M tokens/run, free Kaggle T4s, $0): with sweep-tuned LRs the
  standard baseline beat DiffAttn by 0.089 val loss (p=5e-4, all 5 seeds) and QK-norm by 0.144; DiffAttn
  still tripled attention share on the answer span in a retrieval probe."
- Not run: a full-budget LR check (2 seeds of qknorm/diff at extra LRs) suggested by the verifier. It is
  outside the pre-registered protocol and would cost another multi-hour Kaggle session; the paper states
  the open question instead.
