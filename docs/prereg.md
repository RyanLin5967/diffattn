# Pre-registration (written before any sweep or main run; amendments append-only at the bottom)

## Question
Does Differential Attention (DiffAttn) beat a standard GPT on held-out loss, and how much of that gain
does adding QK-norm to the standard GPT recover?

## Arms (matched; see docs/decision-log.md)
`std` (no QK-norm), `qknorm`, `diff` (paper's DiffAttn, no QK-norm). Same data order per seed,
tokenizer, matmul shapes (param count within 0.01%), optimizer (AdamW 0.9/0.95, wd 0.1 on matrices,
eps 1e-8, clip 1.0), schedule (2% linear warmup, cosine to 10% of peak), batch, sequence length and
token budget. FLOPs/token reported for both (DiffAttn attention costs 1.5x).

## Protocol
1. LR sweep: lr in {1e-3, 2e-3, 4e-3} for every arm, seed 100, at 25% of the main token budget with
   the schedule compressed to that length. Best = lowest final full-val loss among runs that finish
   without divergence. If an arm's best LR is at the edge of the grid, that arm (and, to keep tuning
   equal, every arm whose best is at the same edge) gets one more sweep run at the next factor-2 value
   in that direction before main runs start.
2. Main: best LR per arm, seeds 0-4, full budget. Seed sets init and data order; the same seed gives
   the same batch sequence in every arm.
3. Nothing is rerun or excluded after seeing main-run results. A diverged main run is reported as
   such and counted as a failed run for its arm; it is not replaced.

## Metrics and tests
- Primary: final validation loss on the held-out FineWeb-Edu val set (first 10M tokens of
  documents of sample-10BT/000_00000.parquet, disjoint from training), evaluated at the last step.
- Tests, for each pair (std vs diff, std vs qknorm, qknorm vs diff): two-sided Welch t-test and an
  exact paired-by-seed sign-flip permutation test, alpha 0.05. With 5 seeds the permutation test's
  smallest attainable two-sided p is 0.0625, so it cannot reject at 0.05; it is reported anyway.
- Reported: per-arm mean and sd, differences with 95% Welch CI, the gain G = loss(std) - loss(diff),
  the recovery R = (loss(std) - loss(qknorm)) / G with a seed-bootstrap 95% interval, and the minimum
  detectable effect (80% power, two-sided 0.05) from the observed seed sd.
- Secondary: zero-shot HellaSwag (acc_norm, val 10042), ARC-Easy (acc_norm, test 2376), PIQA (acc,
  val 1838), LAMBADA (last-word greedy acc, 5153); attention-noise probe (head-averaged attention mass
  of the final position on the needle's number tokens vs on the rest of the context, per layer,
  200 fixed prompts of ~900 tokens). No significance claims on secondary metrics.

## Budget
Filled in from the T4 pilot measurement before the sweep starts (amendment 1).

## Amendments

### Amendment 1 (2026-11-24, before any GPU run): token budget formula
Main steps S (batch 65,536 tokens) = the largest multiple of 100 such that the slower of the two T4
lanes finishes all 15 main runs within 8.0 hours, using the measured pilot throughput per arm (60
steps, steps 40-60), +4% for periodic small-val evals, plus a measured fixed overhead per run
(startup + compile + final evals). Sweep runs use S // 4 steps. Computed automatically in
`orchestrate.py` stage A and written to `budget.json`; micro-batch 16 per arm, 8 if 16 fails.

### Amendment 2 (2026-11-24, before any GPU run): failed runs and fp16
- fp16 (T4): a step whose loss overflows is skipped by GradScaler and excluded from the logged train
  loss average (count logged as `nonfinite_steps`). A run is DIVERGED if a log window has no finite
  step, the val loss is non-finite, or the loss scale falls below 1.
- Diverged main runs are reported per arm and excluded from the loss tests: Welch uses each arm's
  finished runs; the paired test uses seeds finished in both arms; the bootstrap resamples seeds
  finished in all three arms. They are not rerun or replaced.
- Needle probe cross-arm comparison uses the scale-free answer share |answer|/(|answer|+|noise|),
  because DiffAttn rows sum to 1 - lambda; the whole needle sentence is excluded from noise.
- Recovery R = Q/G is reported with paired-seed bootstrap CIs for G and Q separately; the CI of R is
  conditional on G > 0 and labelled as such, with the fraction of resamples where G > 0.
