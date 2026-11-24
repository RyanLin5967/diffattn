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
