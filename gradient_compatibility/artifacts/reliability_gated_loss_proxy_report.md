# Reliability-gated post-warmup loss proxy

## Frozen rule

1. Compute the median target-gradient split-half cosine for every probe domain.
2. Keep domains with median split-half cosine greater than `0.2`.
3. Evaluate correct-image validation loss after projector warmup.
4. Rank tokenizers within each retained domain (lower loss is better) and average
   the domain ranks with equal weight.

The reliability gate and rank aggregation do not use MLLM labels. Both calibration
cohorts retain `caption` and `reasoning`; `vqa` and `ocr` are rejected because their
mean-gradient directions are not reproducible.

This rule is frozen and is what `summarize_full_sweep` bets on the 79-tokenizer
sweep. It is recorded here as a pre-registered bet, **not** as a validated metric.
The calibration evidence below does not support the latter.

## Full 256-example results at 4096 warmup examples

| Cohort | Tokenizer | Caption loss | Reasoning loss | Mean rank | Qwen2.5 MLLM |
|---|---|---:|---:|---:|---:|
| PE pair | pe_lang_g14_448 | 2.3936 | 0.9961 | 1.0 | 58.03 |
| PE pair | pe_core_g14_448 | 4.0753 | 7.8262 | 2.0 | 51.88 |
| Three-tokenizer | siglip2_b16_256 | 2.2251 | 0.9023 | 1.5 | 47.62 |
| Three-tokenizer | mc1_b16_224_2.5b | 2.4752 | 0.8924 | 2.0 | 44.74 |
| Three-tokenizer | raev2 | 2.3279 | 1.2802 | 2.5 | 36.26 |

Both cohort orders match the Qwen2.5 order (Spearman `1.0` each). The four findings
below explain why that agreement is not yet evidence.

## Why the calibration agreement is not evidence

### 1. `pe_core_g14_448` did not lose the PE pair, it failed to train

At 4096 warmup examples its caption loss is `4.0753` against a no-image baseline of
`3.6564`, and its reasoning loss is `7.8262` against a baseline of `1.1596`
(`real_minus_zero = +6.67`). Giving the model the image makes it 6.7 nats worse than
giving it nothing. The failure is already present at 256 warmup examples, where
`real_minus_shuffled = -0.0003`: the run never separated the correct image from a
deranged one.

That is a diverged projector, not an interface-compatibility gap. The PE pair
therefore tests nothing; a broken run happened to sort last.

### 2. The `reasoning` domain carries almost no visual signal, yet it casts half the vote

Correct-versus-deranged gaps on ScienceQA-IMG, at 4096 warmup examples:

| Tokenizer | reasoning `real_minus_shuffled` |
|---|---:|
| siglip2_b16_256 | -0.0219 |
| mc1_b16_224_2.5b | **+0.0084** |
| raev2 | -0.0260 |

`mc1_b16_224_2.5b` *won* the reasoning domain (loss `0.8924`) while being the one
tokenizer whose correct image is worse than a deranged image. Across the first five
sweep models the median `|real_minus_shuffled|` on reasoning is `0.0156` while the
between-tokenizer loss spread is `0.9730` — the ranking signal is 60x larger than the
visual effect it is supposed to measure, so this domain ranks something other than
the image.

The split-half gate did not catch this because it measures gradient-direction
reproducibility, not visual dependence. **The gate selects on the wrong quantity.**

### 3. The deciding margin is smaller than the protocol's own run-to-run variance

The trio order is decided by `0.8924` versus `0.9023` — `0.0099` nats.

For `pe_lang_g14_448` caption, under identical warmup budget, learning rate, seed and
(confirmed by a matching `zero` baseline of `3.6564050` / `3.6564341`) an identical
evaluation set, the pilot and the sweep disagree: `real` is `2.3936` versus `2.3001`
and `shuffled` is `3.0877` versus `3.2777`. That is `0.09`-`0.19` nats of variance
from configuration-order effects alone, an order of magnitude above the margin being
used to rank.

### 4. The raw `real` loss carries a sequence-length term

The no-image baseline is not tokenizer-independent, because the zero prefix keeps the
tokenizer's own length: `siglip2_b16_256` and `raev2` (256 tokens) both give
`3.7686`, `mc1_b16_224_2.5b` (196 tokens) gives `3.7039`, `siglip2_sm14_384` gives
`3.6527`. That is up to `0.116` nats of drift driven only by prefix length — again
larger than the deciding margins.

The frozen rule ranks `real`, which includes this term. `real_minus_zero` cancels it
and is the quantity the diagnostics prefer.

### Combined

Taken entirely at face value, matching a 2-way and a 3-way order is `1/2 x 1/6 = 1/12`
(`p ~ 0.083`) — not significant before any of the above. Finding 1 removes the 2-way
result outright.

## Scope

Five tokenizer instances, seed 0, one diverged run, and two orders decided inside the
protocol's own noise floor. The frozen rule should be carried to the 79-tokenizer
sweep because a pre-registered bet is worth settling, but it should not be described
as validated, and no further metric tuning should happen on these five models.

Post-hoc diagnostics for the sweep are in `gradient_compatibility/analyze_full_sweep.py`
and are written to `analysis/diagnostics.json`. They never modify the blind
`summary/predictions.json`.
