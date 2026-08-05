# Changelog

## 0.1.4 (unreleased)

Documentation accuracy and packaging hygiene. **No behavioral change**: no loss value, gradient, or API
signature changes in this release, apart from `MultiL1SNRDBLoss` gaining a `spec_reg_coef` pass-through
whose default matches the value already used internally.

**Documentation corrections.** Several documented claims did not match the implementation:

- **All README examples now run.** Four of the five runnable examples raised
  `RuntimeError: element 0 of tensors does not require grad` on `loss.backward()`, because the example
  tensors were built without `requires_grad=True`. A fifth omitted `.backward()` entirely. Every example is
  now executed by the test suite so they cannot regress.
- **Corrected the v0.1.3 MPS root cause.** The previous entry (below) attributed the incorrect gradients to
  `torch.abs()` backward on complex tensors. That path does not execute in the default configuration: the
  default splits into real and imaginary parts before calling `torch.abs`, so `torch.abs` never receives a
  complex tensor unless `use_regularization=True`. The actual cause is `torch.stft` backward on MPS, which is
  wrong above roughly 65,000 samples and worsens with input size. The fix was and remains effective; only the
  explanation was wrong. This matters because the README invites users to disable the workaround once
  upstream is fixed, and they would have been checking the wrong thing.
- **Corrected the D1 notation.** Docstrings rendered D1 with a summed L1 norm where the code takes a mean.
  The code is right (it matches the authors' reference implementation); the notation was not. Reading the
  old formula literally would predict values differing by about 1.5 dB.
- **Scoped the gradient-balancing claim.** The README said the L1 component is scaled to "approximately
  match gradient magnitudes". That holds near 0 dB SNR only; the ratio is roughly 1:1 at 0 dB, 9:1 at 20 dB
  and 75:1 at 60 dB. The scaling's real purpose is to keep the two components' gradient profiles distinct.
- **Scoped the NaN/Inf robustness claim** to `STFTL1SNRDBLoss` and `MultiL1SNRDBLoss`. The time-domain
  losses do not sanitize non-finite input, and a NaN estimate yields a NaN loss.
- **Corrected the short-audio docstring**, which claimed zero loss is returned. A time-domain fallback is
  returned instead, and it is a different objective: roughly a 2x shift in value across the boundary,
  because the STFT path sums a real and an imaginary term.
- **Removed a false novelty claim.** Time versus spectrogram loss balancing was listed as a novel extension,
  but the authors' own `bandit` exposes equivalent `time_weight`/`freq_weight` controls.
- **Marked the `l1_weight` "smearing" passage as anecdote**, which the text itself already stated.
- **Documented the regularizer constants as local choices.** `lambda0`, `delta_lambda` and `lmin` appeared
  under a citation to arXiv:2501.16171, but that paper specifies no numeric values for them and no official
  implementation of it exists. The citation covers the form only.
- **Documented `spec_weight` precisely.** It is a loss-value weight, exactly as the coefficients say. But
  `spec_loss` sums a real and an imaginary D1 term while `time_loss` is a single term, so equal coefficients
  do not mean equal gradient contribution. The default `0.5` is the value at which the time, real and
  imaginary terms all carry weight 0.5, reproducing the equal 1:1:1 weighting of arXiv:2406.18747.
- **Documented the quiet-target dynamic-range collapse.** D1's floor at perfect reconstruction is
  `10*log10(eps / (mean|y| + eps))`, so a target near -58 dBFS RMS has under 3 dB of usable loss range.
- **Documented `dbrms`**, which was exported in `__all__` but appeared nowhere in the README.
- **Documented 10 constructor parameters** that were missing from their class docstrings, including
  `spec_reg_coef`, which was also unreachable from `MultiL1SNRDBLoss` except through `spec_loss_params`.
- **Added attribution** for the authors' reference implementations, `kwatcharasupat/bandit` (Apache-2.0) and
  `kwatcharasupat/query-bandit` (MIT, Copyright (c) 2024 Karn Watcharasupat), alongside the papers.
- **Fixed the citation for arXiv:2309.02539**, published in IEEE Open Journal of Signal Processing vol. 5,
  pp. 73-81, **2024**, doi:10.1109/OJSP.2023.3339428 (previously cited as 2023, with no volume or DOI).
- **Unswapped references [1] and [2]**, which disagreed between the source header and the README.
- **Completed this changelog.** Versions 0.0.1 through 0.0.5 were absent, and 0.1.0 was labelled the initial
  release with the wrong date.

**Packaging.**

- Removed `numpy` from the dependencies. It was declared and documented but never imported.
- Added `py.typed` so type hints reach downstream checkers, and corrected `Optional[dict]` annotations.
- Stripped authoring metadata from `images/logo.png`, which carried `pdf:Author` and Canva document, user
  and brand identifiers in its PNG text chunks.
- Unified the copyright identifier across `LICENSE`, the source header and `setup.cfg`.
- Corrected `python_requires` and the version classifiers to a tested floor.
- CI now runs on pull requests and pushes to `main`, not only on release tags, and the release workflow
  fails if the tag does not match `__version__`.

## 0.1.3 (2026-02-24)

**Fix:** MPS (Apple Silicon) compatibility for STFT-based losses.

PyTorch's MPS backend produces numerically incorrect (inflated) gradients from `torch.stft` backward on
longer inputs, causing `STFTL1SNRDBLoss` and `MultiL1SNRDBLoss` to produce gradient norms far too large on
MPS and corrupt model weights within a few training steps. The forward transform is correct; only the
backward pass is affected, above roughly 65,000 samples.

**Fix:** Added `mps_cpu_fallback` parameter (default `True`) to `STFTL1SNRDBLoss` and `MultiL1SNRDBLoss`.
When on MPS, STFT loss computation is automatically routed through CPU via differentiable `.cpu()` calls,
producing correct gradients. The scalar loss is moved back to MPS for the optimizer step.

- **MPS users:** Losses now work correctly out of the box. No code changes needed.
- **CUDA/CPU users:** Completely unaffected. The fallback is a no-op when not on MPS.
- **Time-domain losses** (`L1SNRLoss`, `L1SNRDBLoss`): Unaffected, they never had this issue.
- Set `mps_cpu_fallback=False` to disable the workaround if a future PyTorch release fixes the MPS bug.

> **Note added in 0.1.4:** this entry originally attributed the bug to `torch.abs()` backward on complex
> tensors. That was incorrect; the cause is `torch.stft` backward above roughly 65,000 samples. See the
> 0.1.4 entry. The fix itself was effective.

## 0.1.2 (2026-02-04)

**Bugfix:** Fixed gradient equivalence in L1/L1SNR auto-balancing.

The `l1_weight` parameter (for mixing L1 and L1SNR losses) was improperly applied due to scaling by error
magnitude, which made L1 gradients identical to L1SNR gradients. Now scales by reference signal magnitude,
preserving distinct gradient behaviors:
- L1SNR: inverse-error-scaled gradients (larger updates for smaller errors)
- L1: uniform gradients regardless of error magnitude

Credit: [Karn N. Watcharasupat](https://github.com/kwatcharasupat) for identifying the issue.

## 0.1.1 (2026-01-30)

Improved README documentation with better context and clarity.

## 0.1.0 (2026-01-30)

First release of the current four-class API (`L1SNRLoss`, `L1SNRDBLoss`, `STFTL1SNRDBLoss`,
`MultiL1SNRDBLoss`).

## 0.0.5 (2026-01-29)

Pre-release iteration.

## 0.0.4 (2026-01-17)

Pre-release iteration.

## 0.0.3 (2026-01-17)

Pre-release iteration.

## 0.0.2 (2026-01-17)

Pre-release iteration.

## 0.0.1 (2026-01-17)

Initial upload to PyPI.
