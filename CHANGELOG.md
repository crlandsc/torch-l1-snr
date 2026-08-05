# Changelog

## 0.2.0 (unreleased)

### BREAKING CHANGES

Read this section before upgrading. **Users at the default `l1_weight=0.0` are unaffected numerically**: all
four losses return bit-identical values and gradients to 0.1.x on that path. The changes below affect the
blended path, previously-accepted invalid input, and checkpoint keys.

**1. `l1_weight > 0` now scales the L1 term by a fixed reference level.**

The scale was `c * mean_b(1 / (mean|y|_b + eps))`, a mean of reciprocals computed over the batch. That had two
measurable problems. One quiet target inflated the scale for the whole batch, so samples at identical relative
error saw their gradients change because a *different* sample went quiet: up to 4.8x on rows held otherwise
constant. And the knob's meaning drifted with batch content, with the batch-to-batch spread in how far
`l1_weight` moves the loss toward L1 reaching 75 percentage points at `l1_weight=0.5`.

Two new parameters replace it. `ref_level` (default `0.05`) is the typical mean-absolute amplitude of your
targets; `spec_ref_level` defaults to `0.19 * ref_level`, the measured ratio between the normalized-STFT and
time-domain reference magnitudes. Cross-sample contamination is now exactly zero and the spread falls to 4.7
points at `l1_weight=0.1` and 19.7 at `0.5`.

*To migrate:* if you train with `l1_weight > 0`, measure the mean absolute value of your targets over a few
batches and pass it as `ref_level`. The default suits MUSDB-style stems. Being off by 2x shifts the knob about
5 points; being off by 10x matters. Loss values on the blended path will differ from 0.1.x.

**2. Loss buffers no longer appear in `state_dict`.**

`STFTL1SNRDBLoss` held its STFT windows as persistent buffers, so holding the loss as a model submodule added
3 keys and 3584 floats to every checkpoint, and changing `n_ffts` made earlier checkpoints fail to load.

*To migrate:* a checkpoint written by 0.1.x contains `...spectrogram_transforms.N.window` keys that 0.2.0 does
not expect. Load with `strict=False`, or filter those keys out first.

**3. Previously-accepted invalid input now raises `ValueError`.**

- `spec_weight` outside `[0, 1]`. Above 1 the time-domain coefficient `(1 - spec_weight)` went negative, which
  instructed the optimizer to *maximize* waveform error. This was accepted silently and the docstring invited
  it by saying "set higher to emphasize spectral accuracy" with no stated bound.
- `l1_weight` outside `[0, 1]` on `L1SNRLoss`, the only class that had not validated it. `-0.5` silently took
  the pure-SNR branch and `2.0` the pure-L1 branch.
- `estimates` and `actuals` of differing shape. These were previously reconciled by flattening, so
  `(2, 4, 8000)` against `(2, 2, 16000)` returned a plausible number from a wrong pairing.
- An unrecognized `window_fn`, which previously surfaced as an `AttributeError` about a mangled attribute name.
- A non-positive `ref_level`.

Validation now raises `ValueError` rather than using bare `assert`, which `python -O` strips. Under 0.1.x with
`-O`, out-of-range values produced a silently wrong loss.

**4. `l1_weight` is read-only after construction.**

It is baked into child modules and mode flags when the loss is built, so assigning to it changed the number
without changing the behaviour it had already determined. Construct a new loss instead.

**5. Short-audio and dtype behaviour changed.**

- `STFTL1SNRDBLoss(use_regularization=True)` below `min_audio_length` previously dropped the regularizer
  silently. A total collapse then scored exactly `0.000000`, which is the failure the regularizer exists to
  prevent. It is now applied, at a coefficient matched to `spec_reg_coef` so its weight is continuous across
  the boundary.
- Output dtype follows input dtype. `STFTL1SNRDBLoss` previously downcast float64 to float32 while its
  siblings did not.
- Inputs of 512-1024 samples now warn about which STFT resolutions were dropped. The resolutions used are
  unchanged; the arity of the multi-resolution average is simply no longer silent.

**Not breaking, but stated for completeness:** `dbrms` applied a second, dimensionally inconsistent epsilon
outside the square root. It could never prevent a log of zero and is removed. Measured effect below 0.001 dB
across levels from silence to 10.0; the -80 dB silence floor is preserved exactly.

### Performance

`STFTL1SNRDBLoss` and `MultiL1SNRDBLoss` gain `check_finite` (default `True`, preserving current behaviour).
The non-finite input scan costs four full-tensor passes whose results are consumed by a Python `if`, which on
CUDA forces a host-device synchronization and serializes the pipeline. Setting it `False` saves about 1% of a
CPU forward and removes those synchronizations; the loss then propagates `NaN` rather than replacing it with
zeros, which is arguably preferable during training anyway.

**The spectrogram loss is about 14% faster.** The STFT window normalization is folded into the window itself
rather than dividing the whole complex output by the same constant on every call. This is mathematically
identical, since the transform is linear in the window, and it removes one full-tensor pass per resolution
per tensor. Measured +14.2% with a standard deviation of 3.2% across interleaved A/B trials on
`[8, 2, 264600]`, against a noise floor of 0.5%.

Because the multiply moves inside the transform, float32 results can differ in the last bit: measured
relative difference 1.6e-07 on gradients, at float32 machine epsilon, with loss values unchanged. Float64
accuracy slightly **improves**, from 6.2e-10 to 5.2e-10 relative error against a fully-float64 reference,
because the normalization constant is now computed in double precision. If you are comparing loss values
across this upgrade to more than six significant figures, this is why they differ.

Two further optimizations were investigated and did not ship, recorded because the reasons are useful:

- Reducing over dimensions instead of reshaping the strided real and imaginary views avoids roughly 129 MB of
  contiguous copies per forward. The forward is bit-identical, but the backward graph is not, and the
  measured time saving was 0.06%. Not worth a change to gradients for no measurable gain.
- Sharing one STFT between the reconstruction and regularizer paths: **there was nothing to share.** A call
  counter shows six transform calls with the regularizer enabled and six with it disabled, so the transform
  was never recomputed. The premise was wrong.

The per-call device move on the spectrogram transforms is now guarded rather than unconditional, worth about
0.02 ms.

### Other changes

- Silent degradations are now audible. One-shot warnings per loss instance when non-finite input is sanitized
  (a diverged run previously reported a healthy `0.0`), when every STFT resolution fails, and when the
  spectrogram branch falls back to the time domain inside `MultiL1SNRDBLoss`, which makes both of its branches
  the same quantity.
- The all-resolutions-failed path returns a graph-connected zero. Previously `.backward()` raised, and inside
  `MultiL1SNRDBLoss` the detached zero silently contributed nothing while the time term went on training.
- `README` documents what `l1_weight` actually delivers, measured, instead of stating it as a proportion.
- Mutable list defaults (`n_ffts`, `hop_lengths`, `win_lengths`) are copied per instance.
- Dead code removed: an unreachable pure-L1 branch, a redundant shape reconciliation, and a write-only
  counter superseded by the warning above.


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
  wrong above an input length of 65,536 samples (2^16), in a pattern that is not monotone in size. The fix was and remains effective; only the
  explanation was wrong. This matters because the README invites users to disable the workaround once
  upstream is fixed, and they would have been checking the wrong thing.
- **Corrected the D1 notation.** Docstrings rendered D1 with a summed L1 norm where the code takes a mean.
  The code is right (it matches the authors' reference implementation); the notation was not. Reading the
  old formula literally would predict values differing by about 1.5 dB.
- **Scoped the gradient-balancing claim.** The README said the L1 component is scaled to "approximately
  match gradient magnitudes". It is comparable near 0 dB SNR only: the L1 term's gradient is independent of
  the error while L1SNR's grows as the error shrinks, so the ratio widens as the model improves. The README
  now states measured figures for a stated configuration rather than as universal constants, since they
  depend on target level. The scaling's real purpose is to keep the two components' gradient profiles
  distinct.
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

**Test suite rebuilt.** Not shipped in the wheel, so this changes nothing for users, but it is the reason to
trust the corrections above. 112 of the previous 127 test cases compared a wrapper against the function it
wraps, which is structurally `f(x) == f(x)` and cannot fail for any implementation. Stubbing all four
`forward` methods to a constant left 122 of 127 passing, and a byte-exact revert of the v0.1.2 bugfix left
all 127 green.

Tests now compare against an independent reference derived from the published definitions and validated
against hand-computed values. 259 tests, coverage 79% to 90%, and CI enforces both a mutation gate (every
test that exercises a `forward` must fail when it returns a constant) and a coverage floor. Reverting the
v0.1.2 fix now breaks 50 tests. The MPS tests were rewritten to use an input size where the underlying
PyTorch bug actually manifests; the previous ones compared CPU against CPU and would have passed regardless.

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

PyTorch's MPS backend produces numerically incorrect gradients from `torch.stft` backward on longer
inputs, causing `STFTL1SNRDBLoss` and `MultiL1SNRDBLoss` to corrupt model weights within a few training
steps. The forward transform is correct; only the
backward pass is affected, above an input length of 65,536 samples (2^16).

**Fix:** Added `mps_cpu_fallback` parameter (default `True`) to `STFTL1SNRDBLoss` and `MultiL1SNRDBLoss`.
When on MPS, STFT loss computation is automatically routed through CPU via differentiable `.cpu()` calls,
producing correct gradients. The scalar loss is moved back to MPS for the optimizer step.

- **MPS users:** Losses now work correctly out of the box. No code changes needed.
- **CUDA/CPU users:** Completely unaffected. The fallback is a no-op when not on MPS.
- **Time-domain losses** (`L1SNRLoss`, `L1SNRDBLoss`): Unaffected, they never had this issue.
- Set `mps_cpu_fallback=False` to disable the workaround if a future PyTorch release fixes the MPS bug.

> **Note added in 0.1.4:** this entry originally attributed the bug to `torch.abs()` backward on complex
> tensors. That was incorrect; the cause is `torch.stft` backward above an input length of 65,536 samples.
> See the 0.1.4 entry. The fix itself was effective. The gradient error direction also differs by PyTorch
> version: this entry reported inflation, while torch 2.10 shows deflation. The magnitude quoted originally
> is not reproducible on current versions and has been removed.

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
