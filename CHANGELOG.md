# Changelog

## 0.2.0 (2026-08-31)

### BREAKING CHANGES

Read this section before upgrading.

**What happens at the default `l1_weight=0.0`.** Two losses are bit-identical to 0.1.x and two are not:

| loss | loss value | gradients |
|---|---|---|
| `L1SNRLoss` | bit-identical | bit-identical |
| `L1SNRDBLoss(use_regularization=False)` | bit-identical | bit-identical |
| `L1SNRDBLoss(use_regularization=True)` | differs, up to **4e-05** relative | differs |
| `STFTL1SNRDBLoss(use_regularization=False)` | differs, up to 1e-07 relative | differs |
| `STFTL1SNRDBLoss(use_regularization=True)`, `MultiL1SNRDBLoss` | differs, up to **4e-05** relative | differs |

Two changes cause it, and they are of different sizes. The window-normalization fold moves values at float32
rounding, around 1e-07 relative. The `dbrms` epsilon cleanup is larger and is **not** a rounding effect: it
changes the level of a near-silent signal by up to 8e-04 dB, and the regularizer's `R = |L_pred - L_true|`
term amplifies that when either level sits near the -80 dB floor. So it appears in float64 as well as
float32, and the worst case is a **silent estimate against a quiet target**, which is exactly the collapse
case the regularizer exists to detect and the case you are most likely to construct while testing.

If you pin a loss value in a regression test, expect to update it for any configuration with regularization
enabled, or for either spectrogram class.

Gradients move considerably more, and by an amount that depends both on how converged your estimate is and
on how large your tensors are. Relative L2 difference:

| relative error of the estimate | small tensors (< 1e5 elements) | realistic training shapes |
|---|---|---|
| 10% | 1e-07 | **5e-04** |
| 1% | 1e-03 | 1e-03 |
| 0.1% | 5e-03 | 5e-03 |
| 0.001% | 5e-02 | 5e-02 |

The mechanism is not instability introduced here. `torch.abs` has a discontinuous subgradient at zero, so
when a residual bin sits near zero a last-bit change flips its sign and moves that element's gradient by its
full magnitude. More elements means more bins near zero, which is why the effect grows with tensor size.
Perturbing 0.1.x's own window by a single floating-point step produces a difference of the same order, so
this is a property of an L1 objective rather than of this release, and it is far below the gradient noise of
minibatch training.

**If you are diffing to validate the upgrade, use a relative tolerance, and prefer loss values to
gradients.** Loss values differ by at most 4e-05 relative, so `rtol=1e-4` passes on every path. Gradients
differ by ~5e-04 relative at a realistic batch shape even at 10% error, so a strict gradient comparison will
fail and tell you nothing useful.

The remaining changes below affect the blended path, previously-accepted invalid input, and checkpoint keys.

**Positional constructor calls written against 0.1.3 keep their meaning.** Every parameter 0.1.3 shipped is
in its original position on all four classes, and the parameters new in 0.2.0 come after them, so 0.1.3's
order is a strict prefix. Two tests pin this: one compares each signature against the 0.1.3 source in git,
and one constructs `MultiL1SNRDBLoss` with all 20 of its 0.1.3 arguments positionally and checks the
override still lands. Use keyword arguments anyway -- these constructors take 23 parameters.

**1. `l1_weight > 0` now scales the L1 term by a fixed reference level.**

The scale was `c * mean_b(1 / (mean|y|_b + eps))`, a mean of reciprocals computed over the batch. That had two
measurable problems. One quiet target inflated the scale for the whole batch, so samples at identical relative
error saw their gradients change because a *different* sample went quiet. How far depends on the batch: 2.1x
with one quiet row in four at `l1_weight=0.5`, 4.8x with seven quiet rows in eight, and over 30x at
`l1_weight=0.9` with a loud held row. Second, the knob's meaning drifted with batch content: the
batch-to-batch spread in how far `l1_weight` moves the loss toward L1 reached 75 percentage points at
`l1_weight=0.5`.

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
- **Input with fewer than 2 dimensions.** A bare `[time]` tensor was read as T batch rows of one sample
  each, returning a mean of per-sample D1s: measured 0.16 to 1.9 dB optimistic across relative errors from
  0.5 to 0.001, always in the flattering direction and always in a believable dB range.
- **Input with any zero-size dimension.** An empty batch already raised, with a cryptic reshape error; a
  zero-size non-batch dimension reached `torch.mean` over an empty reduction and returned NaN, and
  `[B, 0, T]` -- an empty stem selection -- gave NaN from the time-domain classes but 0.0 from the
  spectrogram one. All four classes now raise the same `ValueError`, naming the tensor and its shape.
- **Non-floating-point input.** int16 PCM without an explicit conversion made `torch.stft` raise for every
  resolution; the per-resolution handler swallowed it and the spectral loss became a permanent 0.0 after one
  warning. The time-domain classes raised on the same input, so the four disagreed.
- **A negative `weight`, `lambda0`, `delta_lambda`, `reg_coef` or `spec_reg_coef`, and a non-positive
  `l1snr_eps` or `dbrms_eps`.** `spec_weight` was confined to [0, 1] precisely because a negative
  coefficient instructs the optimizer to maximize what it scales; nothing else was checked, so
  `weight=-1.0` negated the whole objective and `spec_reg_coef=-5.0` turned anti-collapse into a reward for
  collapsing. Zero remains legal for the coefficients, since that is how a term is switched off.
- **Malformed STFT parameters:** a non-positive `n_fft`, `hop_length` or `win_length`, a non-list passed
  where a list of resolutions is expected, a non-int entry, or an empty resolution list. `hop_lengths=[0]`
  previously constructed without complaint and then raised `ZeroDivisionError` on the first forward,
  mid-training; a bare `n_ffts=512` gave "object of type 'int' has no len()"; a float from a config file
  failed inside the window function; and empty lists were accepted, silently making the spectrogram branch a
  permanent time-domain fallback.

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
- **If you lower `min_audio_length` below 512, inputs of 257-511 samples now compute a real STFT loss where
  they previously fell back to the time domain.** The old length check rejected an input whenever *any*
  configured hop produced fewer than two frames, which below 512 samples was always true, so the fallback was
  unreachable-by-design rather than deliberate. Per-resolution checking now admits the resolutions that
  genuinely fit. The value changes by about 90% across that band (measured -9.12 to -17.14 at 300 samples),
  because the objective changes from one time-domain D1 to a one-resolution STFT loss. At the default
  `min_audio_length=512` nothing in this band is reachable and nothing changes.

**`dbrms` lost a redundant epsilon.** It applied a second, dimensionally inconsistent epsilon outside the
square root, on an amplitude, where the first already sits inside on a power. The outer one could never
prevent a log of zero and is removed. The silence floor becomes exactly -80 dB rather than -79.99913.

At the default `dbrms_eps=1e-8` the effect on a level is below 0.001 dB across target levels from silence to
10.0. **That bound is a property of the default, not of the change**: `dbrms_eps` is a documented constructor
parameter, and at 1e-4 the shift is 0.017 dB and at 1e-2 it is 0.74 dB. If you have raised `dbrms_eps`, treat
this as a breaking change rather than a rounding one.

Note also that this is the larger of the two contributors to the loss-value differences in the compatibility
table above, because the regularizer amplifies a level shift when a level sits near the floor.

### Added

- **`L2SNRLoss`** (experimental, opt-in): an energy-ratio time-domain loss, a sibling of `L1SNRLoss`. Nothing selects it by default and no existing behaviour changes. See its docstring for details.
- **`MultiL1SNRDBLoss(time_loss_module=...)`** (opt-in): inject a pre-built module in place of the built-in time-domain branch. Default `None` keeps prior behaviour, and the parameter is appended after `check_finite` so positional calls written against 0.1.3 keep their meaning.

### Performance

`STFTL1SNRDBLoss` and `MultiL1SNRDBLoss` gain `check_finite` (default `True`, preserving current behaviour).
The non-finite input scan costs four full-tensor passes whose results are consumed by a Python `if`, which on
CUDA forces a host-device synchronization and serializes the pipeline. Setting it `False` saves about 1% of a
CPU forward and removes those synchronizations; the loss then propagates `NaN` rather than sanitizing it,
which is arguably preferable during training anyway. With `check_finite=False` a non-finite value in **either**
input reaches the loss: earlier in this branch a non-finite *target* was silently absorbed and returned a
clean `-0.0` with a zero gradient, because the all-resolutions-failed fallback carried non-finiteness only
from the estimate.

Note what the scan substitutes when it is enabled: `NaN` becomes `0.0`, but `±Inf` becomes `±1.0`, which is
full-scale audio, not silence. A corrupt `Inf` sample is replaced by a full-scale click rather than by
nothing, which matters when reading a level-matching regularizer's output. The warning and two docstrings
previously described the substitution as zeroing, and mentioned only the NaN case.

**The spectrogram loss is about 14% faster.** The STFT window normalization is folded into the window itself
rather than dividing the whole complex output by the same constant on every call. This is mathematically
identical, since the transform is linear in the window, and it removes one full-tensor pass per resolution
per tensor. Measured +14.2% with a standard deviation of 3.2% across interleaved A/B trials on
`[8, 2, 264600]`, against a noise floor of 0.5%.

Loss values move only at float32 rounding; gradients move more. Both are quantified in the table in BREAKING
CHANGES above. Float64 accuracy **improves** rather than degrading, because the normalization constant is computed in double
precision before being cast to the window's dtype. Two independent measurements against differently-built
float64 references put the improvement between 1.4x and 8x, so the exact ratio is a property of the reference
as much as of the code and no single figure is quoted. Against 0.1.x the gain is larger still, since that
version downcast the whole spectrogram result to float32.

Two further optimizations were investigated and did not ship, recorded because the reasons are useful:

- Reducing over dimensions instead of reshaping the strided real and imaginary views avoids roughly 129 MB of
  contiguous copies per resolution, so about 389 MB per forward across the three defaults. The forward is bit-identical, but the backward graph is not, and the
  measured time saving was 0.06%. Not worth a change to gradients for no measurable gain.
- Sharing one STFT between the reconstruction and regularizer paths: **there was nothing to share.** A call
  counter shows six transform calls with the regularizer enabled and six with it disabled, so the transform
  was never recomputed. The premise was wrong.
- Guarding the per-call device move on the spectrogram transforms against a cached device, worth 0.02 ms
  (about 0.007% of a forward). The cache was set in `__init__` and written only in `forward`, but
  `nn.Module.to()` moves the window buffers without touching a plain Python attribute, so after
  `loss.to(device)` the guard skipped a move that was needed, every resolution failed on a device mismatch,
  and the loss became exactly `0.0` with a zero gradient. On Apple silicon that was the default path, because
  `mps_cpu_fallback` puts the input on CPU while the buffers are on MPS. A correct guard is possible by
  reading the buffer's device instead of a cache; 0.007% does not justify an invariant whose violation
  silently zeroes a training signal.

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




### Documentation corrections

Several documented claims did not match the implementation:

- **Documented three inherent properties an edge-case sweep surfaced.** None is a code change; each is a
  behaviour a user can hit and could not have predicted from the previous text. `STFTL1SNRDBLoss` alone is
  **not monotone in reconstruction quality**: on `[4, 2, 44100]` at amplitude 0.05 a DC offset equal to the
  signal amplitude scores -23.6 dB where 10% white noise scores -17.6 dB, though the time-domain D1 rates the
  DC error about 10 dB worse. Two independent causes, both from mean-reducing a D1 over spectrogram
  components. A real-valued error is nearly invisible to the imaginary term (`mean|err_im|` 4.9e-09 against
  `mean|err_re|` 2.7e-03), and `l1snr_eps` *limits* that rather than causing it -- at `eps=0` the gap widens
  from 5.9 dB to 57.8 dB. Separately, the mean over bins dilutes a concentrated error: 66.7% of the DC error
  sits in 1 of 1025 bins, which is why reformulating on the complex modulus does not fix it.
  `MultiL1SNRDBLoss` at the default `spec_weight=0.5` orders both pairs correctly. `dbrms` **overflows in float32** once `|x|` reaches about 2.5e16
  at realistic shapes, because `mean(x**2)` squares before reducing; the inputs stay finite so `check_finite`
  cannot see it. And the level-matching regularizer has **exactly zero gradient at digital silence**, since
  `d/dx sqrt(mean(x^2) + eps)` is 0 at 0, so escape pressure at a fully collapsed output comes from the D1
  term rather than from the regularizer.

- **All README examples now run.** Four of the five runnable examples raised
  `RuntimeError: element 0 of tensors does not require grad` on `loss.backward()`, because the example
  tensors were built without `requires_grad=True`. A fifth omitted `.backward()` entirely. Every example is
  now executed by the test suite so they cannot regress.
- **Corrected the v0.1.3 MPS root cause.** The 0.1.3 entry below attributed the incorrect gradients to
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
  `spec_reg_coef`. It was already a constructor parameter of `MultiL1SNRDBLoss` in 0.1.3 and already
  forwarded to the spectrogram loss; only its documentation was missing.
- **Added attribution** for the authors' reference implementations, `kwatcharasupat/bandit` (Apache-2.0) and
  `kwatcharasupat/query-bandit` (MIT, Copyright (c) 2024 Karn Watcharasupat), alongside the papers.
- **Fixed the citation for arXiv:2309.02539**, published in IEEE Open Journal of Signal Processing vol. 5,
  pp. 73-81, **2024**, doi:10.1109/OJSP.2023.3339428 (previously cited as 2023, with no volume or DOI).
- **Unswapped references [1] and [2]**, which disagreed between the source header and the README.
- **Completed this changelog.** Versions 0.0.1 through 0.0.5 were absent, and 0.1.0 was labelled the initial
  release with the wrong date.

### Test suite

Rebuilt. Not shipped in the wheel, so this changes nothing for users, but it is the reason to
trust the corrections above. 112 of the previous 127 test cases compared a wrapper against the function it
wraps, which is structurally `f(x) == f(x)` and cannot fail for any implementation. Stubbing all four
`forward` methods to a constant left 122 of 127 passing, and a byte-exact revert of the v0.1.2 bugfix left
all 127 green.

Tests now compare against an independent reference derived from the published definitions and validated
against hand-computed values. 259 tests, coverage 79% to 90%, and CI enforces both a mutation gate (every
test that exercises a `forward` must fail when it returns a constant) and a coverage floor. Reverting the
v0.1.2 fix now breaks 50 tests. The MPS tests were rewritten to use an input size where the underlying
PyTorch bug actually manifests; the previous ones compared CPU against CPU and would have passed regardless.

The suite also no longer litters the working tree. `tests/test_docs.py` used `tempfile.mkdtemp()` and
`tests/mutation_gate.py --audit-markers` used `tempfile.mktemp()`, both of which trust
`tempfile.gettempdir()` to point outside the repository. It does not always: `gettempdir()` probes `TMPDIR`,
`TEMP`, `TMP`, `/tmp`, `/var/tmp` and `/usr/tmp` for writability and falls back to `os.getcwd()` when all of
them fail, which is what a sandboxed or locked-down `/tmp` produces. Both then wrote into the repository
root, and neither path was gitignored. This is environment-dependent rather than universal: on the
`ubuntu-latest` runner `/tmp` is writable, so CI never saw it, and it reproduces wherever `/tmp` is not.
Both now use
`tempfile.TemporaryDirectory`, which cleans up wherever it lands, and a new gate asserts that running them
adds nothing to `git status --porcelain --untracked-files=all`.

### Packaging

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

> **Note added in 0.2.0:** this entry originally attributed the bug to `torch.abs()` backward on complex
> tensors. That was incorrect; the cause is `torch.stft` backward above an input length of 65,536 samples.
> See the 0.2.0 entry. The fix itself was effective. The gradient error direction also differs by PyTorch
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
