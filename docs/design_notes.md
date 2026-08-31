# torch-l1-snr design notes

Measured detail behind the README. Everything here is a property of the losses, not a how-to; the README covers installation and usage. These figures were measured on the shipped code and are kept honest by `tests/test_docs.py`.

## Blending L1 with L1SNR: `l1_weight`, `ref_level`, and `spec_ref_level`

#### `l1_weight` is an interpolation coefficient, not a behaviour fraction

`l1_weight=0.1` does **not** mean "10% L1 behaviour". Both terms push the gradient in the same direction, so what the knob actually controls is how strongly each sample's update is scaled by its own error magnitude: L1SNR scales inversely with error, L1 does not. Measuring how far that scaling has moved from pure L1SNR toward pure L1:

| `l1_weight` | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| targets at `mean\|y\| ~ 0.05`, i.e. at `ref_level` | **1.1%** | **4.1%** | **9.0%** | 18.7% | 47.1% |
| targets at `mean\|y\| ~ 0.2`, 4x above `ref_level` | 3.6% | 12.5% | 25.0% | 43.8% | 75.0% |
| targets at `mean\|y\| ~ 0.5`, 10x above `ref_level` | 8.2% | 25.6% | 44.6% | 65.2% | 87.9% |

So the knob is biased toward the SNR end across most of its range, and how strongly depends on how much your target levels vary within a batch. Earlier versions of this README stated these as flat percentages matching the parameter value, which was wrong in both directions depending on the data.

**What determines which row applies to you is your target level relative to `ref_level`, not how much your levels vary within a batch.** Level spread turns out to make almost no difference: a batch at `mean|y| ~ 0.05` gives 9.0% at `l1_weight=0.5` whether it is uniform or spans 40 dB. What matters is the level of the *loudest* stem in the batch, since that is what sets the gradient profile the metric measures.

For MUSDB-style stems at the default `ref_level=0.05`, expect somewhere between the first two rows. The measured median stem level is 0.053, but the 99th percentile is 0.116, a little over 2x `ref_level`, and a batch containing one such stem lands near 20% rather than 9%. Read the first row as a floor and the second as a typical case.

Two practical consequences. Starting at `l1_weight=0.1` introduces far less L1 character than the number suggests -- roughly 1-3% for MUSDB-style data -- so if the knob seems to do nothing, try substantially larger values before concluding the feature does not help. And the effect differs between domains: at `l1_weight=0.5` with targets at `ref_level`, the time-domain component moves about 9% toward L1 while the spectrogram component moves further, because the spectrogram operates at a lower reference magnitude relative to the shared epsilon. In `MultiL1SNRDBLoss` the single `l1_weight` therefore means somewhat different things in its two halves.

#### `ref_level`: what the L1 term is scaled against

When blending, the L1 term is scaled to be comparable with a decibel quantity, using `ref_level` (default `0.05`) as the assumed typical mean-absolute amplitude of your targets. That default is the measured median for MUSDB-style stems.

Before v0.2.0 this scale was computed per batch as a mean of reciprocals, which had two problems: one quiet target inflated the gradient for every other sample in the batch, and the knob's meaning drifted with batch content. Batch-to-batch spread in the figures above was 49.7 points at `l1_weight=0.1` and 75.4 at `0.5`; it is now 4.7 and 19.7.

To set it for your own data, measure the mean absolute value of your targets over a few batches and pass that. Being off by 2x moves the knob roughly 5 points; being off by 10x matters. `ref_level` also works as a deliberate calibration handle, since it shifts the whole curve:

| `ref_level` | 1.0 | 0.2 | **0.05** | 0.0125 | 0.005 |
|---|---|---|---|---|---|
| toward L1 at `l1_weight=0.5`, targets at `mean\|y\| ~ 0.05` | 0.5% | 2.4% | **9.0%** | 27.2% | 45.7% |

The spectrogram domain uses `spec_ref_level`, which defaults to `0.19 * ref_level`. That ratio is measured: across 496 real stem excerpts the normalized-STFT reference magnitude is about 5.6x below the time-domain one. Setting `spec_ref_level` equal to `ref_level` would be roughly 5x too large and cost about 20 points of knob position at `l1_weight=0.5`, so prefer leaving it derived unless you have measured your own.

The implementation is optimized for efficiency: if `l1_weight` is `0.0` or `1.0`, the unused loss component is not computed, saving computational resources.

**Note on Gradient Balancing:** When blending losses (`0.0 < l1_weight < 1.0`), the implementation scales the L1 component by the reference signal magnitude rather than the error magnitude. The purpose is to keep the two components' gradient *profiles* distinct: L1SNR produces inverse-error-scaled gradients while L1 produces uniform ones, and scaling by the error would collapse the second into the first (this was the v0.1.2 bugfix).

The scaling brings the two components to a comparable magnitude near 0 dB SNR, but it does not equalize them across the range, and it is not intended to. The L1 term's gradient magnitude is independent of the error, while L1SNR's grows as the error shrinks, so the ratio between them necessarily widens as the model improves. Measured on a single 8192-sample target with `ref_level=0.05`, the D1-to-L1 gradient-norm ratio is about 1.0 at 0 dB SNR, 8.5 at 20 dB, 34 at 40 dB and 49 at 60 dB when the target's `mean|y|` is exactly 0.05. Those figures shift by 20-25% with the target level -- at `mean|y| = 0.04` they are 1.25, 10.2, 36.5 and 49.0 -- so read them as the shape of the trend, not as constants.

Those particular numbers depend on the target level and on how `ref_level` is set, so treat them as an illustration of the trend rather than constants. The point to take away is that `l1_weight` is an interpolation coefficient to tune, not a guaranteed balance between the two terms.

## Numerical edge behaviours

- **`STFTL1SNRDBLoss` alone is not monotone in reconstruction quality.** Measured on `[4, 2, 44100]` stems at amplitude 0.05: a DC offset equal to the signal amplitude scores **-23.6 dB** while 10% white noise scores **-17.6 dB**, even though the time-domain D1 rates the DC error about 10 dB *worse* (+0.96 against -9.13). Two independent mechanisms, both properties of applying a mean-reduced D1 to spectrogram components rather than defects in this code. First, a real-valued error is almost invisible to the imaginary term: for the DC offset, `mean|err_im|` is 4.9e-09 against `mean|err_re|` of 2.7e-03, six orders of magnitude apart, so a near-purely-real error gets a near-free pass on half the objective. Note that `l1snr_eps` *limits* this rather than causing it, by flooring how large the free pass can be: at `l1snr_eps=0` the same comparison widens from 5.9 dB to 57.8 dB. Second, the mean over frequency bins dilutes a spectrally concentrated error: 66.7% of the DC error's magnitude sits in a single one of 1025 bins. The second mechanism survives reformulating the loss on the complex modulus instead of on Re and Im separately, so that is not a fix. Practically: do not use a bare `STFTL1SNRDBLoss` where a systematic offset or a strongly tonal error is plausible, and do not read the spectral term of a `MultiL1SNRDBLoss` run as a standalone quality metric. `MultiL1SNRDBLoss` at the default `spec_weight=0.5` orders both pairs correctly, because the time-domain term carries enough weight to dominate, which is one reason it is the recommended entry point.

- **`dbrms` overflows in float32 well below float32's range.** `mean(x**2)` squares before reducing, so the sum overflows once `|x|` reaches roughly 2.5e16 at a realistic `[8, 2, 264600]` shape (2.0e17 at 8192 elements per row) — far below the 3.4e38 the dtype can hold. The inputs are still finite, so `check_finite` cannot catch it, and the loss becomes `inf`, or `NaN` once both levels overflow and the regularizer computes `|inf - inf|`. This is only reachable on a run already diverging through 1e16, and `inf`/`NaN` is loud rather than silent, which is why the numerics are left alone. float64 is unaffected.

- **The level-matching regularizer exerts no force at exactly digital silence.** `d/dx sqrt(mean(x²) + eps)` is exactly zero at `x ≡ 0`, so the term that exists to penalize collapse contributes no gradient precisely *at* the collapse point; it becomes meaningful around 1e-4. The D1 term still supplies gradient there, so a model is not stuck — the total gradient at exact silence is nonzero — but if you use a mask-based separator whose mask saturates to exactly 0, the escape pressure comes from D1 rather than from the regularizer.
