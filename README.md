![torch-l1-snr-logo](https://raw.githubusercontent.com/crlandsc/torch-l1-snr/main/images/logo.png)

[![LICENSE](https://img.shields.io/github/license/crlandsc/torch-l1-snr)](https://github.com/crlandsc/torch-l1-snr/blob/main/LICENSE) [![GitHub Repo stars](https://img.shields.io/github/stars/crlandsc/torch-l1-snr)](https://github.com/crlandsc/torch-l1-snr/stargazers) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch-l1-snr)](https://pypi.org/project/torch-l1-snr/) [![PyPI - Version](https://img.shields.io/pypi/v/torch-l1-snr)](https://pypi.org/project/torch-l1-snr/) [![Downloads](https://img.shields.io/pepy/dt/torch-l1-snr)](https://pepy.tech/project/torch-l1-snr)


L1 Signal-to-Noise Ratio (SNR) loss functions for audio source separation in PyTorch. This package provides four loss functions that combine implementations from recent academic research with novel extensions, designed to integrate easily into any audio separation or enhancement training pipeline.

The core [`L1SNRLoss`](#example-l1snrloss-time-domain) is based on the loss function described in [[1]](https://arxiv.org/abs/2309.02539). [`L1SNRDBLoss`](#example-l1snrdbloss-time-domain-with-regularization) adds adaptive level-matching regularization proposed in [[2]](https://arxiv.org/abs/2501.16171). [`STFTL1SNRDBLoss`](#example-stftl1snrdbloss-spectrogram-domain) provides a spectrogram-domain L1SNR-style loss (real/imag STFT components as in [[1]](https://arxiv.org/abs/2309.02539) / [[3]](https://arxiv.org/abs/2406.18747)). [`MultiL1SNRDBLoss`](#example-multil1snrdbloss-combined-time--spectrogram) combines time-domain and spectrogram-domain losses into a single loss function for convenience and flexibility. Optional novel algorithmic extensions have also been included (such as multi-resolution STFT averaging, spectrogram-domain adaptation of the level-matching regularizer from [[2]](https://arxiv.org/abs/2501.16171), and blending of standard L1 loss) with the goal of increasing flexibility for improved performance depending on the specific task.

## Quick Start

```python
import torch
from torch_l1_snr import MultiL1SNRDBLoss

# Create combined time + spectrogram domain loss function with adaptive regularization
loss_fn = MultiL1SNRDBLoss(name="multi_l1_snr_db_loss")

# Calculate loss between model output and target
estimates = torch.randn(4, 32000, requires_grad=True)  # (batch, samples)
targets = torch.randn(4, 32000)
loss = loss_fn(estimates, targets)
loss.backward()
```

## Loss Functions

- [**Time-Domain L1SNR Loss**](#example-l1snrloss-time-domain): A basic, time-domain L1-SNR loss, based on [[1]](https://arxiv.org/abs/2309.02539).
- [**Regularized Time-Domain L1SNRDBLoss**](#example-l1snrdbloss-time-domain-with-regularization): An extension of the L1SNR loss with adaptive level-matching regularization from [[2]](https://arxiv.org/abs/2501.16171), plus an optional L1 loss component.
- [**Multi-Resolution STFT L1SNRDBLoss**](#example-stftl1snrdbloss-spectrogram-domain): A spectrogram-domain L1SNR-style loss (real/imag STFT components as in [[1]](https://arxiv.org/abs/2309.02539) / [[3]](https://arxiv.org/abs/2406.18747)), computed over multiple STFT resolutions, with optional spectrogram-domain level-matching regularization inspired by its time-domain counterpart in [[2]](https://arxiv.org/abs/2501.16171).
- [**Combined Multi-Domain Loss**](#example-multil1snrdbloss-combined-time--spectrogram): `MultiL1SNRDBLoss` combines time-domain and spectrogram-domain losses into a single, weighted objective function.

## Additional Features

- **L1 Loss Blending**: The `l1_weight` parameter allows mixing between L1SNR and standard L1 loss, softening the ["all-or-nothing" behavior](#all-or-nothing-behavior-and-l1_weight) of pure SNR losses for more nuanced separation.
- **Multi-Resolution STFT Averaging** - Extending an STFT-based loss to multiple resolutions is common in recent literature.
- **Spectrogram-Domain Adaptation of Level-Matching Regularizer [[2]](https://arxiv.org/abs/2501.16171)** - Options to extend adaptive level-matching regularization to spectrogram-domain. Experimental and not used by default.
- **Time vs. Spectrogram Loss Balancing** - Allows fine-tuning the relative contribution of time-domain and spectrogram-domain losses in `MultiL1SNRDBLoss` via the `spec_weight` parameter. Not a novel extension: the authors' own [`bandit`](https://github.com/kwatcharasupat/bandit) exposes equivalent `time_weight`/`freq_weight` controls, and the single-knob `spec_weight` is a convenience over the same idea.
- **Numerical Stability**: Robust handling of `NaN` and `inf` values during training in `STFTL1SNRDBLoss` (and in `MultiL1SNRDBLoss` through it), controlled by `check_finite` (default `True`; set it `False` to skip the scan and let non-finite values propagate visibly, which also removes four host-device synchronizations per call on CUDA). The time-domain losses `L1SNRLoss` and `L1SNRDBLoss` do **not** sanitize non-finite input: a `NaN` estimate propagates to a `NaN` loss, which is visible rather than silent.
- **Short Audio Fallback**: Graceful fallback to time-domain loss when audio is too short for STFT processing.

## Installation

### Install from PyPI

```bash
pip install torch-l1-snr
```

### Install from GitHub

```bash
pip install git+https://github.com/crlandsc/torch-l1-snr.git
```

Or, you can clone the repository and install it in editable mode for development:

```bash
git clone https://github.com/crlandsc/torch-l1-snr.git
cd torch-l1-snr
pip install -e .
```

## Dependencies

- [PyTorch](https://pytorch.org/)
- [torchaudio](https://pytorch.org/audio/stable/index.html)

## Supported Tensor Shapes

All loss functions in this package (`L1SNRLoss`, `L2SNRLoss`, `L1SNRDBLoss`, `STFTL1SNRDBLoss`, and `MultiL1SNRDBLoss`) accept standard audio tensors of shape `(batch, samples)`, `(batch, channels, samples)`, or `(batch, num_sources, channels, samples)`. For the time-domain losses, any 3D/4D input is flattened across all non-batch dimensions (e.g., sources, channels, and samples) into a single vector per example before the loss is computed. For the spectrogram-domain loss, inputs are reshaped to `(batch, streams, samples)` by flattening all non-time dimensions into a “stream” dimension (e.g., `streams = channels` or `streams = num_sources * channels`), and a separate STFT is computed for each stream.

## Usage

The loss functions can be imported directly from the `torch_l1_snr` package.

### `L1SNRLoss` (Time Domain)

The simplest loss function - pure L1SNR without regularization.

```python
import torch
from torch_l1_snr import L1SNRLoss

# Create dummy audio signals
estimates = torch.randn(4, 2, 44100, requires_grad=True)  # Batch of 4, stereo, 44100 samples
actuals = torch.randn(4, 2, 44100)

# Basic L1SNR loss
loss_fn = L1SNRLoss(name="l1_snr_loss")

# Calculate loss
loss = loss_fn(estimates, actuals)
loss.backward()

print(f"L1SNRLoss: {loss.item()}")
```

### `L2SNRLoss` (Time Domain, Energy Ratio, Experimental)

Energy-ratio sibling of `L1SNRLoss`, computing `10*log10((mean(e²) + tau*mean(y²) + eps) / (mean(y²) + eps))`. Where `L1SNRLoss` measures mean-absolute error, this measures error **energy**, which is what SDR measures. It is the tau-clamped SNR of the universal sound separation literature.

**This is an opt-in experiment, not a recommendation.** Nothing selects it by default and no other loss changes because it exists. The theoretical case for matching the metric's norm is real, but published music-separation results point the other way: both [Demucs](https://arxiv.org/abs/1911.13254) and [HS-TasNet](https://arxiv.org/abs/2402.17701) chose L1 over SI-SNR/SD-SDR for this task after finding the metric-matched losses converged more slowly and scored worse. It is provided so that comparison can be run rather than argued.

```python
import torch
from torch_l1_snr import L2SNRLoss

estimates = torch.randn(4, 2, 44100, requires_grad=True)
actuals = torch.randn(4, 2, 44100)

# tau caps the attainable SNR at -10*log10(tau) dB; the default 1e-3 gives a 30 dB cap
loss_fn = L2SNRLoss(name="l2_snr_loss", tau=1e-3)

loss = loss_fn(estimates, actuals)
loss.backward()

print(f"L2SNRLoss: {loss.item()}")
```

Two floors, both needed. `tau` is relative to the target and bounds the gradient growth as a source converges. `eps` is absolute and is what keeps a **silent** target finite: `tau*mean(y²)` is zero when the target is zero, so `tau` alone would leave the loss unbounded on exactly the silent chunks that are common in stem training.

**Both floors are level-dependent, because the effective floor is whichever dominates (`tau*mean(y²) + eps`).** Two consequences worth knowing before you read a loss curve:

- The **30 dB cap holds only above about -30 dBFS RMS** (`mean(y²) >> eps/tau`). Below that `eps` takes over and the best attainable value rises: -29.6 dB at -20 dBFS, -19.6 dB at -40 dBFS, -3.0 dB at -60 dBFS. `L1SNRLoss` has the same level-dependence, documented under Limitations.
- The **gradient ceiling is `c/sqrt(tau*mean(y²) + eps)`**: 4343 on a silent target, about 136 at 0 dBFS. `L1SNRLoss`'s ceiling is 4343 regardless of level, so the two agree at silence and diverge by up to ~32x on loud targets. The `eps` default of `1e-6` is the power-domain analogue of `L1SNRLoss`'s amplitude-domain `1e-3`, which is what makes the **silent-target** ceilings match. It does not make them match everywhere.

**Do not run this loss in pure `float16`, and do not lower `eps` below `1e-6` if you might.** float16's smallest subnormal is `5.96e-08`, so `eps=1e-8` rounds to exactly zero; with a silent target the numerator and denominator then both collapse and the loss returns **`+inf`**, killing the run. The `1e-6` default survives but only as a float16 subnormal (`1.013e-06`), so hardware that flushes subnormals to zero would break it too, and float16's smallest *normal* value (`6.1e-05`) is far too large to use as a floor. There is no fully float16-safe setting.

`bfloat16` is unaffected, and the reason is worth knowing: it trades mantissa for exponent (8 mantissa bits against float16's 11, but float32's exponent range), so `1e-8` does not underflow while being *less* precise per digit (no binary float represents `1e-8` exactly -- bfloat16 stores it as `1.0012e-08`, float32 as `9.99999994e-09`; what matters is that neither rounds it to zero). **`torch.autocast` does not rescue this.** `mean` and `log10` are not autocast-cast operations, so under autocast the computation stays in the input dtype: measured, float16 input under `autocast(float16)` returns a float16 result, not a promoted float32 one. What actually saves a typical mixed-precision loop is ordinary dtype promotion in `estimates - actuals` when one side is float32. If both sides are float16, autocast does not help. Measured accuracy against a float64 reference at 10% error: float32 exact, bfloat16 0.028 dB at `eps=1e-6` and 0.0006 dB at `eps=1e-8`.

Because a silent target can score anywhere from 0 to +60 dB while a normal one sits near -20 dB, a single silent row pulls a batch mean up hard: one silent row in four moved a measured batch mean from -19.3 to -10.9 dB. `tau=0` is accepted and removes the relative floor entirely, leaving `eps` alone.

**It reports about twice the decibels `L1SNRLoss` does for the same estimate, and neither is wrong.** `L1SNRLoss` takes `10*log10` of an *amplitude* ratio (the authors' bandit convention, preserved bit-exactly); a power ratio in decibels is `10*log10`, so this is effectively `20*log10` of the equivalent amplitude ratio. Measured ratio at the defaults: **1.26x to 2.13x**, not a fixed 2x. It approaches 2x only for Gaussian error on a quiet target far from convergence, and falls as the model improves (2.13x at 10% relative error on a 0.05-amplitude target, 1.26x at 0.3% on a unit-amplitude one).

**Error *shape* moves it independently, and this is the more interesting half.** At fixed error energy, `L2SNRLoss` is shape-blind because it only sees RMS, while `L1SNRLoss` reports a *better* score as the error becomes impulsive, because `mean|e|` falls at constant RMS. Measured at 10% relative error: Gaussian error gives D1 -9.14 and D2 -19.43 (ratio 2.13), while error confined to 0.1% of samples gives D1 -15.59 and D2 -19.58 (ratio 1.26) -- D2 barely moved, D1 improved by 6.5 dB for the same energy. Since uSDR is itself an energy ratio, that shape-blindness is the substantive argument for this loss, rather than the decibel scale.

That matters when you use it as `time_loss_module`. `MultiL1SNRDBLoss` combines its branches as `(1 - spec_weight) * time + spec_weight * spec`, so a time term of twice the magnitude takes twice the share of the objective at the same `spec_weight`: measured at `spec_weight=0.5` and 10% error, `|time| / |spectral|` goes from **0.518** with `L1SNRLoss` to **1.101** with this one. **A two-arm A/B that swaps only the time loss varies the norm and the domain balance together.** Sweep `spec_weight` in the L2 arm, or raise it to roughly **0.68** to restore the L1 arm's balance.

To A/B it against `L1SNRDBLoss` inside the multi-domain loss, pass it as `MultiL1SNRDBLoss(time_loss_module=L2SNRLoss('time'))`. Note that doing so drops the time-domain regularizer and shifts the domain balance, both of which the constructor warns about -- so prefer comparing the two losses **standalone**, which isolates the norm and avoids both effects.

### `L1SNRDBLoss` (Time Domain with Regularization)

Adds adaptive level-matching regularization to prevent silence collapse.

```python
import torch
from torch_l1_snr import L1SNRDBLoss

# Create dummy audio signals
estimates = torch.randn(4, 2, 44100, requires_grad=True)  # Batch of 4, stereo, 44100 samples
actuals = torch.randn(4, 2, 44100)

# Initialize the loss function with regularization enabled
# l1_weight=0.1 leans heavily toward L1SNR; see the calibration note below
loss_fn = L1SNRDBLoss(
    name="l1_snr_db_loss",
    use_regularization=True,  # Enable adaptive level-matching regularization
    l1_weight=0.1             # interpolation coefficient, not a behaviour fraction
)

# Calculate loss
loss = loss_fn(estimates, actuals)
loss.backward()

print(f"L1SNRDBLoss: {loss.item()}")
```

### `STFTL1SNRDBLoss` (Spectrogram Domain)

Computes L1SNR loss across multiple STFT resolutions.

```python
import torch
from torch_l1_snr import STFTL1SNRDBLoss

# Create dummy audio signals
estimates = torch.randn(4, 2, 44100, requires_grad=True)  # Batch of 4, stereo, 44100 samples
actuals = torch.randn(4, 2, 44100)

# Initialize the loss function without regularization or traditional L1
# Uses multiple STFT resolutions by default: [512, 1024, 2048] FFT sizes
loss_fn = STFTL1SNRDBLoss(
    name="stft_l1_snr_db_loss",
    l1_weight=0.0              # Pure L1SNR (no regularization, no L1)
)

# Calculate loss
loss = loss_fn(estimates, actuals)
loss.backward()

print(f"STFTL1SNRDBLoss: {loss.item()}")
```

### `MultiL1SNRDBLoss` (Combined Time + Spectrogram)

Combines time-domain and spectrogram-domain losses into a single weighted objective.

```python
import torch
from torch_l1_snr import MultiL1SNRDBLoss

# Create dummy audio signals
estimates = torch.randn(4, 2, 44100, requires_grad=True)  # Batch of 4, stereo, 44100 samples
actuals = torch.randn(4, 2, 44100)

# Initialize the multi-domain loss function
loss_fn = MultiL1SNRDBLoss(
    name="multi_l1_snr_db_loss",
    weight=1.0,                    # Overall weight for this loss
    spec_weight=0.6,               # coefficients: 0.4 * time_loss + 0.6 * spec_loss
                                   # (see Limitations: 0.5 is the paper-faithful default)
    l1_weight=0.1,                 # applies to both domains; see the calibration note
    use_time_regularization=True,  # Enable regularization in time domain
    use_spec_regularization=False  # Disable regularization in spec domain
)

# Calculate loss
loss = loss_fn(estimates, actuals)
loss.backward()

print(f"Multi-domain Loss: {loss.item()}")
```

### `dbrms` (Utility)

Also exported is `dbrms`, the level measurement the regularizers are built on. It returns the RMS level in decibels for each element of a batch, flattening all non-batch dimensions:

```python
import torch
from torch_l1_snr import dbrms

audio = torch.randn(4, 2, 44100) * 0.1  # (batch, channels, samples)
levels = dbrms(audio)                   # (4,) tensor of dBRMS values

print(levels)
```

`dbrms(x, eps=1e-8)` computes `20 * log10(sqrt(mean(x**2) + eps))`. The `eps` sits inside the square root, on a power quantity, and puts the floor for a digitally silent input at exactly -80 dB -- deliberately well below the `lmin=-60` threshold the adaptive regularizer uses, so a silent target is correctly recognized as silent. (Before v0.2.0 a second epsilon was also added outside the root, on an amplitude; it could never prevent a log of zero and shifted the silence floor to -79.99913 dB.)

## Motivation

The goal of these loss functions is to provide a perceptually-informed and robust alternative to common audio losses like L1, L2 (MSE), and SI-SDR for training audio source separation models.

- **Robustness**: The L1 norm is less sensitive to large outliers than the L2 norm, making it more suitable for audio signals which can have sharp transients.
- **Perceptual Relevance**: The loss is scaled to decibels (dB), which more closely aligns with human perception of loudness.
- **Adaptive Regularization**: Prevents the model from collapsing to silent outputs by penalizing mismatches in the overall loudness (dBRMS) between the estimate and the target.

This package is motivated by, and largely follows, the objectives and regularizers described in the cited papers ([1–3]). Several novel algorithmic extensions have been included with the goal of increasing flexibility for improved performance depending on the specific task.

### Level-Matching Regularization

A key feature of `L1SNRDBLoss` is the adaptive regularization term, as described in [[2]](https://arxiv.org/abs/2501.16171). This component calculates the difference in decibel-scaled root-mean-square (dBRMS) levels between the estimated and actual signals. An adaptive weight (`lambda`) is applied to this difference, which increases when the model incorrectly silences a non-silent target. This encourages the model to learn the correct output level and specifically avoids the model collapsing to a trivial silent solution when uncertain.

### Multi-Resolution Spectrogram Analysis

The `STFTL1SNRDBLoss` module applies the L1SNRDB loss across multiple time-frequency (spectrogram) resolutions. While not mentioned in the cited papers, by analyzing the signal with *multiple different* STFT window sizes and hop lengths, the loss function can capture a wider range of artifacts - from short, transient errors to longer, tonal discrepancies. This provides a more comprehensive error signal to the model during training. Using multiple resolutions for an STFT loss is common among many recent source separation works, such as the [Band-Split RoPE Transformer](https://arxiv.org/abs/2309.02612).

### "All-or-Nothing" Behavior and `l1_weight`

A characteristic of these SNR-style losses that I experienced in many training experiments is that they encourage the model to make definitive, "all-or-nothing" separation decisions. This can be highly effective for well-defined sources (e.g. drums vs vocals), as it pushes the model to be confident in its estimations. However, this can also lead to "confident errors," where the model completely removes a signal component it should have kept. This poses a tradeoff for sources that may share greater similarities (e.g. speech vs singing vocals).

While the Level-Matching Regularization prevents a *total collapse to silence*, it does not by itself solve this issue of overly confident, hard-boundary separation. To provide a tunable solution, this implementation introduces a novel `l1_weight` hyperparameter. This allows you to create a hybrid loss, blending the decisive L1SNR objective with a standard L1 loss to soften its "all-or-nothing"-style behavior and allow for more nuanced separation.

> **Anecdotal, not measured.** While this can potentially reduce the "cleanliness" of separations and slightly harm metrics like SDR, I found that re-introducing some standard L1 loss allows for slightly more "smearing" of sound between sources to mask large errors and be more perceptually acceptable for sources with many similarities. I have no hard numbers to report on this yet, just my experience.

So I recommend starting with no standard L1 mixed in (`l1_weight=0.0`), and then slowly increasing from there based on your needs.

-   `l1_weight=0.0` (Default): Pure L1SNR (+ regularization).
-   `l1_weight=1.0`: Pure standard L1 loss.
-   `0.0 < l1_weight < 1.0`: A weighted combination of the two.

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

## Device Compatibility

All loss functions work on **CPU**, **CUDA**, and **MPS** (Apple Silicon).

**MPS note:** PyTorch's MPS backend produces numerically incorrect gradients from `torch.stft` backward above an input length of 65,536 samples (2^16). The forward transform is correct to float32 precision, so the failure is silent. The error is not a simple function of size: a handful of specific lengths are exact while neighbouring ones are wrong by anywhere from 30% to 99%, so it cannot be predicted or avoided by choosing a particular window length. Batch sizes above 1 fail even at the lengths that are exact at batch 1. This affects `STFTL1SNRDBLoss` and `MultiL1SNRDBLoss` (which use STFT internally). As of v0.1.3, these losses automatically route STFT computation through CPU when on MPS (`mps_cpu_fallback=True` by default), producing correct gradients with negligible performance impact. Time-domain losses (`L1SNRLoss`, `L1SNRDBLoss`) are unaffected. CUDA and CPU users are completely unaffected by this change.

Typical audio training uses windows well above that threshold (6 seconds at 44.1 kHz is 264,600 samples), so **leave the fallback enabled on Apple silicon** unless you have verified on your own PyTorch version that the backward pass is correct.

To disable the workaround (e.g., if a future PyTorch release fixes the MPS bug):
```python
loss_fn = STFTL1SNRDBLoss(name="stft_loss", mps_cpu_fallback=False)
```

## Limitations

- The L1SNR loss is not scale-invariant. Unlike SI-SNR, it requires the model's output to be correctly scaled relative to the target.
- While the dB scaling and regularization are psychoacoustically motivated, the loss does not model more complex perceptual phenomena like auditory masking.
- **The usable dynamic range collapses for quiet targets.** Because `eps` sits in both the numerator and denominator, D1's floor at perfect reconstruction is `10*log10(eps / (mean|y| + eps))` rather than negative infinity. With the default `eps=1e-3` that floor is roughly -30 dB at `mean|y|=1`, -20 dB at 0.1, -10 dB at 0.01, and only -3 dB at 1e-3. A target near -58 dBFS RMS therefore has under 3 dB of total loss range to optimize within. This is inherited from the reference implementation rather than introduced here, but it means very quiet stems carry correspondingly little gradient signal, and it is worth checking your target levels before attributing poor performance on quiet sources to the model. The papers note the same constraint, that the loss is numerically stable for `eps` *not* much smaller than the signal norm.
- **`spec_weight` is a loss-value weight, not a gradient share.** In `MultiL1SNRDBLoss` the combination is `(1 - spec_weight) * time_loss + spec_weight * spec_loss`, so the coefficients are exactly as documented. But `spec_loss` internally sums a real and an imaginary D1 term while `time_loss` is a single term, so equal coefficients do not mean the two domains contribute equally to the gradient. The default `0.5` is chosen for a specific reason: it is the value at which the time, real and imaginary terms all receive weight 0.5, reproducing the equal 1:1:1 weighting of the objective in [[3]](https://arxiv.org/abs/2406.18747). Prefer to leave it there unless you have a reason to shift domain emphasis.
- **`STFTL1SNRDBLoss` alone is not monotone in reconstruction quality.** Measured on `[4, 2, 44100]` stems at amplitude 0.05: a DC offset equal to the signal amplitude scores **-23.6 dB** while 10% white noise scores **-17.6 dB**, even though the time-domain D1 rates the DC error about 10 dB *worse* (+0.96 against -9.13). Two independent mechanisms, both properties of applying a mean-reduced D1 to spectrogram components rather than defects in this code. First, a real-valued error is almost invisible to the imaginary term: for the DC offset, `mean|err_im|` is 4.9e-09 against `mean|err_re|` of 2.7e-03, six orders of magnitude apart, so a near-purely-real error gets a near-free pass on half the objective. Note that `l1snr_eps` *limits* this rather than causing it, by flooring how large the free pass can be: at `l1snr_eps=0` the same comparison widens from 5.9 dB to 57.8 dB. Second, the mean over frequency bins dilutes a spectrally concentrated error: 66.7% of the DC error's magnitude sits in a single one of 1025 bins. The second mechanism survives reformulating the loss on the complex modulus instead of on Re and Im separately, so that is not a fix. Practically: do not use a bare `STFTL1SNRDBLoss` where a systematic offset or a strongly tonal error is plausible, and do not read the spectral term of a `MultiL1SNRDBLoss` run as a standalone quality metric. `MultiL1SNRDBLoss` at the default `spec_weight=0.5` orders both pairs correctly, because the time-domain term carries enough weight to dominate, which is one reason it is the recommended entry point.
- **`L2SNRLoss` overflows in float32 sooner than the L1 losses.** It reduces `mean(e**2)`, which squares before summing, so at 44100 samples per row the accumulator overflows around `|x| ~ 8.8e16` and the loss becomes `+inf` (measured: 364.8 at 8.7e16, `inf` at 8.8e16). `NaN` needs the *target* to overflow as well, which a diverging estimate against a fixed target does not cause. `L1SNRLoss` reduces `mean|e|` and survives past 1e30. Same mechanism as the `dbrms` bullet below, reachable only on a run already diverging through 1e16, and loud rather than silent. float64 is unaffected.
- **`dbrms` overflows in float32 well below float32's range.** `mean(x**2)` squares before reducing, so the sum overflows once `|x|` reaches roughly 2.5e16 at a realistic `[8, 2, 264600]` shape (2.0e17 at 8192 elements per row) — far below the 3.4e38 the dtype can hold. The inputs are still finite, so `check_finite` cannot catch it, and the loss becomes `inf`, or `NaN` once both levels overflow and the regularizer computes `|inf - inf|`. This is only reachable on a run already diverging through 1e16, and `inf`/`NaN` is loud rather than silent, which is why the numerics are left alone. float64 is unaffected.
- **The level-matching regularizer exerts no force at exactly digital silence.** `d/dx sqrt(mean(x²) + eps)` is exactly zero at `x ≡ 0`, so the term that exists to penalize collapse contributes no gradient precisely *at* the collapse point; it becomes meaningful around 1e-4. The D1 term still supplies gradient there, so a model is not stuck — the total gradient at exact silence is nonzero — but if you use a mask-based separator whose mask saturates to exactly 0, the escape pressure comes from D1 rather than from the regularizer.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any bug fixes, improvements, or new features to suggest.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

The loss functions implemented here are largely based on the work of the authors of the referenced papers. Thank you for your research!

The core D1 objective follows the authors' own reference implementations, not only the papers. In particular the mean-normalized form used here (rather than the summed `L1` norm the papers write) matches their code, and this implementation is numerically equivalent to it. Those repositories are:

- [`kwatcharasupat/bandit`](https://github.com/kwatcharasupat/bandit) - Apache-2.0. Reference implementation for [[1]](https://arxiv.org/abs/2309.02539).
- [`kwatcharasupat/query-bandit`](https://github.com/kwatcharasupat/query-bandit) - MIT, Copyright (c) 2024 Karn Watcharasupat. Reference implementation for [[3]](https://arxiv.org/abs/2406.18747).

No official implementation of [[2]](https://arxiv.org/abs/2501.16171) has been released, so the level-matching regularizer here follows the published equations alone.

## References

[1] K. N. Watcharasupat, C.-W. Wu, Y. Ding, I. Orife, A. J. Hipple, P. A. Williams, S. Kramer, A. Lerch, and W. Wolcott, "A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation," IEEE Open Journal of Signal Processing, vol. 5, pp. 73-81, 2024. doi: [10.1109/OJSP.2023.3339428](https://doi.org/10.1109/OJSP.2023.3339428). [arXiv:2309.02539](https://arxiv.org/abs/2309.02539)

[2] K. N. Watcharasupat and A. Lerch, "Separate This, and All of these Things Around It: Music Source Separation via Hyperellipsoidal Queries," [arXiv:2501.16171](https://arxiv.org/abs/2501.16171). Preprint; not peer-reviewed at time of writing.

[3] K. N. Watcharasupat and A. Lerch, "A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems," Proceedings of the 25th International Society for Music Information Retrieval Conference, 2024. [arXiv:2406.18747](https://arxiv.org/abs/2406.18747)
