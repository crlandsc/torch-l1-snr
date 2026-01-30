![torch-l1-snr-logo](https://raw.githubusercontent.com/crlandsc/torch-l1-snr/main/images/logo.png)

[![LICENSE](https://img.shields.io/github/license/crlandsc/torch-l1-snr)](https://github.com/crlandsc/torch-l1-snr/blob/main/LICENSE) [![GitHub Repo stars](https://img.shields.io/github/stars/crlandsc/torch-l1-snr)](https://github.com/crlandsc/torch-l1-snr/stargazers) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch-l1-snr)](https://pypi.org/project/torch-l1-snr/) [![PyPI - Version](https://img.shields.io/pypi/v/torch-l1-snr)](https://pypi.org/project/torch-l1-snr/) [![Number of downloads from PyPI per month](https://img.shields.io/pypi/dm/torch-l1-snr)](https://pypi.org/project/torch-l1-snr/)


L1 Signal-to-Noise Ratio (SNR) loss functions for audio source separation in PyTorch. This package provides four loss functions that combine implementations from recent academic research with novel extensions, designed to integrate easily into any audio separation or enhancement training pipeline.

The core [`L1SNRLoss`](#example-l1snrloss-time-domain) is based on the loss function described in [[1]](https://arxiv.org/abs/2309.02539). [`L1SNRDBLoss`](#example-l1snrdbloss-time-domain-with-regularization) adds adaptive level-matching regularization proposed in [[2]](https://arxiv.org/abs/2501.16171). [`STFTL1SNRDBLoss`](#example-stftl1snrdbloss-spectrogram-domain) provides a spectrogram-domain L1SNR-style loss (real/imag STFT components as in [[1]](https://arxiv.org/abs/2309.02539) / [[3]](https://arxiv.org/abs/2406.18747)). [`MultiL1SNRDBLoss`](#example-multil1snrdbloss-combined-time--spectrogram) combines time-domain and spectrogram-domain losses into a single loss function for convenience and flexibility. Optional novel algorithmic extensions have also been included (such as multi-resolution STFT averaging, spectrogram-domain adaptation of the level-matching regularizer from [[2]](https://arxiv.org/abs/2501.16171), time vs. spectrogram loss balancing, and blending of standard L1 loss) with the goal of increasing flexibility for improved performance depending on the specific task.

## Quick Start

```python
import torch
from torch_l1_snr import MultiL1SNRDBLoss

# Create combined time + spectrogram domain loss function with adaptive regularization
loss_fn = MultiL1SNRDBLoss(name="multi_l1_snr_db_loss")

# Calculate loss between model output and target
estimates = torch.randn(4, 32000)  # (batch, samples)
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
- **Time vs. Spectrogram Loss Balancing.** - Allows fine-tuning the relative contribution of time-domain and spectrogram-domain losses in `MultiL1SNRDBLoss` via the `spec_weight` parameter.
- **Numerical Stability**: Robust handling of `NaN` and `inf` values during training.
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
- [NumPy](https://numpy.org/) (>=1.21.0)

## Supported Tensor Shapes

All loss functions in this package (`L1SNRLoss`, `L1SNRDBLoss`, `STFTL1SNRDBLoss`, and `MultiL1SNRDBLoss`) accept standard audio tensors of shape `(batch, samples)`, `(batch, channels, samples)`, or `(batch, num_sources, channels, samples)`. For the time-domain losses, any 3D/4D input is flattened across all non-batch dimensions (e.g., sources, channels, and samples) into a single vector per example before the loss is computed. For the spectrogram-domain loss, inputs are reshaped to `(batch, streams, samples)` by flattening all non-time dimensions into a “stream” dimension (e.g., `streams = channels` or `streams = num_sources * channels`), and a separate STFT is computed for each stream.

## Usage

The loss functions can be imported directly from the `torch_l1_snr` package.

### `L1SNRLoss` (Time Domain)

The simplest loss function - pure L1SNR without regularization.

```python
import torch
from torch_l1_snr import L1SNRLoss

# Create dummy audio signals
estimates = torch.randn(4, 2, 44100)  # Batch of 4, stereo, 44100 samples
actuals = torch.randn(4, 2, 44100)

# Basic L1SNR loss
loss_fn = L1SNRLoss(name="l1_snr_loss")

# Calculate loss
loss = loss_fn(estimates, actuals)
loss.backward()

print(f"L1SNRLoss: {loss.item()}")
```

### `L1SNRDBLoss` (Time Domain with Regularization)

Adds adaptive level-matching regularization to prevent silence collapse.

```python
import torch
from torch_l1_snr import L1SNRDBLoss

# Create dummy audio signals
estimates = torch.randn(4, 2, 44100)  # Batch of 4, stereo, 44100 samples
actuals = torch.randn(4, 2, 44100)

# Initialize the loss function with regularization enabled
# l1_weight=0.1 blends 90% L1SNR+Regularization with 10% L1 loss
loss_fn = L1SNRDBLoss(
    name="l1_snr_db_loss",
    use_regularization=True,  # Enable adaptive level-matching regularization
    l1_weight=0.1             # 10% L1 loss, 90% L1SNR + regularization
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
estimates = torch.randn(4, 2, 44100)  # Batch of 4, stereo, 44100 samples
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
estimates = torch.randn(4, 2, 44100)  # Batch of 4, stereo, 44100 samples
actuals = torch.randn(4, 2, 44100)

# Initialize the multi-domain loss function
loss_fn = MultiL1SNRDBLoss(
    name="multi_l1_snr_db_loss",
    weight=1.0,                    # Overall weight for this loss
    spec_weight=0.6,               # 60% spectrogram loss, 40% time-domain loss
    l1_weight=0.1,                 # Use 10% L1, 90% L1SNR+Reg in both domains
    use_time_regularization=True,  # Enable regularization in time domain
    use_spec_regularization=False  # Disable regularization in spec domain
)

# Calculate loss
loss = loss_fn(estimates, actuals)
print(f"Multi-domain Loss: {loss.item()}")
```

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

While this can potentially reduce the "cleanliness" of separations and slightly harm metrics like SDR, I found that re-introducing some standard L1 loss allows for slightly more "smearing" of sound between sources to mask large errors and be more perceptually acceptable for sources with many similarities. I have no hard numbers to report on this yet, just my experience. So I recommend starting with no standard L1 mixed in (`l1_weight=0.0`), and then slowly increasing from there based on your needs.

-   `l1_weight=0.0` (Default): Pure L1SNR (+ regularization).
-   `l1_weight=1.0`: Pure standard L1 loss.
-   `0.0 < l1_weight < 1.0`: A weighted combination of the two.

The implementation is optimized for efficiency: if `l1_weight` is `0.0` or `1.0`, the unused loss component is not computed, saving computational resources.

**Note on Gradient Balancing:** When blending losses (`0.0 < l1_weight < 1.0`), the implementation automatically scales the L1 component to approximately match the gradient magnitudes of the L1SNR component. This helps maintain stable training without manual tuning.

## Limitations

- The L1SNR loss is not scale-invariant. Unlike SI-SNR, it requires the model's output to be correctly scaled relative to the target.
- While the dB scaling and regularization are psychoacoustically motivated, the loss does not model more complex perceptual phenomena like auditory masking.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any bug fixes, improvements, or new features to suggest.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

The loss functions implemented here are largely based on the work of the authors of the referenced papers. Thank you for your research!

## References

[1] K. N. Watcharasupat, C.-W. Wu, Y. Ding, I. Orife, A. J. Hipple, P. A. Williams, S. Kramer, A. Lerch, and W. Wolcott, "A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation," IEEE Open Journal of Signal Processing, 2023. [arXiv:2309.02539](https://arxiv.org/abs/2309.02539)

[2] K. N. Watcharasupat and A. Lerch, "Separate This, and All of these Things Around It: Music Source Separation via Hyperellipsoidal Queries," [arXiv:2501.16171](https://arxiv.org/abs/2501.16171).

[3] K. N. Watcharasupat and A. Lerch, "A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems," Proceedings of the 25th International Society for Music Information Retrieval Conference, 2024. [arXiv:2406.18747](https://arxiv.org/abs/2406.18747)
