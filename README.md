![torch-l1-snr-logo](https://raw.githubusercontent.com/crlandsc/torch-l1-snr/main/images/logo.png)

# NOTE: Repo is currently a work-in-progress and not ready for installation & use.

[![LICENSE](https://img.shields.io/github/license/crlandsc/torch-l1snr)](https://github.com/crlandsc/torch-l1snr/blob/main/LICENSE) [![GitHub Repo stars](https://img.shields.io/github/stars/crlandsc/torch-l1snr)](https://github.com/crlandsc/torch-l1snr/stargazers)

A PyTorch implementation of L1-based Signal-to-Noise Ratio (SNR) loss functions for audio source separation. This package provides implementations and novel extensions based on concepts from recent academic papers, offering flexible and robust loss functions that can be easily integrated into any PyTorch-based audio separation pipeline.

The core `L1SNRLoss` is based on the loss function described in [1], while `L1SNRDBLoss` and `STFTL1SNRDBLoss` are extensions of the adaptive level-matching regularization technique proposed in [2].

## Features

- **Time-Domain L1SNR Loss**: A basic, time-domain L1-SNR loss, based on [1].
- **Regularized Time-Domain L1SNRDBLoss**: An extension of the L1SNR loss with adaptive level-matching regularization from [2], plus an optional L1 loss component.
- **Multi-Resolution STFT L1SNRDBLoss**: A spectrogram-domain version of the loss from [2], calculated over multiple STFT resolutions.
- **Modular Stem-based Loss**: A wrapper that combines time and spectrogram domain losses and can be configured to run on specific stems.
- **Efficient & Robust**: Includes optimizations for pure L1 loss calculation and robust handling of `NaN`/`inf` values and short audio segments.

## Installation

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch-l1-snr)](https://pypi.org/project/torch-l1-snr/) [![PyPI - Version](https://img.shields.io/pypi/v/torch-l1-snr)](https://pypi.org/project/torch-l1-snr/) [![Number of downloads from PyPI per month](https://img.shields.io/pypi/dm/torch-l1-snr)](https://pypi.org/project/torch-l1-snr/)

## Install from PyPI

```bash
pip install torch-l1-snr
```

## Install from GitHub

```bash
pip install git+https://github.com/crlandsc/torch-l1snr.git
```

Or, you can clone the repository and install it in editable mode for development:

```bash
git clone https://github.com/crlandsc/torch-l1snr.git
cd torch-l1snr
pip install -e .
```

## Dependencies

- [PyTorch](https://pytorch.org/)
- [torchaudio](https://pytorch.org/audio/stable/index.html)

## Supported Tensor Shapes

All loss functions in this package (`L1SNRLoss`, `L1SNRDBLoss`, `STFTL1SNRDBLoss`, and `MultiL1SNRDBLoss`) accept standard audio tensors of shape `(batch, samples)` or `(batch, channels, samples)`. For 3D tensors, the channel and sample dimensions are flattened before the time-domain losses are calculated. For the spectrogram-domain loss, a separate STFT is computed for each channel.

## Usage

The loss functions can be imported directly from the `torch_l1snr` package.

### Example: `L1SNRDBLoss` (Time Domain)

```python
import torch
from torch_l1snr import L1SNRDBLoss

# Create dummy audio signals
estimates = torch.randn(4, 32000)  # Batch of 4, 32000 samples
actuals = torch.randn(4, 32000)

# Initialize the loss function with regularization enabled
# l1_weight=0.1 blends L1SNR+Regularization with 10% L1 loss
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

### Example: `STFTL1SNRDBLoss` (Spectrogram Domain)

```python
import torch
from torch_l1snr import STFTL1SNRDBLoss

# Create dummy audio signals
estimates = torch.randn(4, 32000)
actuals = torch.randn(4, 32000)

# Initialize the loss function
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

### Example: `MultiL1SNRDBLoss` for a Combined Time+Spectrogram Loss

This loss combines the time-domain and spectrogram-domain losses into a single, weighted objective function.

```python
import torch
from torch_l1snr import MultiL1SNRDBLoss

# Create dummy audio signals
# Shape: (batch, channels, samples)
estimates = torch.randn(2, 2, 44100) # Batch of 2, stereo
actuals = torch.randn(2, 2, 44100)

# --- Configuration ---
loss_fn = MultiL1SNRDBLoss(
    name="multi_l1_snr_db_loss",
    weight=1.0,                    # Overall weight for this loss
    spec_weight=0.6,               # 60% spectrogram loss, 40% time-domain loss
    l1_weight=0.1,                 # Use 10% L1, 90% L1SNR+Reg in both domains
    use_time_regularization=True,  # Enable regularization in time domain
    use_spec_regularization=False  # Disable regularization in spec domain
)
loss = loss_fn(estimates, actuals)
print(f"Multi-domain Loss: {loss.item()}")
```

## Motivation

The goal of these loss functions is to provide a perceptually-informed and robust alternative to common audio losses like L1, L2 (MSE), and SI-SDR for training audio source separation models.

- **Robustness**: The L1 norm is less sensitive to large outliers than the L2 norm, making it more suitable for audio signals which can have sharp transients.
- **Perceptual Relevance**: The loss is scaled to decibels (dB), which more closely aligns with human perception of loudness.
- **Adaptive Regularization**: Prevents the model from collapsing to silent outputs by penalizing mismatches in the overall loudness (dBRMS) between the estimate and the target.

#### Level-Matching Regularization

A key feature of `L1SNRDBLoss` is the adaptive regularization term, as described in [2]. This component calculates the difference in decibel-scaled root-mean-square (dBRMS) levels between the estimated and actual signals. An adaptive weight (`lambda`) is applied to this difference, which increases when the model incorrectly silences a non-silent target. This encourages the model to learn the correct output level and specifically avoids the model collapsing to a trivial silent solution when uncertain.

#### Multi-Resolution Spectrogram Analysis

The `STFTL1SNRDBLoss` module applies the L1SNRDB loss across multiple time-frequency resolutions. By analyzing the signal with different STFT window sizes and hop lengths, the loss function can capture a wider range of artifacts—from short, transient errors to longer, tonal discrepancies. This provides a more comprehensive error signal to the model during training.

#### "All-or-Nothing" Behavior and `l1_weight`

A characteristic of SNR-style losses is that they encourage the model to make definitive, "all-or-nothing" separation decisions. This can be highly effective for well-defined sources, as it pushes the model to be confident in its estimations. However, this can also lead to "confident errors," where the model completely removes a signal component it should have kept.

While the Level-Matching Regularization prevents a *total collapse to silence*, it does not by itself solve this issue of overly confident, hard-boundary separation. To provide a tunable solution, this implementation introduces a novel `l1_weight` hyperparameter. This allows you to create a hybrid loss, blending the decisive L1SNR objective with a standard L1 loss to soften its "all-or-nothing"-style behavior and allow for more nuanced separation.

-   `l1_weight=0.0` (Default): Pure L1SNR (+ regularization).
-   `l1_weight=1.0`: Pure L1 loss.
-   `0.0 < l1_weight < 1.0`: A weighted combination of the two.

The implementation is optimized for efficiency: if `l1_weight` is `0.0` or `1.0`, the unused loss component is not computed, saving computational resources.

**Note on Gradient Balancing:** When blending losses (`0.0 < l1_weight < 1.0`), you may need to tune `l1_scale_time` and `l1_scale_spec`. This is to ensure the gradients of the L1 and L1SNR components are balanced, which is crucial for stable training. The default values provide a reasonable starting point, but monitoring the loss components is recommended to ensure they are scaled appropriately.

## Limitations

- The L1SNR loss is not scale-invariant. Unlike SI-SNR, it requires the model's output to be correctly scaled relative to the target.
- While the dB scaling and regularization are psychoacoustically motivated, the loss does not model more complex perceptual phenomena like auditory masking.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or new features to suggest.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

The loss functions implemented here are based on the work of the authors of the referenced papers.

## References

[1] K. N. Watcharasupat, C.-W. Wu, Y. Ding, I. Orife, A. J. Hipple, P. A. Williams, S. Kramer, A. Lerch, and W. Wolcott, "A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation," IEEE Open Journal of Signal Processing, 2023. (arXiv:2309.02539)

[2] K. N. Watcharasupat and A. Lerch, "Separate This, and All of these Things Around It: Music Source Separation via Hyperellipsoidal Queries," arXiv:2501.16171.

[3] K. N. Watcharasupat and A. Lerch, "A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems," Proceedings of the 25th International Society for Music Information Retrieval Conference, 2024. (arXiv:2406.18747) 