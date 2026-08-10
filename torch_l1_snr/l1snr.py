# PyTorch implementation of L1SNR loss functions for audio source separation
# https://github.com/crlandsc/torch-l1-snr
# Copyright (c) 2026 Christopher Landschoot
# MIT License
#
# This implementation is based on and extends the loss functions described in:
# [1] "A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation"
#     Karn N. Watcharasupat, Chih-Wei Wu, Yiwei Ding, Iroro Orife, Aaron J. Hipple, Phillip A. Williams, Scott Kramer, Alexander Lerch, William Wolcott
#     IEEE Open Journal of Signal Processing, vol. 5, pp. 73-81, 2024
#     doi:10.1109/OJSP.2023.3339428
#     arXiv:2309.02539
# [2] "Separate This, and All of these Things Around It: Music Source Separation via Hyperellipsoidal Queries"
#     Karn N. Watcharasupat, Alexander Lerch
#     arXiv:2501.16171  (preprint; not peer-reviewed at time of writing)
# [3] "A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems"
#     Karn N. Watcharasupat, Alexander Lerch
#     Proceedings of the 25th International Society for Music Information Retrieval Conference, 2024
#     arXiv:2406.18747
#
# The core D1 objective follows the authors' own reference implementations rather than only the papers:
# the mean-normalized form used here (in place of the summed L1 norm the papers write) matches their code,
# and this implementation is numerically equivalent to it.
#   https://github.com/kwatcharasupat/bandit        Apache-2.0                              -- ref. impl. for [1]
#   https://github.com/kwatcharasupat/query-bandit  MIT, (c) 2024 Karn Watcharasupat        -- ref. impl. for [3]
# No official implementation of [2] has been released, so the level-matching regularizer here follows the
# published equations alone.

import warnings

import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram
import math
from typing import Optional, List

_WINDOW_FNS = ("hann", "hamming", "blackman", "bartlett", "kaiser")


# Ratio of the normalized-STFT reference magnitude to the time-domain one, measured as the median over 496
# real MUSDB stem-chunks (mean|Re S| = 0.00950 against mean|y| = 0.05348). Stable across the three default
# resolutions to within 1.34x and between the real and imaginary parts to within 1%, so one constant serves
# all six slots. See _audit_local/P0_SPEC_REF_LEVEL.md.
_STFT_REF_RATIO = 0.19


def _validate_ref_level(value):
    """Reject a reference level that cannot produce a usable scale.

    `not (value > 0)` rather than `value <= 0`, because NaN fails every comparison and would otherwise pass:
    a NaN ref_level yields a NaN scale and, on the spectrogram path, a loss of -0.0 with a warning that
    blames the FFT sizes. Inf is rejected for the opposite reason: it makes the scale exactly 0, so
    l1_weight silently stops mixing in any L1 at all.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"ref_level must be a positive number, got {value!r}")
    if not (value > 0.0) or not math.isfinite(value):
        raise ValueError(
            f"ref_level must be a positive finite number, got {value}. It is the typical mean-absolute "
            "amplitude of your targets; measure it over a few batches."
        )


def _validate_matching_shapes(estimates, actuals):
    """Reject differing shapes before any reshape flattens the difference away.

    Without this, reshape(batch_size, -1) makes any two tensors that agree in batch size and element count
    compare successfully, so a permuted stem/channel axis or a duration mismatch trains against a silently
    wrong pairing instead of raising.
    """
    if estimates.shape != actuals.shape:
        raise ValueError(
            f"estimates and actuals must have the same shape, got {tuple(estimates.shape)} and "
            f"{tuple(actuals.shape)}"
        )


def _validate_input_tensors(estimates, actuals):
    """Reject input shapes and dtypes that reshape() would turn into a plausible wrong number.

    Three separate silent failures, all found by feeding the losses input a training loop really produces:

    - **Rank 1.** `reshape(shape[0], -1)` reads a `[T]` waveform as T batch rows of one sample each, so the
      result is a mean of per-sample D1s. Measured 0.16 to 1.9 dB optimistic across relative errors from 0.5
      to 0.001, always in the flattering direction and always in a believable dB range.
    - **A zero-size non-batch dimension.** `torch.mean` over an empty reduction is NaN, and `[B, 0, T]` -- an
      empty stem selection -- gave NaN from the time-domain classes and 0.0 from the spectrogram one. An
      empty *batch* already raised, so the inconsistency was the tell.
    - **Integer dtype.** int16 PCM without `.float()` made `torch.stft` raise for every resolution; the
      per-resolution handler swallowed it and the spectral loss became a permanent 0.0 after one warning.
      The time-domain classes raised on the same input, so the classes disagreed.
    """
    for name, t in (("estimates", estimates), ("actuals", actuals)):
        if t.ndim < 2:
            raise ValueError(
                f"{name} must be batch-first with at least 2 dimensions, got shape {tuple(t.shape)}. A bare "
                f"[time] tensor is read as {t.shape[0] if t.ndim else 0} batch rows of one sample each and "
                "returns an optimistic number rather than an error; add a batch dimension."
            )
        if t.numel() == 0:
            raise ValueError(
                f"{name} is empty (shape {tuple(t.shape)}); a reduction over no elements is NaN"
            )
        if not t.is_floating_point():
            raise ValueError(
                f"{name} must be a floating-point tensor, got {t.dtype}. Integer PCM needs an explicit "
                "conversion, e.g. estimates.float() / 32768.0"
            )


def _validate_non_negative(name, value):
    """Reject a negative or non-finite coefficient.

    `spec_weight` is confined to [0, 1] precisely because a negative coefficient instructs the optimizer to
    maximize the error it multiplies. The same argument applies to every other coefficient and none of them
    were checked: `weight=-1.0` negated the whole objective, and `spec_reg_coef=-5.0` turned the anti-collapse
    regularizer into a reward for collapsing. Zero stays legal -- it is how a term is switched off.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative number, got {value!r}")
    if not (value >= 0.0) or not math.isfinite(value):
        raise ValueError(
            f"{name} must be non-negative and finite, got {value}. A negative coefficient instructs the "
            "optimizer to maximize the quantity it scales."
        )


def _validate_positive(name, value):
    """Reject a non-positive or non-finite stability constant. Zero defeats the purpose of an epsilon."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    if not (value > 0.0) or not math.isfinite(value):
        raise ValueError(f"{name} must be positive and finite, got {value}")


def _validate_unit_range(name, value):
    """Reject a weight outside [0, 1] with a ValueError.

    A ValueError rather than an assert: `python -O` strips assert statements, and an out-of-range weight then
    produces a silently wrong loss instead of an error.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number between 0.0 and 1.0 inclusive, got {value!r}")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0.0 and 1.0 inclusive, got {value}")


def _validate_stft_params(n_ffts, hop_lengths, win_lengths, window_fn):
    # Only list lengths, win_length <= n_fft and window_fn were checked. Everything else surfaced late and
    # confusingly, or not at all: hop_lengths=[0] constructed fine and then raised ZeroDivisionError from
    # _usable_resolutions on the first forward, mid-training; a bare n_ffts=512 gave "object of type 'int'
    # has no len()"; a float from YAML died inside hann_window; a negative hop silently dropped its
    # resolution while the warning claimed the input was too short; and three empty lists were accepted,
    # making the spectrogram branch a permanent time-domain fallback that reported the same wrong reason.
    for name, seq in (("n_ffts", n_ffts), ("hop_lengths", hop_lengths), ("win_lengths", win_lengths)):
        if isinstance(seq, (str, bytes)) or not isinstance(seq, (list, tuple)):
            raise ValueError(
                f"{name} must be a list or tuple of ints, got {seq!r}. A single resolution still needs a "
                f"list, e.g. {name}=[{seq!r}]." if isinstance(seq, int) and not isinstance(seq, bool)
                else f"{name} must be a list or tuple of ints, got {seq!r}"
            )
        if len(seq) == 0:
            raise ValueError(
                f"{name} is empty, which leaves no STFT resolution to compute and makes the spectrogram "
                "loss a permanent time-domain fallback"
            )
        for value in seq:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"every entry of {name} must be an int, got {value!r}. A float here fails later inside "
                    "the window function; config files are the usual source."
                )
            if value <= 0:
                raise ValueError(f"every entry of {name} must be positive, got {value}")
    if not (len(n_ffts) == len(hop_lengths) == len(win_lengths)):
        raise ValueError(
            f"n_ffts, hop_lengths and win_lengths must all have the same length, got "
            f"{len(n_ffts)}, {len(hop_lengths)} and {len(win_lengths)}"
        )
    for n_fft, win_length in zip(n_ffts, win_lengths):
        if win_length > n_fft:
            raise ValueError(
                f"win_length ({win_length}) must not exceed the FFT size ({n_fft})"
            )
    if window_fn not in _WINDOW_FNS:
        raise ValueError(
            f"window_fn must be one of {', '.join(_WINDOW_FNS)}, got {window_fn!r}"
        )


def dbrms(x, eps=1e-8):
    """
    Compute RMS level in decibels for a batch of signals.

    The epsilon sits inside the sqrt, on the power quantity, where it both guards log10(0) and sets the
    floor: a digitally silent input reads 20*log10(sqrt(eps)) = -80 dB at the default. That floor is
    deliberate. It is well below the lmin=-60 threshold the adaptive regularizer uses, so a silent target is
    correctly recognized as silent, and it improves on the reference implementation's -30 dB.

    A second epsilon was previously also added outside the sqrt, on an amplitude rather than a power
    quantity. Since rms is already at least sqrt(eps) = 1e-4, log10 could never see zero and the outer term
    was inert: measured under 0.001 dB across levels from silence to 10.0. Removed as a dimensional
    confusion rather than a behavioural change; tests hold both the bound and the floor.

    Args:
        x: (batch, time) or (batch, ...) tensor
        eps: stability constant, applied to the mean power. Also sets the silence floor at 10*log10(eps).
    Returns:
        (batch,) tensor of dB RMS
    """
    x = x.reshape(x.shape[0], -1)
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1) + eps)
    return 20.0 * torch.log10(rms)


class L1SNRLoss(torch.nn.Module):
    """
    Implements the L1 Signal-to-Noise Ratio (SNR) loss with optional weighted L1 loss
    component to balance "all-or-nothing" behavior.

    D1(ŷ; y) form, following the authors' reference implementation:
      D1 = 10 * log10( (mean|ŷ - y| + eps) / (mean|y| + eps) )
    L1SNR_loss = mean(D1)

    Note the means. The papers write this with a summed L1 norm, but the authors' own code
    mean-normalizes and this implementation matches their code (verified numerically equivalent).
    Reading the formula as a literal sum would predict values differing by roughly 1.5 dB.

    When l1_weight > 0, the loss combines L1SNR with scaled L1:
      loss = (1 - l1_weight) * L1SNR_loss + l1_weight * L1_auto_scaled

    Input Shape:
        Accepts waveform tensors (time-domain audio) of any shape as long as they are batch-first.
        Recommended shapes:
        - [batch, time] for single-source audio
        - [batch, num_sources, time] for multi-source audio
        - [batch, num_sources, channels, time] for multi-channel multi-source audio

    Attributes:
        name (str): Name identifier for the loss.
        weight (float): Global weight multiplier for the loss.
        eps (float): Small epsilon for numerical stability in D1 (default 1e-3 per the papers).
        l1_weight (float): Weight for the L1 term mixed into L1SNR.
        ref_level (float): Typical mean-absolute amplitude of your targets, used to scale the L1
            term when l1_weight > 0. Default 0.05, the measured median for MUSDB-like stems.
            Only affects the blended path: at l1_weight=0.0 or 1.0 it is unused.
            Replaces a per-batch statistic. Previously the scale was
            c * mean_b(1 / (mean|y|_b + eps)), a mean of reciprocals, so one quiet target inflated
            the gradient for every other sample in the batch (measured up to 5.73x) and l1_weight
            meant different things from run to run (0.5 delivering 15% to 91% of the way toward L1
            depending on batch content). To set it for your data, measure the mean absolute value
            of your targets over a few batches. Being off by 2x moves the knob about 5 points;
            being off by 10x matters.
    """
    def __init__(
        self,
        name,
        weight: float = 1.0,
        eps: float = 1e-3,
        l1_weight: float = 0.0,
        ref_level: float = 0.05,
    ):
        super().__init__()
        _validate_unit_range("l1_weight", l1_weight)
        _validate_ref_level(ref_level)
        self.ref_level = ref_level
        self.name = name
        _validate_non_negative("weight", weight)
        self.weight = weight
        self.eps = eps
        self._l1_weight = l1_weight

    @property
    def l1_weight(self):
        """Read-only. The value is baked into child modules and mode flags at construction, so mutating it
        afterwards took effect inconsistently (Q13). Construct a new loss instead."""
        return self._l1_weight

    @l1_weight.setter
    def l1_weight(self, value):
        raise AttributeError(
            "l1_weight is read-only: it is baked into child modules and mode flags when the loss is "
            "constructed, so assigning to it would take effect inconsistently. Construct a new loss with "
            "the l1_weight you want."
        )

    def forward(self, estimates, actuals, *args, **kwargs):
        _validate_matching_shapes(estimates, actuals)
        _validate_input_tensors(estimates, actuals)
        batch_size = estimates.shape[0]

        est_source = estimates.reshape(batch_size, -1)
        act_source = actuals.reshape(batch_size, -1)

        # L1 errors and reference
        l1_error = torch.mean(torch.abs(est_source - act_source), dim=-1)
        l1_true = torch.mean(torch.abs(act_source), dim=-1)

        # Auto-balanced L1/SNR mixing
        w = float(self.l1_weight)

        # Pure-L1 shortcut: avoid D1 computation
        if w >= 1.0:
            return torch.mean(l1_error) * self.weight

        # If pure SNR (w == 0) we can skip L1 scaling math
        if w <= 0.0:
            d1 = 10.0 * torch.log10((l1_error + self.eps) / (l1_true + self.eps))
            l1snr_loss = torch.mean(d1)
            return l1snr_loss * self.weight

        # Mixed path
        d1 = 10.0 * torch.log10((l1_error + self.eps) / (l1_true + self.eps))
        l1snr_loss = torch.mean(d1)

        c = 10.0 / math.log(10.0)
        # Scale by a fixed reference level, not by a batch statistic. Scaling by the *reference* rather than
        # the error is what keeps the two gradient profiles distinct (the v0.1.2 fix); using a constant rather
        # than mean_b(1 / (ref_b + eps)) is what stops one quiet target in the batch from inflating every
        # other sample's gradient, and what makes l1_weight mean the same thing from run to run.
        scale_time = c / (self.ref_level + self.eps)
        l1_term = torch.mean(l1_error) * scale_time

        loss = (1.0 - w) * l1snr_loss + w * l1_term
        return loss * self.weight


class L2SNRLoss(torch.nn.Module):
    """
    Energy-ratio sibling of L1SNRLoss: a log of floored error ENERGY rather than of mean-absolute error.

      D2 = 10 * log10( (mean((ŷ - y)^2) + tau * mean(y^2) + eps) / (mean(y^2) + eps) )
      L2SNR_loss = mean(D2)

    Note the parentheses: every term is a mean OF SQUARES, not a square of a mean. Reading
    `mean(y)^2` literally gives roughly zero for zero-mean audio and a value about 18 dB off.

    This is the tau-clamped SNR of the universal-sound-separation literature, written as a ratio so it
    matches L1SNRLoss's shape. It exists because uSDR is an energy ratio, so an energy-ratio loss is the
    metric-matched choice, whereas D1 measures mean-absolute error. Whether metric-matching actually wins
    for music separation is unsettled -- published results point the other way (Demucs and HS-TasNet both
    chose L1 over SI-SNR/SD-SDR for MSS) -- so this ships as an opt-in arm of an A/B, not as a
    recommendation. Nothing selects it by default.

    Two floors, and both are load-bearing:
      * `tau` is relative to the target and bounds the 1/x gradient growth as a source converges.
      * `eps` is absolute, and it is what keeps a SILENT target finite. tau*mean(y^2) is zero when the
        target is zero, so tau alone would leave 10*log10(mean((y_hat)^2)) unfloored on exactly the
        silent chunks that are routine in stem training.

    **The floor is whichever of the two dominates, so both the SNR cap and the gradient ceiling are
    level-dependent.** The effective floor is `tau*mean(y^2) + eps`, so:

      * The cap is -10*log10(tau) dB, i.e. 30 dB at the default, only while `mean(y^2) >> eps/tau`
        (1e-3 at the defaults, about -30 dBFS RMS). Below that `eps` takes over and the best attainable
        value rises: measured -29.6 dB at -20 dBFS RMS, -19.6 dB at -40 dBFS, -3.0 dB at -60 dBFS.
      * The gradient ceiling is c/sqrt(tau*mean(y^2) + eps), which is 4343 on a silent target but only
        about 136 at 0 dBFS. D1's ceiling is c/eps' = 4343 regardless of target level, so the two agree
        AT SILENCE and diverge by up to ~32x on loud targets. Choosing eps = eps'^2 = 1e-6 is what makes
        the silent-target ceilings match; it does not make them match everywhere. Both directions are
        gated in tests/test_losses.py.

    `tau=0` is accepted and simply removes the relative floor, leaving `eps` alone; the cap is then
    undefined rather than 30 dB.

    **Do not run this loss in pure float16, and do not lower `eps` below 1e-6 if you might.** float16's
    smallest subnormal is 5.96e-08, so `eps=1e-8` rounds to exactly zero; with a silent target both
    numerator and denominator then collapse and the loss returns **+inf**, which kills a run. The
    default 1e-6 survives, but only as a float16 subnormal (1.013e-06), so hardware that flushes
    subnormals to zero would break it too. float16's smallest NORMAL value is 6.1e-05, which is far too
    large to use as a floor here, so there is no fully float16-safe setting.

    bfloat16 is unaffected: it trades mantissa for exponent (8 mantissa bits against float16's 11, but
    float32's exponent range), so it represents 1e-8 exactly while being *less* precise per digit. That
    trade is why the wider-range format is the safe one here. `torch.autocast` is also fine on both,
    because it promotes the reduction and the log to float32 -- measured, the returned dtype is float32
    and the silent case gives 0.0004 rather than inf. Gated in tests/test_losses.py.

    **This loss reports about TWICE the decibels D1 does for the same estimate, and that is not a bug
    in either.** D1 takes 10*log10 of an AMPLITUDE ratio (mean|e| / mean|y|), which is the authors'
    bandit convention and is preserved here bit-exactly; a power ratio in decibels is 10*log10, so
    L2SNRLoss reports 20*log10 of the equivalent amplitude ratio. Measured on Gaussian error, the ratio
    of the two losses is 1.9x to 2.1x across relative errors from 0.003 to 0.5.

    The consequence is not cosmetic when this is used as `MultiL1SNRDBLoss(time_loss_module=...)`.
    That class combines its branches as `(1 - spec_weight) * time + spec_weight * spec`, so doubling
    the magnitude of the time term doubles its share of the objective at an unchanged `spec_weight`.
    Measured at `spec_weight=0.5` and 10% relative error, |time| / |spectral| goes from 0.518 with D1
    to 1.101 with this loss. **A two-arm A/B that swaps only the time loss is therefore confounded**:
    it varies the norm and the domain balance together. Either sweep `spec_weight` in the L2 arm, or
    raise it to roughly 0.68 to restore the D1 arm's balance. Gated in tests/test_losses.py.

    Input Shape:
        Same batch-first shapes as L1SNRLoss -- [batch, time], [batch, sources, time] or
        [batch, sources, channels, time]. Non-batch dimensions are flattened.

    Attributes:
        name (str): Name identifier for the loss.
        weight (float): Global weight multiplier for the loss.
        eps (float): Absolute floor on the error energy, guarding silent targets (default 1e-6).
        tau (float): Target-relative floor; caps SNR at -10*log10(tau) dB (default 1e-3, i.e. 30 dB).
    """
    def __init__(
        self,
        name,
        weight: float = 1.0,
        eps: float = 1e-6,
        tau: float = 1e-3,
    ):
        super().__init__()
        self.name = name
        _validate_non_negative("weight", weight)
        _validate_positive("eps", eps)
        _validate_non_negative("tau", tau)
        self.weight = weight
        self.eps = eps
        self.tau = tau

    def forward(self, estimates, actuals, *args, **kwargs):
        _validate_matching_shapes(estimates, actuals)
        _validate_input_tensors(estimates, actuals)
        batch_size = estimates.shape[0]

        est_source = estimates.reshape(batch_size, -1)
        act_source = actuals.reshape(batch_size, -1)

        err_energy = torch.mean((est_source - act_source) ** 2, dim=-1)
        ref_energy = torch.mean(act_source ** 2, dim=-1)

        d2 = 10.0 * torch.log10(
            (err_energy + self.tau * ref_energy + self.eps) / (ref_energy + self.eps)
        )
        return torch.mean(d2) * self.weight


class L1SNRDBLoss(torch.nn.Module):
    """
    Implements L1SNR plus adaptive level-matching regularization in the time domain
    as described in arXiv:2501.16171, with optional L1 loss component to balance
    "all-or-nothing" behavior.

    The loss combines three components:
    1. L1SNR loss: mean(10*log10((l1_error + eps) / (l1_true + eps)))
    2. Level-matching regularization: λ*|L_pred - L_true|
       Where λ is adaptively computed based on the signal levels
    3. Optional L1 loss: mean(l1_error)

    The complete loss is structured as:
    When l1_weight < 1.0: total_loss = l1snr_loss + (1-l1_weight) * mean(reg_loss)
    When l1_weight = 1.0: total_loss = l1_loss (pure L1, bypassing all other computations)

    The adaptive weighting λ for regularization increases when loud parts of a stem aren't
    reconstructed properly, helping balance between quality and level preservation.

    When l1_weight=1.0, this loss efficiently switches to a pure L1 loss calculation,
    bypassing all SNR and regularization computations for standard L1 behavior.
    This is useful when you want to avoid the "all-or-nothing" behavior of the SNR-style loss.

    Input Shape:
        Accepts waveform tensors (time-domain audio) of any shape as long as they are batch-first.
        Recommended shapes:
        - [batch, time] for single-source audio
        - [batch, num_sources, time] for multi-source audio
        - [batch, num_sources, channels, time] for multi-channel multi-source audio

    Attributes:
        name (str): The name identifier for the loss.
        weight (float): The overall weight multiplier for the loss.
        lambda0 (float): Minimum regularization weight (λ_min).
        delta_lambda (float): Range of extra weight for regularization (Δλ).
        l1snr_eps (float): Epsilon value for the L1SNR component to avoid log(0).
        dbrms_eps (float): Epsilon value for dBRMS calculation to avoid log(0).
        lmin (float): Minimum dBRMS considered non-silent for adaptive weighting.
        use_regularization (bool): Whether to use level-matching regularization.
            If False, only the L1SNR (and optional L1) components are used.
        l1_weight (float): Weight for the L1 loss component. Default 0 (disabled).
            As this increases, the regularization term is also scaled down proportionally.
            When set to 1.0, efficiently computes only L1 loss.
        reg_coef (float): Coefficient scaling the level-matching regularization, applied on top of
            (1 - l1_weight). Default 1.0, the historical behaviour. Exists so STFTL1SNRDBLoss can
            give its short-audio fallback the same regularizer weight as its own spectral path
            (spec_reg_coef), keeping that weight continuous across the fallback boundary instead of
            jumping by 10x.
        ref_level (float): Typical mean-absolute amplitude of your targets, used to scale the L1
            term when l1_weight > 0. Default 0.05, the measured median for MUSDB-like stems.
            Only affects the blended path: at l1_weight=0.0 or 1.0 it is unused.
            Replaces a per-batch statistic. Previously the scale was
            c * mean_b(1 / (mean|y|_b + eps)), a mean of reciprocals, so one quiet target inflated
            the gradient for every other sample in the batch (measured up to 5.73x) and l1_weight
            meant different things from run to run (0.5 delivering 15% to 91% of the way toward L1
            depending on batch content). To set it for your data, measure the mean absolute value
            of your targets over a few batches. Being off by 2x moves the knob about 5 points;
            being off by 10x matters.
    """
    def __init__(
        self,
        name,
        weight: float = 1.0,
        lambda0: float = 0.1,
        delta_lambda: float = 0.9,
        l1snr_eps: float = 1e-3,
        dbrms_eps: float = 1e-8,
        lmin: float = -60.0,
        use_regularization: bool = True,
        l1_weight: float = 0.0,
        reg_coef: float = 1.0,
        ref_level: float = 0.05,
    ):
        super().__init__()
        _validate_ref_level(ref_level)
        self.ref_level = ref_level
        self.name = name
        _validate_non_negative("weight", weight)
        self.weight = weight
        _validate_non_negative("reg_coef", reg_coef)
        _validate_non_negative("lambda0", lambda0)
        _validate_non_negative("delta_lambda", delta_lambda)
        _validate_positive("l1snr_eps", l1snr_eps)
        _validate_positive("dbrms_eps", dbrms_eps)
        self.reg_coef = reg_coef
        self.lambda0 = lambda0          # minimum regularization weight
        self.delta_lambda = delta_lambda # range of extra weight
        self.l1snr_eps = l1snr_eps
        self.dbrms_eps = dbrms_eps
        self.lmin = lmin
        self.use_regularization = use_regularization

        _validate_unit_range("l1_weight", l1_weight)
        self._l1_weight = l1_weight

        # Initialize component losses based on l1_weight
        if self.l1_weight == 1.0:
            # Pure L1 mode - only need L1 loss
            self.l1snr_loss: Optional[L1SNRLoss] = None
            self.l1_loss: Optional[torch.nn.Module] = torch.nn.L1Loss()
        else:
            # Standard mode with L1SNR (and optional weighted L1 if l1_weight > 0)
            self.l1snr_loss = L1SNRLoss(
                name="l1_snr",
                weight=1.0,  # We'll apply the weight at the end
                eps=l1snr_eps,
                l1_weight=l1_weight,
                ref_level=ref_level,
            )
            self.l1_loss = None

    @property
    def l1_weight(self):
        """Read-only. The value is baked into child modules and mode flags at construction, so mutating it
        afterwards took effect inconsistently (Q13). Construct a new loss instead."""
        return self._l1_weight

    @l1_weight.setter
    def l1_weight(self, value):
        raise AttributeError(
            "l1_weight is read-only: it is baked into child modules and mode flags when the loss is "
            "constructed, so assigning to it would take effect inconsistently. Construct a new loss with "
            "the l1_weight you want."
        )

    @staticmethod
    def compute_adaptive_weight(L_pred, L_true, L_min, lambda0, delta_lambda, R):
        """
        Implements the adaptive weighting of the level-matching regularization term, per arXiv:2501.16171.

        The citation covers the *form* of the weighting only. The paper specifies no numeric values for
        lambda0, delta_lambda or lmin, and no official implementation of it has been released, so the
        defaults used here (0.1, 0.9, -60.0) are a local choice rather than paper-derived constants.

        Args:
            L_pred: predicted dBRMS, shape (batch,)
            L_true: reference dBRMS, shape (batch,)
            L_min: minimum dBRMS considered non-silent (float)
            lambda0: minimum weight for regularization
            delta_lambda: range of extra weight for regularization
            R: |L_pred - L_true|, shape (batch,)
        Returns:
            lambda_weight: shape (batch,)
        """
        # Compute eta: 1 if L_true > max(L_pred, L_min), else 0
        max_val = torch.max(L_pred, torch.full_like(L_true, L_min))
        eta = (L_true > max_val).float()
        denom = (L_true - L_min).clamp(min=1e-6)
        clamp_arg = (R / denom).clamp(0.0, 1.0)
        lam = lambda0 + eta * delta_lambda * clamp_arg
        return lam.detach()  # Stop-gradient

    def forward(self, estimates, actuals, *args, **kwargs):
        _validate_matching_shapes(estimates, actuals)
        _validate_input_tensors(estimates, actuals)
        batch_size = estimates.shape[0]

        est_source = estimates.reshape(batch_size, -1)
        act_source = actuals.reshape(batch_size, -1)

        # Pure L1 mode - efficient path that bypasses SNR and regularization
        if self.l1_loss is not None:
            l1_loss = self.l1_loss(est_source, act_source)
            return l1_loss * self.weight

        # Standard mode with L1SNR, regularization, and optional weighted L1
        # 1. L1SNR reconstruction loss (with L1 component if l1_weight > 0)
        l1snr_loss = self.l1snr_loss(estimates, actuals, *args, **kwargs)

        # Only compute and apply regularization if enabled
        if self.use_regularization:
            # 2. Level-matching regularization
            L_true = dbrms(act_source, self.dbrms_eps)   # (batch,)
            L_pred = dbrms(est_source, self.dbrms_eps)   # (batch,)
            R = torch.abs(L_pred - L_true)               # (batch,)

            lambda_weight = self.compute_adaptive_weight(L_pred, L_true, self.lmin, self.lambda0, self.delta_lambda, R)  # (batch,)

            reg_loss = lambda_weight * R

            # Scale regularization by the same factor as L1SNR loss
            l1snr_weight = 1.0 - self.l1_weight
            total_loss = l1snr_loss + (l1snr_weight * self.reg_coef * torch.mean(reg_loss))
        else:
            # Skip regularization calculation entirely
            total_loss = l1snr_loss

        return total_loss * self.weight


class STFTL1SNRDBLoss(torch.nn.Module):
    """
    Implements L1SNR plus adaptive level-matching regularization in the spectrogram domain
    as described in arXiv:2501.16171, with multi-resolution STFT and optional L1 loss component
    to balance "all-or-nothing" behavior.

    This loss applies the same principles as L1SNRDBLoss but operates in the complex
    spectrogram domain across multiple time-frequency resolutions. For each resolution:

    1. L1SNR loss: Computed on the complex STFT representation (real/imaginary parts)
    2. Level-matching regularization: Applied to STFT magnitudes with adaptive weighting
    3. Optional L1 loss: Direct L1 penalty on STFT differences

    Multi-resolution processing helps capture both fine temporal details and frequency
    characteristics. The loss averages results across all valid STFT resolutions.

    The complete loss structure is similar to L1SNRDBLoss:
    When l1_weight < 1.0: total_loss = l1snr_loss + (1-l1_weight) * spec_reg_coef * mean(reg_loss)
    When l1_weight = 1.0: total_loss = l1_loss (pure L1 in spectrogram domain, bypassing all other computations)

    When l1_weight=1.0, this loss efficiently switches to a pure L1 loss calculation in the
    spectrogram domain, bypassing all SNR and regularization computations for standard L1 behavior.
    This is useful when you want to avoid the "all-or-nothing" behavior of the SNR-style loss.

    Note: PyTorch's MPS backend produces numerically incorrect gradients from torch.stft
    backward above an input length of 65,536 samples (2^16). The forward transform is correct to
    float32 precision, so the failure is silent. The error is not a simple function of size: a
    few specific lengths are exact while neighbours are wrong by 30-99%, and any batch size above
    1 fails even at those lengths. When mps_cpu_fallback=True (default), STFT computation is
    routed through CPU to avoid this issue. The .cpu() call is differentiable, so gradients
    flow correctly through CPU kernels. Performance impact is minimal since the STFT loss
    computation is small relative to model forward/backward. Typical training windows are well
    above the threshold (6 s at 44.1 kHz is 264,600 samples), so leave the fallback enabled on
    Apple silicon unless you have verified the backward pass on your own PyTorch version.

    Input Shape:
        Accepts waveform tensors (time-domain audio) of any shape as long as they are batch-first
        and time-last. Recommended shapes:
        - [batch, time] for single-source audio
        - [batch, num_sources, time] for multi-source audio
        - [batch, num_sources, channels, time] for multi-channel multi-source audio

    Attributes:
        name (str): The name identifier for the loss.
        weight (float): The overall weight multiplier for the loss.
        lambda0 (float): Minimum regularization weight (λ_min).
        delta_lambda (float): Range of extra weight for regularization (Δλ).
        l1snr_eps (float): Epsilon value for the L1SNR component to avoid log(0).
        dbrms_eps (float): Epsilon value for dBRMS calculation to avoid log(0).
        lmin (float): Minimum dBRMS considered non-silent for adaptive weighting.
        n_ffts (List[int]): List of FFT sizes for multi-resolution STFT analysis.
        hop_lengths (List[int]): List of hop lengths (STFT time steps) for each resolution.
        win_lengths (List[int]): List of window lengths for each resolution.
        window_fn (str): Window function for the STFT. One of 'hann', 'hamming', 'blackman',
            'bartlett' or 'kaiser'; any other value raises ValueError.
        min_audio_length (int): Minimum audio length required for STFT processing. If audio is
            shorter, the loss falls back to a time-domain L1SNR computation rather than failing.
            Note that the fallback is a different objective: it returns a single time-domain D1
            where the STFT path sums a real and an imaginary term, so values shift by roughly 2x
            across the boundary.
        spec_reg_coef (float): Coefficient scaling the spectrogram-domain level-matching
            regularization, applied on top of (1 - l1_weight). Only used when
            use_regularization=True. Has no counterpart in [2]; it is a local choice for keeping
            the spectral regularizer subordinate to the reconstruction term.
        use_regularization (bool): Whether to use level-matching regularization.
            If False, only the L1SNR (and optional L1) components are used.
        l1_weight (float): Weight for the L1 loss component. Default 0 (disabled).
            As this increases, the regularization term is also scaled down proportionally.
            When set to 1.0, efficiently computes only L1 loss.
        mps_cpu_fallback (bool): When True (default), routes STFT computation through
            CPU on MPS devices to avoid incorrect gradients from a PyTorch MPS backend
            bug in torch.stft backward above an input length of 65,536 samples.
        ref_level (float): Typical mean-absolute amplitude of your targets, used to scale the L1
            term when l1_weight > 0. Default 0.05, the measured median for MUSDB-like stems.
            Only affects the blended path: at l1_weight=0.0 or 1.0 it is unused.
            Replaces a per-batch statistic. Previously the scale was
            c * mean_b(1 / (mean|y|_b + eps)), a mean of reciprocals, so one quiet target inflated
            the gradient for every other sample in the batch (measured up to 5.73x) and l1_weight
            meant different things from run to run (0.5 delivering 15% to 91% of the way toward L1
            depending on batch content). To set it for your data, measure the mean absolute value
            of your targets over a few batches. Being off by 2x moves the knob about 5 points;
            being off by 10x matters.
        spec_ref_level (float): Same idea for the spectrogram domain. Leave as None to derive it as
            0.19 * ref_level, which is the measured median ratio of normalized-STFT reference
            magnitude to time-domain magnitude over 496 real MUSDB stem-chunks. Do not simply set
            it equal to ref_level: the STFT reference is about 5.6x lower, and that error costs
            roughly 20 points of knob position at l1_weight=0.5. The spectrogram knob is 2-3x more
            sensitive to this than the time-domain one.
        check_finite (bool): When True (default), scan the inputs for NaN and Inf each call and
            sanitize them, warning once: NaN becomes 0.0 and +/-Inf becomes +/-1.0, which is full-scale
            audio rather than silence. Costs four full-tensor scans whose results are
            consumed by a Python `if`, which on CUDA forces a host-device synchronization and
            serializes the pipeline. Measured at roughly 3 ms of a 344 ms CPU forward on
            [8, 2, 264600]. Set False once you trust your data pipeline to be finite; the loss will
            then propagate NaN rather than hiding it, which is arguably the better default for
            training anyway.
    """
    def __init__(
        self,
        name,
        weight: float = 1.0,
        lambda0: float = 0.1,
        delta_lambda: float = 0.9,
        l1snr_eps: float = 1e-3,
        dbrms_eps: float = 1e-8,
        lmin: float = -60.0,
        n_ffts: List[int] = [512, 1024, 2048],
        hop_lengths: List[int] = [128, 256, 512],
        win_lengths: List[int] = [512, 1024, 2048],
        window_fn: str = 'hann',
        min_audio_length: int = 512,
        use_regularization: bool = False,
        spec_reg_coef: float = 0.1,
        l1_weight: float = 0.0,
        mps_cpu_fallback: bool = True,
        ref_level: float = 0.05,
        spec_ref_level: Optional[float] = None,
        check_finite: bool = True,
    ):
        super().__init__()
        _validate_ref_level(ref_level)
        if spec_ref_level is not None:
            _validate_ref_level(spec_ref_level)
        self.check_finite = check_finite
        self.ref_level = ref_level
        self.spec_ref_level = spec_ref_level
        # Derived rather than defaulted to ref_level: the normalized-STFT reference is 5.6x below the
        # time-domain one, so reusing ref_level here would be 5.3x too large. Deriving also means a user who
        # sets ref_level for quieter audio gets the spectrogram side right for free.
        self._resolved_spec_ref_level = (
            spec_ref_level if spec_ref_level is not None else _STFT_REF_RATIO * ref_level
        )
        self.name = name
        _validate_non_negative("weight", weight)
        self.weight = weight
        self.min_audio_length = min_audio_length

        _validate_stft_params(n_ffts, hop_lengths, win_lengths, window_fn)

        # Copy: a list default is created once at function-definition time, so storing it by reference
        # makes every instance alias the same object and leaks in-place mutation across them.
        self.n_ffts = list(n_ffts)
        self.hop_lengths = list(hop_lengths)
        self.win_lengths = list(win_lengths)

        # Minimum input length each resolution needs. For center=True with pad_mode="reflect" the binding
        # constraint is length > n_fft // 2, which min_audio_length alone never expressed: at 512 samples
        # only 1 of the 3 default resolutions could run and at 513-1024 only 2, silently changing the arity
        # of the multi-resolution average.
        self._min_lengths = [n_fft // 2 + 1 for n_fft in self.n_ffts]
        self.window_fn_name = window_fn

        # Pre-initialize Spectrogram transforms for maximum efficiency
        self.spectrogram_transforms = nn.ModuleList()
        _base_window_fn = getattr(torch, f"{window_fn}_window")

        def window_fn_callable(length, _f=_base_window_fn):
            """The window, pre-scaled by 1 / ||w||_2.

            Paired with normalized=False below. torchaudio's normalized=True divides the whole complex
            output by the window's L2 norm on every call; folding that constant into the window instead is
            mathematically identical, because stft is linear in the window, and saves one full-tensor pass
            per resolution per tensor. Measured 2.21 ms of 19.25 ms per transform call, about 4% of the
            forward.

            Zero-padding a shorter window up to n_fft does not change its L2 norm, so this holds for
            win_length < n_fft too.

            Safe against the obvious future mistake: because torchaudio takes the norm of whatever window it
            is handed, and this one already has norm 1, setting normalized=True again would divide by 1.0.
            That is wasted work, not a silent rescale. Verified.
            """
            w = _f(length)
            # Normalize in float64 then cast back, so the stored window is the best representation of the
            # correct value rather than the result of a lower-precision division. Against a fully-float64
            # ground truth this measurably improves the float64 path rather than degrading it, which matters
            # because A9 exists to preserve float64. No absolute figure is quoted here: two careful
            # measurements with differently-constructed references disagreed by an order of magnitude, so
            # the ratio is a property of the comparison as much as of the code. What is robust, and gated in
            # tests/test_edge_cases.py, is that the stored window is closer to the exact normalized window
            # than a float32 division would be, and that the end-to-end error stays under a measured bound.
            w64 = w.double()
            return (w64 / w64.pow(2).sum().sqrt()).to(w.dtype)

        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            # Create a spectrogram transform for each resolution
            transform = Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                pad_mode="reflect",
                center=True,
                window_fn=window_fn_callable,
                normalized=False,   # the window is pre-normalized; see window_fn_callable
                power=None,  # This ensures the output is complex
            )
            # torchaudio registers `window` as a *persistent* buffer, so a ModuleList of Spectrograms
            # otherwise enters the enclosing model's state_dict. Re-register it non-persistent: the window is
            # derived deterministically from n_fft, so it is configuration rather than learned state, and
            # keeping it in checkpoints also breaks loading when n_ffts changes.
            window = transform.window
            del transform._buffers["window"]
            transform.register_buffer("window", window, persistent=False)
            self.spectrogram_transforms.append(transform)

        # Parameters for spectrogram domain level-matching
        _validate_non_negative("lambda0", lambda0)
        _validate_non_negative("delta_lambda", delta_lambda)
        self.lambda0 = lambda0
        self.delta_lambda = delta_lambda
        self.lmin = lmin
        _validate_positive("dbrms_eps", dbrms_eps)
        _validate_positive("l1snr_eps", l1snr_eps)
        self.dbrms_eps = dbrms_eps
        self.l1snr_eps = l1snr_eps

        _validate_unit_range("l1_weight", l1_weight)
        self._l1_weight = l1_weight

        # Flag for pure L1 mode
        self.pure_l1_mode = (self.l1_weight == 1.0)
        # Create L1 loss function for pure L1 mode
        self.l1_loss: Optional[torch.nn.Module] = torch.nn.L1Loss() if self.pure_l1_mode else None


        # Add this parameter to control regularization
        self.use_regularization = use_regularization
        # Coefficient to scale spectral regularization (disabled by default)
        _validate_non_negative("spec_reg_coef", spec_reg_coef)
        self.spec_reg_coef = spec_reg_coef

        # Fallback time-domain loss (used when audio is too short for TF processing)
        self.fallback_time_loss = L1SNRDBLoss(
            name=f"{name}_fallback_time",
            weight=1.0,
            lambda0=self.lambda0,
            delta_lambda=self.delta_lambda,
            l1snr_eps=self.l1snr_eps,
            dbrms_eps=self.dbrms_eps,
            lmin=self.lmin,
            # Pass the flag through rather than hardcoding False. Dropping it silently removed the
            # anti-collapse protection the user explicitly asked for, on exactly the inputs where a silent
            # estimate scores zero. reg_coef is matched to spec_reg_coef so the regularizer's weight is
            # continuous across the fallback boundary: the STFT path scales by spec_reg_coef, and
            # L1SNRDBLoss had no coefficient at all, so a naive pass-through would be 10x stronger.
            use_regularization=use_regularization,
            reg_coef=spec_reg_coef,
            l1_weight=self.l1_weight,
            ref_level=ref_level,
        )

        # MPS CPU fallback for correct gradients
        self.mps_cpu_fallback = mps_cpu_fallback
        self._mps_warned = False
        self._nan_warned = False
        self._fallback_warned = False
        self._dropped_warned = False
        self._allfailed_warned = False


    @property
    def l1_weight(self):
        """Read-only. The value is baked into child modules and mode flags at construction, so mutating it
        afterwards took effect inconsistently (Q13). Construct a new loss instead."""
        return self._l1_weight

    @l1_weight.setter
    def l1_weight(self, value):
        raise AttributeError(
            "l1_weight is read-only: it is baked into child modules and mode flags when the loss is "
            "constructed, so assigning to it would take effect inconsistently. Construct a new loss with "
            "the l1_weight you want."
        )

    def _compute_complex_spec_l1snr_loss(self, est_spec, act_spec):
        """
        Compute TF-domain loss as per the papers:
        - D1 on real part + D1 on imaginary part, summed.
        - Optional L1 mixing applied symmetrically to Re/Im.
        est_spec, act_spec: complex tensors with shape (B, C, F, T)
        """
        # Ensure same shape (assert to avoid silent mismatches)
        assert est_spec.shape == act_spec.shape, f"Spec shapes must match: {est_spec.shape} vs {act_spec.shape}"

        # Split real/imag
        est_re, est_im = est_spec.real, est_spec.imag
        act_re, act_im = act_spec.real, act_spec.imag

        B = est_spec.shape[0]

        # Flatten to (B, -1)
        est_re = est_re.reshape(B, -1)
        act_re = act_re.reshape(B, -1)
        est_im = est_im.reshape(B, -1)
        act_im = act_im.reshape(B, -1)

        # L1 errors and refs
        #
        # A second optimization deliberately NOT taken. Reducing over the non-batch dims directly, instead of
        # reshaping these strided views, avoids a contiguous copy (~129 MB per forward on a realistic batch).
        # The forward result is bit-identical, but the backward graph is not: reshape-then-mean(dim=1) and
        # mean(dim=(1,2,3)) accumulate differently, giving gradient differences at machine epsilon
        # (relative 1.6e-07 in float32, 3.0e-16 in float64). Measured time saving was 0.2 ms of a 344 ms
        # forward, about 0.06%. Declined on the same grounds as the window-normalization fold above: a
        # negligible gain does not justify changing the numbers a training run reports.
        err_re = torch.mean(torch.abs(est_re - act_re), dim=1)
        ref_re = torch.mean(torch.abs(act_re), dim=1)
        err_im = torch.mean(torch.abs(est_im - act_im), dim=1)
        ref_im = torch.mean(torch.abs(act_im), dim=1)

        # D1 = 10*log10((mean|e| + eps)/(mean|y| + eps)); means, not sums -- see L1SNRLoss
        d1_re = 10.0 * torch.log10((err_re + self.l1snr_eps) / (ref_re + self.l1snr_eps))
        d1_im = 10.0 * torch.log10((err_im + self.l1snr_eps) / (ref_im + self.l1snr_eps))
        d1_sum = torch.mean(d1_re + d1_im)  # mean over batch

        # Pure L1 mode
        if self.pure_l1_mode:
            l1_re = torch.mean(err_re)
            l1_im = torch.mean(err_im)
            l1_term = 0.5 * (l1_re + l1_im)
            return l1_term

        # Mixed mode (auto-balanced L1/SNR) with per-batch scaling
        w = float(self.l1_weight)
        if 0.0 < w < 1.0:
            c = 10.0 / math.log(10.0)
            # A fixed spectrogram reference level, for the same reasons as the time domain. The factor 2
            # accounts for d1_sum carrying both a real and an imaginary term.
            scale_spec = 2.0 * c / (self._resolved_spec_ref_level + self.l1snr_eps)
            l1_term = 0.5 * (torch.mean(err_re) + torch.mean(err_im)) * scale_spec

            loss = (1.0 - w) * d1_sum + w * l1_term
            return loss
        else:
            # Pure SNR (D1). There is no w >= 1.0 arm here: pure_l1_mode returns above, so it was
            # unreachable, which 0% coverage on it confirmed.
            return d1_sum

    def _compute_spec_level_matching(self, est_spec, act_spec):
        """
        Compute the level matching regularization term for a spectrogram.
        """
        batch_size = est_spec.shape[0]

        # No shape reconciliation here. forward now requires estimates and actuals to match, so both
        # spectrograms are computed from equal-shaped inputs by the same transform and cannot differ. The
        # crop that used to live here was already dead because the caller had cropped first.

        # For level-matching regularization, we use magnitude information
        est_mag = torch.abs(est_spec)
        act_mag = torch.abs(act_spec)

        # Reshape once for efficiency
        est_mag_flat = est_mag.reshape(batch_size, -1)
        act_mag_flat = act_mag.reshape(batch_size, -1)

        # Calculate dB levels
        L_true = dbrms(act_mag_flat, self.dbrms_eps)
        L_pred = dbrms(est_mag_flat, self.dbrms_eps)

        R = torch.abs(L_pred - L_true)

        # Use the adaptive weighting function
        lambda_weight = L1SNRDBLoss.compute_adaptive_weight(
            L_pred, L_true, self.lmin, self.lambda0, self.delta_lambda, R
        )

        return torch.mean(lambda_weight * R)

    def _usable_resolutions(self, audio_length):
        """Indices of the resolutions this input length can actually support.

        Returns them rather than a single yes/no so a short input keeps whichever resolutions work instead
        of discarding all of them: at 600 samples, two of the three defaults are perfectly usable and
        falling back wholesale would trade two working resolutions for none.
        """
        return [
            i for i, (min_len, hop) in enumerate(zip(self._min_lengths, self.hop_lengths))
            if audio_length >= min_len and (audio_length // hop) + 1 >= 2
        ]

    def forward(self, estimates, actuals, *args, **kwargs):
        _validate_matching_shapes(estimates, actuals)
        _validate_input_tensors(estimates, actuals)
        device = estimates.device
        batch_size = estimates.shape[0]

        # Sanitize non-finite input. Deliberate, but it must not be silent: an all-NaN estimate otherwise
        # yields exactly 0.0 with an all-zero gradient, so a diverged run looks healthy.
        if self.check_finite and (
            torch.isnan(estimates).any() or torch.isinf(estimates).any()
            or torch.isnan(actuals).any() or torch.isinf(actuals).any()
        ):
            if not self._nan_warned:
                warnings.warn(
                    f"{self.name}: non-finite values found in the loss input and sanitized (NaN to 0.0, "
                    "+/-Inf to +/-1.0, i.e. full scale). The loss and its gradient are computed on the "
                    "sanitized tensors, so a "
                    "diverging model can produce a healthy-looking zero loss. This warning is issued once "
                    "per loss instance; check your model and data if it appears.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._nan_warned = True
            estimates = torch.nan_to_num(estimates, nan=0.0, posinf=1.0, neginf=-1.0)
            actuals = torch.nan_to_num(actuals, nan=0.0, posinf=1.0, neginf=-1.0)

        est_source = estimates.reshape(batch_size, -1, estimates.shape[-1])
        act_source = actuals.reshape(batch_size, -1, actuals.shape[-1])

        # Validate audio length
        audio_length = est_source.shape[-1]
        usable = self._usable_resolutions(audio_length) if audio_length >= self.min_audio_length else []
        if usable and len(usable) < len(self.n_ffts):
            if not self._dropped_warned:
                dropped = [self.n_ffts[i] for i in range(len(self.n_ffts)) if i not in usable]
                warnings.warn(
                    f"{self.name}: input length {audio_length} is too short for resolution(s) with "
                    f"n_fft={dropped} (each needs at least n_fft//2 + 1 samples). The loss is the average "
                    f"over the {len(usable)} usable resolution(s) with "
                    f"n_fft={[self.n_ffts[i] for i in usable]}, so it is not comparable with a run where "
                    "all resolutions apply. Warned once per loss instance.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._dropped_warned = True
        if not usable:
            # Fallback to a time-domain loss. This is a *different objective*, not a degraded version of the
            # same one: it returns a single time-domain D1 where the STFT path sums a real and an imaginary
            # term. Inside MultiL1SNRDBLoss it also makes both branches the same quantity, so the user is
            # optimizing one domain at effective weight 1.0 while believing otherwise. Say so once.
            if not self._fallback_warned:
                # Name the constraint that actually bit. This branch is reached either because the input is
                # below min_audio_length or because no configured resolution fits it, and those want different
                # fixes. It used to report the min_audio_length text unconditionally, so 1024 samples with
                # n_ffts=[2048, 4096] read "input length 1024 is below min_audio_length (512)" -- pointing at
                # a threshold the input clears, and at the wrong knob. Its neighbour case emits a correct
                # dropped-resolutions message, which made the wrong one look authoritative.
                if audio_length < self.min_audio_length:
                    cause = (f"input length {audio_length} is below min_audio_length "
                             f"({self.min_audio_length})")
                else:
                    need = min(self._min_lengths)
                    cause = (f"input length {audio_length} fits none of the configured resolutions "
                             f"(n_ffts={list(self.n_ffts)} need at least {need} samples); "
                             "min_audio_length is not the constraint here")
                warnings.warn(
                    f"{self.name}: {cause}, so the spectrogram loss is falling back to a time-domain "
                    "computation. This is a different objective, roughly 2x smaller in magnitude, and "
                    "inside MultiL1SNRDBLoss it makes the time and spectral branches identical. Warned "
                    "once per loss instance.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._fallback_warned = True
            return self.fallback_time_loss(estimates, actuals, *args, **kwargs) * self.weight

        # MPS workaround: route STFT computation through CPU to avoid
        # incorrect gradients from torch.stft backward above ~65,536 samples.
        # .cpu() is differentiable — backward uses CPU kernels automatically.
        _use_cpu = self.mps_cpu_fallback and device.type == 'mps'
        if _use_cpu:
            if not self._mps_warned:
                warnings.warn(
                    f"{self.name}: Routing STFT loss through CPU to work around "
                    "PyTorch MPS backend bug in torch.stft backward above an input length of "
                    "65,536 samples. "
                    "Set mps_cpu_fallback=False to disable.",
                    stacklevel=2,
                )
                self._mps_warned = True
            est_source = est_source.cpu()
            act_source = act_source.cpu()

        # Device for STFT computation (CPU if fallback active, else original)
        compute_device = est_source.device

        # Accumulate in the input's own dtype. An untyped torch.tensor(0.0) is float32, which silently
        # downcast float64 input while the time-domain siblings preserved it. Half precision promotes to
        # float32 so accumulation does not happen in half.
        acc_dtype = est_source.dtype if est_source.dtype in (torch.float32, torch.float64) else torch.float32
        total_spec_loss = torch.zeros((), dtype=acc_dtype, device=compute_device)
        total_spec_reg_loss = torch.zeros((), dtype=acc_dtype, device=compute_device)
        valid_transforms = 0

        # Ensure transforms are on the correct device. Unconditional, deliberately.
        #
        # T4-5 guarded this against a cached `self._transforms_device` to save the submodule walk. That was
        # worth 0.02 ms, about 0.007% of a forward, and it introduced a silent-zero bug: nn.Module.to() moves
        # the window buffers but cannot update a plain Python attribute, so after `loss.to(device)` the cache
        # still said "cpu" while the buffers had moved. The guard then skipped a move that was needed, every
        # resolution failed on a device mismatch, and the spectral loss became exactly 0.0 with a zero
        # gradient. On Apple silicon that is the *default* path, because mps_cpu_fallback moves the input to
        # CPU and so `compute_device` is cpu while the buffers are on MPS.
        #
        # A truthful guard is possible -- compare against `self.spectrogram_transforms[0].window.device`
        # rather than a cache -- but 0.007% does not justify carrying an invariant that can silently zero a
        # training signal. Reverted.
        self.spectrogram_transforms.to(compute_device)

        # Process each usable resolution. The try/except blocks below remain as a backstop, but with the
        # requirement checked up front they should no longer be the mechanism that decides the arity.
        for i in usable:
            transform = self.spectrogram_transforms[i]
            try:
                # Compute spectrograms using pre-initialized transforms
                try:
                    est_spec = transform(est_source)
                    act_spec = transform(act_source)
                except RuntimeError as e:
                    warnings.warn(
                        f"Error computing spectrogram for resolution {i}: {e}. "
                        f"Parameters: n_fft={self.n_ffts[i]}, hop_length={self.hop_lengths[i]}, win_length={self.win_lengths[i]}"
                    )
                    continue

                # Ensure same (B, C, F, T); crop only (F, T) if needed
                if est_spec.shape != act_spec.shape:
                    min_f = min(est_spec.shape[-2], act_spec.shape[-2])
                    min_t = min(est_spec.shape[-1], act_spec.shape[-1])
                    est_spec = est_spec[..., :min_f, :min_t]
                    act_spec = act_spec[..., :min_f, :min_t]

                # Compute complex spectral loss (either L1 or L1SNR based on self.pure_l1_mode)
                try:
                    spec_loss = self._compute_complex_spec_l1snr_loss(est_spec, act_spec)
                except RuntimeError as e:
                    warnings.warn(f"Error computing complex spectral loss for resolution {i}: {e}")
                    continue

                # Check for numerical issues
                if torch.isnan(spec_loss) or torch.isinf(spec_loss):
                    continue

                # Only compute regularization if not in pure L1 mode and regularization is enabled
                if not self.pure_l1_mode and self.use_regularization:
                    try:
                        spec_reg_loss = self._compute_spec_level_matching(est_spec, act_spec)

                        # Check for numerical issues
                        if torch.isnan(spec_reg_loss) or torch.isinf(spec_reg_loss):
                            spec_reg_loss = 0.0  # Use zero reg_loss if there are issues

                        # Accumulate regularization loss
                        total_spec_reg_loss += spec_reg_loss
                    except RuntimeError as e:
                        warnings.warn(f"Error computing spectral level-matching for resolution {i}: {e}")

                # Accumulate loss
                total_spec_loss += spec_loss
                valid_transforms += 1

            except RuntimeError as e:
                warnings.warn(f"Runtime error in spectrogram transform {i}: {e}")
                continue

        # If all transforms failed, return a zero that is still attached to the graph. A bare
        # torch.tensor(0.0) has no grad_fn, so .backward() raises here and, worse, silently contributes
        # nothing inside MultiL1SNRDBLoss while the time term goes on training.
        if valid_transforms == 0 and not self._allfailed_warned:
            self._allfailed_warned = True
            warnings.warn(
                f"{self.name}: every spectrogram resolution failed, so the spectral loss is zero and "
                "carries no learning signal. Check n_ffts against your input length, that both inputs are "
                "finite, and that they are on the same device as this module's window buffers.",
                RuntimeWarning,
                stacklevel=2,
            )
        if valid_transforms == 0:
            # Sum BOTH inputs, not just the estimate. Multiplying a sum by 0.0 propagates non-finiteness
            # (nan * 0 and inf * 0 are both nan), which is what makes this zero honest under
            # check_finite=False -- but only for whichever tensor is in the expression. With est_source
            # alone, a NaN in the *target* left every resolution NaN, dropped them all, and returned a clean
            # -0.0 with a zero gradient: a corrupt target read as a perfectly healthy step. The estimate's
            # sum still supplies the graph connection; the target's contributes only its finiteness.
            zero = (est_source.sum() + act_source.sum().detach()) * 0.0
            return zero.to(device) * self.weight

        # Average losses across valid transforms
        avg_spec_loss = total_spec_loss / valid_transforms

        # For standard mode, apply regularization if enabled
        if not self.pure_l1_mode and self.use_regularization:
            avg_spec_reg_loss = total_spec_reg_loss / valid_transforms
            # Scale spectral regularization by both (1 - l1_weight) and spec_reg_coef
            l1snr_weight = 1.0 - self.l1_weight
            final_loss = avg_spec_loss + l1snr_weight * (self.spec_reg_coef * avg_spec_reg_loss)
        else:
            final_loss = avg_spec_loss

        result = final_loss * self.weight

        # Move loss back to original device (gradient graph preserved)
        if _use_cpu:
            result = result.to(device)

        return result


class MultiL1SNRDBLoss(torch.nn.Module):
    """
    A modular loss function that combines time-domain and spectrogram-domain L1SNR and
    adaptive level-matching losses, as described in arXiv:2501.16171, with optional
    L1 loss component to balance "all-or-nothing" behavior.

    This implementation uses separate specialized components:
    - L1SNRDBLoss for time domain processing
    - STFTL1SNRDBLoss for spectrogram domain processing

    The loss combines time-domain and spectrogram-domain losses:
    Loss = weight * [(1-spec_weight) * time_loss + spec_weight * spec_loss]

    Where time_loss and spec_loss are computed by L1SNRDBLoss and STFTL1SNRDBLoss respectively,
    each handling their own L1SNR, regularization, and optional L1 components as described
    in their individual docstrings.

    When l1_weight=1.0, this loss efficiently switches to a pure L1 loss calculation in both
    domains, bypassing all SNR and regularization computations for standard L1 behavior.
    This is useful when you want to avoid the "all-or-nothing" behavior of the SNR-style loss.

    The regularization components use adaptive weighting based on level differences
    between estimated and target signals, with weighting controlled by lambda0 and delta_lambda.

    Input Shape:
        Accepts waveform tensors (time-domain audio) of any shape as long as they are batch-first
        and time-last. Recommended shapes:
        - [batch, time] for single-source audio
        - [batch, num_sources, time] for multi-source audio
        - [batch, num_sources, channels, time] for multi-channel multi-source audio

    Attributes:
        name (str): The name identifier for the loss.
        weight (float): The overall weight multiplier for the loss.
        spec_weight (float): Coefficient on the spectrogram-domain loss, in [0.0, 1.0]. The time
            domain receives (1 - spec_weight). Values outside that range raise ValueError:
            above 1.0 the time coefficient would be negative, which instructs the optimizer to
            maximize time-domain error.
            Default 0.5, which is the paper-faithful choice rather than merely "equal": it is the
            value at which the time, real and imaginary terms all carry weight 0.5, reproducing
            the 1:1:1 weighting of arXiv:2406.18747. Note this is a loss-value weight, not a
            gradient share - spec_loss sums a real and an imaginary D1 term while time_loss is a
            single term, so equal coefficients are not equal gradient contribution.
        use_time_regularization (bool): Whether to use level-matching regularization in time domain.
        use_spec_regularization (bool): Whether to use level-matching regularization in spectogram domain.
        l1_weight (float): Weight for the L1 loss component vs the L1SNR+reg components.
            Default 0 (disabled). As this increases, the regularization term is also scaled down.
            When set to 1.0, efficiently computes only L1 loss in both domains.
        lambda0 (float): Minimum regularization weight for both domains. A local choice, not a
            paper-specified value -- see L1SNRDBLoss.compute_adaptive_weight.
        delta_lambda (float): Range of extra weight for regularization in both domains. Also a
            local choice.
        l1snr_eps (float): Epsilon for the L1SNR components in both domains, guarding log(0).
            Note this sets D1's floor as well as its numerical stability: see the dynamic-range
            limitation in the README.
        dbrms_eps (float): Epsilon for the dBRMS calculation in both domains, guarding log(0).
        lmin (float): Minimum dBRMS considered non-silent for adaptive weighting, in both domains.
        n_ffts (List[int]): FFT sizes for the spectrogram domain's multi-resolution analysis.
        hop_lengths (List[int]): Hop lengths for each spectrogram resolution.
        win_lengths (List[int]): Window lengths for each spectrogram resolution.
        window_fn (str): Window function for the STFT ('hann', 'hamming', 'blackman', 'bartlett',
            or 'kaiser').
        min_audio_length (int): Minimum audio length for STFT processing. Below this the
            spectrogram branch falls back to a time-domain loss, which makes both branches the
            same quantity -- see the note in STFTL1SNRDBLoss.
        spec_reg_coef (float): Coefficient scaling the spectrogram-domain regularization. Only
            used when use_spec_regularization=True.
        mps_cpu_fallback (bool): When True (default), routes the spectrogram branch's STFT through
            CPU on MPS devices to work around incorrect torch.stft backward gradients.
        check_finite (bool): Passed to the spectrogram branch. When True (default), its inputs are
            scanned for NaN and Inf each call and sanitized, warning once: NaN to 0.0 and +/-Inf to
            +/-1.0. See
            STFTL1SNRDBLoss for the cost and for why False is defensible.
        time_loss_params (dict): Optional additional parameters to pass to time domain loss.
            Overrides any of the above for the time-domain component only. Ignored when
            time_loss_module is supplied.
        spec_loss_params (dict): Optional additional parameters to pass to spectrogram domain loss.
            Overrides any of the above for the spectrogram component only.
        time_loss_module (torch.nn.Module): Optional pre-built module replacing the built-in
            L1SNRDBLoss time-domain branch, so a different time-domain objective (e.g. L2SNRLoss) can
            be A/B'd without a second multi-domain class. Default None keeps the built-in loss and the
            behaviour of every prior version. When supplied, time_loss_params and
            use_time_regularization no longer apply and a warning says so; parameters shared with the
            spectrogram branch (lambda0, delta_lambda, l1snr_eps, dbrms_eps, lmin, l1_weight,
            ref_level) still work but reach only that branch. spec_weight and weight are unaffected.
            Three things to know, none of which the built-in branch exposes you to:
              * The module's OWN `weight` multiplies on top of `(1 - spec_weight) * weight`. The
                built-in branch is constructed with `weight=1.0` for exactly this reason, so leave an
                injected module's weight at 1.0 unless you intend the product.
              * `pure_l1_mode` and `l1_weight` describe the spectrogram branch only. `l1_weight=1.0`
                with an injected module warns, because `pure_l1_mode` then reads True while the time
                branch is not an L1 loss.
              * The module must return a scalar. A module returning one value per batch item makes
                `forward` return a non-scalar, which surfaces later as an error from `backward` rather
                than from here.
        ref_level (float): Typical mean-absolute amplitude of your targets, used to scale the L1
            term when l1_weight > 0. Default 0.05, the measured median for MUSDB-like stems.
            Only affects the blended path: at l1_weight=0.0 or 1.0 it is unused.
            Replaces a per-batch statistic. Previously the scale was
            c * mean_b(1 / (mean|y|_b + eps)), a mean of reciprocals, so one quiet target inflated
            the gradient for every other sample in the batch (measured up to 5.73x) and l1_weight
            meant different things from run to run (0.5 delivering 15% to 91% of the way toward L1
            depending on batch content). To set it for your data, measure the mean absolute value
            of your targets over a few batches. Being off by 2x moves the knob about 5 points;
            being off by 10x matters.
        spec_ref_level (float): Same idea for the spectrogram domain. Leave as None to derive it as
            0.19 * ref_level, which is the measured median ratio of normalized-STFT reference
            magnitude to time-domain magnitude over 496 real MUSDB stem-chunks. Do not simply set
            it equal to ref_level: the STFT reference is about 5.6x lower, and that error costs
            roughly 20 points of knob position at l1_weight=0.5. The spectrogram knob is 2-3x more
            sensitive to this than the time-domain one.
    """
    def __init__(
        self,
        name,
        weight: float = 1.0,
        spec_weight: float = 0.5,  # Balance between time and frequency domain
        # L1 component parameters
        l1_weight: float = 0.0, # Weight for the L1 loss component vs (L1SNR + Regularization).
                               # Note: Regularization term is also scaled by (1.0 - l1_weight).
                               # When set to 1.0, efficiently computes only L1 loss in both domains.
        # Regularization on/off flags
        use_time_regularization: bool = True,
        use_spec_regularization: bool = False, # likely redundant if already using in time domain
        # Default parameters for both loss components
        lambda0: float = 0.1,
        delta_lambda: float = 0.9,
        l1snr_eps: float = 1e-3,
        dbrms_eps: float = 1e-8,
        lmin: float = -60.0,
        # STFT parameters
        n_ffts: List[int] = [512, 1024, 2048],
        hop_lengths: List[int] = [128, 256, 512],
        win_lengths: List[int] = [512, 1024, 2048],
        window_fn: str = 'hann',
        min_audio_length: int = 512,
        # Position 17 in published 0.1.3, and it must stay here. Moving it to the end to "append new
        # parameters" was itself the break: it shifted time_loss_params from 18 to 17, so a full
        # 20-argument positional call written against 0.1.3 passed a float where a dict was expected.
        # 0.1.3 is the baseline every user actually has -- nothing before 0.2.0 may be reordered.
        spec_reg_coef: float = 0.1,
        # Allow for separate parameter overrides (e.g. different delta_lambda for time and spec)
        time_loss_params: Optional[dict] = None,
        spec_loss_params: Optional[dict] = None,
        # MPS workaround
        mps_cpu_fallback: bool = True,
        # New in 0.2.0, appended so that positional calls written against 0.1.3 keep their meaning.
        # Keep additions here, after every parameter 0.1.3 shipped.
        ref_level: float = 0.05,
        spec_ref_level: Optional[float] = None,
        check_finite: bool = True,
        # Opt-in replacement for the built-in time-domain sub-loss, for A/B-ing a different time-domain
        # objective (e.g. L2SNRLoss) against D1 without a second multi-domain class. Appended, like every
        # 0.2.0 addition, so positional calls written against 0.1.3 keep their meaning.
        time_loss_module: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        # spec_weight above 1 makes the time-domain coefficient (1 - spec_weight) negative in forward,
        # instructing the optimizer to MAXIMISE time-domain error. Never leave this unvalidated.
        _validate_unit_range("spec_weight", spec_weight)
        _validate_ref_level(ref_level)
        if spec_ref_level is not None:
            _validate_ref_level(spec_ref_level)
        self.ref_level = ref_level
        self.spec_ref_level = spec_ref_level
        self.name = name
        _validate_non_negative("weight", weight)
        self.weight = weight
        self.spec_weight = spec_weight

        _validate_unit_range("l1_weight", l1_weight)
        self._l1_weight = l1_weight
        self.use_time_regularization = use_time_regularization
        self.use_spec_regularization = use_spec_regularization

        # Set up default parameters
        default_time_params = {
            "name": f"{name}_time",
            "weight": 1.0,  # Will be scaled by the combined loss
            "lambda0": lambda0,
            "delta_lambda": delta_lambda,
            "l1snr_eps": l1snr_eps,
            "dbrms_eps": dbrms_eps,
            "lmin": lmin,
            "l1_weight": l1_weight,
            "ref_level": ref_level,
            "use_regularization": use_time_regularization  # Apply time domain regularization flag
        }

        default_spec_params = {
            "name": f"{name}_spec",
            "weight": 1.0,  # Will be scaled by the combined loss
            "lambda0": lambda0,
            "delta_lambda": delta_lambda,
            "l1snr_eps": l1snr_eps,
            "dbrms_eps": dbrms_eps,
            "lmin": lmin,
            "n_ffts": n_ffts,
            "hop_lengths": hop_lengths,
            "win_lengths": win_lengths,
            "window_fn": window_fn,
            "min_audio_length": min_audio_length,
            "spec_reg_coef": spec_reg_coef,
            "l1_weight": l1_weight,
            "ref_level": ref_level,
            "spec_ref_level": spec_ref_level,
            "mps_cpu_fallback": mps_cpu_fallback,
            "check_finite": check_finite,
            "use_regularization": use_spec_regularization  # Apply spectrogram domain regularization flag
        }

        # Override with any custom parameters
        if time_loss_params:
            default_time_params.update(time_loss_params)
        if spec_loss_params:
            default_spec_params.update(spec_loss_params)

        # Create the specialized loss components
        # Note: Component losses handle all optimizations internally based on l1_weight
        # When l1_weight=1.0, they will efficiently bypass SNR and regularization calculations
        if time_loss_module is None:
            self.time_loss = L1SNRDBLoss(**default_time_params)
        else:
            if not isinstance(time_loss_module, torch.nn.Module):
                raise ValueError(
                    f"time_loss_module must be a torch.nn.Module, got {type(time_loss_module).__name__}. "
                    "Pass an instantiated loss (e.g. L2SNRLoss('time')), not a class or a name."
                )
            # Only these two are time-exclusive. lambda0, delta_lambda, l1snr_eps, dbrms_eps, lmin,
            # l1_weight and ref_level are shared with the spectrogram branch and keep working, so warning
            # on them would fire on ordinary calls.
            if time_loss_params is not None or not use_time_regularization:
                warnings.warn(
                    "time_loss_module replaces the built-in time-domain loss, so time_loss_params and "
                    "use_time_regularization no longer have any effect. Parameters shared with the "
                    "spectrogram branch (lambda0, delta_lambda, l1snr_eps, dbrms_eps, lmin, l1_weight, "
                    "ref_level) still apply, but now reach only that branch.",
                    UserWarning,
                    stacklevel=2,
                )
            # l1_weight at either endpoint changes what the loss IS, not just how it is weighted, and an
            # injected module does not honour it. At 1.0 in particular `pure_l1_mode` reads True while the
            # time branch is whatever was injected, so the attribute is actively misleading rather than
            # merely inapplicable. Silence here would be the quiet kind of wrong.
            if l1_weight == 1.0:
                warnings.warn(
                    "l1_weight=1.0 puts the spectrogram branch in pure-L1 mode, but time_loss_module is "
                    "used as given and does not honour l1_weight. pure_l1_mode will read True while the "
                    "time branch is not an L1 loss. Set l1_weight=0.0 unless you want that asymmetry.",
                    UserWarning,
                    stacklevel=2,
                )
            self.time_loss = time_loss_module
        self.spec_loss = STFTL1SNRDBLoss(**default_spec_params)

        # For reference only, indicate if we're in pure L1 mode
        self.pure_l1_mode = (self.l1_weight == 1.0)

    @property
    def l1_weight(self):
        """Read-only. The value is baked into child modules and mode flags at construction, so mutating it
        afterwards took effect inconsistently (Q13). Construct a new loss instead."""
        return self._l1_weight

    @l1_weight.setter
    def l1_weight(self, value):
        raise AttributeError(
            "l1_weight is read-only: it is baked into child modules and mode flags when the loss is "
            "constructed, so assigning to it would take effect inconsistently. Construct a new loss with "
            "the l1_weight you want."
        )

    def forward(self, estimates, actuals, *args, **kwargs):
        """
        Forward pass to compute the combined multi-domain loss.

        Args:
            estimates: Model output predictions, shape [batch, ...] (batch-first, ..., time-last)
            actuals: Ground truth targets, shape [batch, ...] (batch-first, ..., time-last)
            *args, **kwargs: Additional arguments passed to sub-losses

        Returns:
            Combined weighted loss from time and spectrogram domains
        """
        _validate_matching_shapes(estimates, actuals)
        _validate_input_tensors(estimates, actuals)
        # Compute time domain loss
        time_loss = self.time_loss(estimates, actuals, *args, **kwargs)

        # Compute spectrogram domain loss
        spec_loss = self.spec_loss(estimates, actuals, *args, **kwargs)

        # Combine with weighting
        combined_loss = (1 - self.spec_weight) * time_loss + self.spec_weight * spec_loss

        # Apply overall weight
        return combined_loss * self.weight
