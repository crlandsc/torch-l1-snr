"""Independent reference implementations, derived from the definitions rather than from the library.

Audit finding M2: 112 of 127 tests compared StemWrappedLoss against base_loss with stem_dimension=None,
where the wrapper forwards straight through. The assertion was structurally f(x) == f(x) and could not fail
for any f. Fixing that needs a reference computed *without* calling library code.

**This module must not import torch_l1_snr.** A reference that calls the thing under test is not a reference.
tests/test_losses.py asserts that this file contains no such import.

Self-validation is the point. These functions are checked against HAND_CASES below -- values worked out on
paper for inputs small enough to verify by eye. That is what makes them trustworthy: not that they were
written carefully, but that they reproduce arithmetic anyone can redo.

One trap worth recording. torchaudio's Spectrogram(normalized=True) divides by the **window L2 norm**
(13.8564 for a 512-point Hann window), while torch.stft(normalized=True) divides by **sqrt(n_fft)**
(22.6274). They differ by a factor of 1.633. Using torch.stft(normalized=True) here would have made every
spectrogram reference wrong by that factor while looking perfectly independent.
"""
import math

import torch

C = 10.0 / math.log(10.0)   # 4.342944819..., d(10*log10(x))/d(ln x)


# ---------------------------------------------------------------------------------------------
# time domain
# ---------------------------------------------------------------------------------------------

def d1_per_element(est, act, eps=1e-3):
    """D1 = 10*log10((mean|est-act| + eps) / (mean|act| + eps)), one value per batch element.

    Means, not sums: this follows the authors' reference implementation, which the papers' written
    L1-norm notation does not convey.
    """
    b = est.shape[0]
    e = est.reshape(b, -1)
    a = act.reshape(b, -1)
    err = (e - a).abs().mean(dim=-1)
    ref = a.abs().mean(dim=-1)
    return 10.0 * torch.log10((err + eps) / (ref + eps))


def l1snr(est, act, eps=1e-3):
    """Batch-reduced L1SNR: mean over batch elements of D1."""
    return d1_per_element(est, act, eps).mean()


def l1snr_blended(est, act, eps=1e-3, l1_weight=0.0, ref_level=None):
    """The l1_weight blend, including the endpoint shortcuts the library takes.

    ref_level=None reproduces the pre-0.2.0 batch statistic (a mean of reciprocals). Passing a float
    reproduces the constant-K form. Both are expressed here from the definition so the test can pin either.
    """
    b = est.shape[0]
    e = est.reshape(b, -1)
    a = act.reshape(b, -1)
    err = (e - a).abs().mean(dim=-1)
    ref = a.abs().mean(dim=-1)

    if l1_weight >= 1.0:
        return err.mean()
    d1 = (10.0 * torch.log10((err + eps) / (ref + eps))).mean()
    if l1_weight <= 0.0:
        return d1
    if ref_level is None:
        scale = C * (1.0 / (ref.detach() + eps)).mean()
    else:
        scale = C / (ref_level + eps)
    return (1.0 - l1_weight) * d1 + l1_weight * (err.mean() * scale)


def dbrms(x, eps=1e-8):
    """20*log10(sqrt(mean(x^2) + eps) + eps), one value per batch element.

    Two epsilons of different physical dimension: one inside the sqrt on a power quantity, one outside on an
    amplitude. Reproduced faithfully because the library's behaviour, not its tidiness, is what is under test.
    """
    v = x.reshape(x.shape[0], -1)
    rms = torch.sqrt((v ** 2).mean(dim=-1) + eps)
    return 20.0 * torch.log10(rms + eps)


def adaptive_lambda(L_pred, L_true, lmin=-60.0, lambda0=0.1, delta_lambda=0.9):
    """eta = 1 if L_true > max(L_pred, lmin) else 0; lam = lambda0 + eta*delta_lambda*clamp(R/(L_true-lmin))."""
    R = (L_pred - L_true).abs()
    eta = (L_true > torch.maximum(L_pred, torch.full_like(L_true, lmin))).float()
    denom = (L_true - lmin).clamp(min=1e-6)
    return lambda0 + eta * delta_lambda * (R / denom).clamp(0.0, 1.0)


def l1snr_db(est, act, eps=1e-3, dbrms_eps=1e-8, lmin=-60.0, lambda0=0.1,
             delta_lambda=0.9, use_regularization=True, l1_weight=0.0, ref_level=None):
    """L1SNR plus the level-matching regularizer, scaled by (1 - l1_weight)."""
    base = l1snr_blended(est, act, eps, l1_weight, ref_level)
    if not use_regularization or l1_weight >= 1.0:
        return base
    L_true = dbrms(act, dbrms_eps)
    L_pred = dbrms(est, dbrms_eps)
    R = (L_pred - L_true).abs()
    lam = adaptive_lambda(L_pred, L_true, lmin, lambda0, delta_lambda)
    return base + (1.0 - l1_weight) * (lam * R).mean()


# ---------------------------------------------------------------------------------------------
# spectrogram domain
# ---------------------------------------------------------------------------------------------

def normalized_stft(x, n_fft, hop_length, win_length):
    """Matches torchaudio.Spectrogram(normalized=True, power=None, center=True, pad_mode='reflect').

    Built from torch.stft with normalized=False and an explicit divide by the window L2 norm. Do NOT
    substitute torch.stft(normalized=True): that divides by sqrt(n_fft) instead, a factor of 1.633 different
    for a 512-point Hann window (verified empirically).
    """
    w = torch.hann_window(win_length, dtype=x.dtype if x.dtype.is_floating_point else None,
                          device=x.device)
    # torch.stft accepts only 1D or 2D input, while torchaudio's Spectrogram accepts arbitrary leading
    # dims. Flatten to 2D, transform, then restore the leading shape.
    lead, t = x.shape[:-1], x.shape[-1]
    S = torch.stft(x.reshape(-1, t), n_fft, hop_length, win_length, w, center=True,
                   pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    S = S / w.pow(2).sum().sqrt()
    return S.reshape(*lead, *S.shape[-2:])


def _spec_parts(est, act, n_fft, hop_length, win_length):
    """Per-batch-element error and reference magnitudes for the real and imaginary parts."""
    b = est.shape[0]
    Se = normalized_stft(est.reshape(b, -1, est.shape[-1]), n_fft, hop_length, win_length)
    Sa = normalized_stft(act.reshape(b, -1, act.shape[-1]), n_fft, hop_length, win_length)
    out = []
    for pe, pa in ((Se.real, Sa.real), (Se.imag, Sa.imag)):
        pe = pe.reshape(b, -1)
        pa = pa.reshape(b, -1)
        out.append(((pe - pa).abs().mean(dim=1), pa.abs().mean(dim=1)))
    return out    # [(err_re, ref_re), (err_im, ref_im)]


def spec_d1(est, act, n_fft, hop_length, win_length, eps=1e-3):
    """Re and Im D1 terms, summed then batch-averaged, for one resolution."""
    (err_re, ref_re), (err_im, ref_im) = _spec_parts(est, act, n_fft, hop_length, win_length)
    d1_re = 10.0 * torch.log10((err_re + eps) / (ref_re + eps))
    d1_im = 10.0 * torch.log10((err_im + eps) / (ref_im + eps))
    return (d1_re + d1_im).mean()


def spec_blended(est, act, n_fft, hop_length, win_length, eps=1e-3, l1_weight=0.0,
                 spec_ref_level=None):
    """The spectrogram-domain l1_weight blend, including both endpoint shortcuts.

    At l1_weight >= 1 the library *averages* Re and Im (0.5 * (l1_re + l1_im)) where the blended path
    *sums* them via the factor 2.0 in scale_spec. That asymmetry is finding M4 and is reproduced here
    rather than smoothed over, because the library's behaviour is what is under test.
    """
    (err_re, ref_re), (err_im, ref_im) = _spec_parts(est, act, n_fft, hop_length, win_length)
    l1_mean = 0.5 * (err_re.mean() + err_im.mean())
    if l1_weight >= 1.0:
        return l1_mean
    d1_re = 10.0 * torch.log10((err_re + eps) / (ref_re + eps))
    d1_im = 10.0 * torch.log10((err_im + eps) / (ref_im + eps))
    d1_sum = (d1_re + d1_im).mean()
    if l1_weight <= 0.0:
        return d1_sum
    if spec_ref_level is None:
        inv = (0.5 * (1.0 / (ref_re.detach() + eps) + 1.0 / (ref_im.detach() + eps))).mean()
        scale = 2.0 * C * inv
    else:
        scale = 2.0 * C / (spec_ref_level + eps)
    return (1.0 - l1_weight) * d1_sum + l1_weight * (l1_mean * scale)


def multi_res_spec_d1(est, act, n_ffts=(512, 1024, 2048), hop_lengths=(128, 256, 512),
                      win_lengths=(512, 1024, 2048), eps=1e-3, l1_weight=0.0,
                      spec_ref_level=None):
    """Average across resolutions of the spectrogram loss, blended or not."""
    vals = [spec_blended(est, act, n, h, w, eps, l1_weight, spec_ref_level)
            for n, h, w in zip(n_ffts, hop_lengths, win_lengths)]
    return torch.stack(vals).mean()


# ---------------------------------------------------------------------------------------------
# hand-derived ground truth -- the reason to trust the above
# ---------------------------------------------------------------------------------------------
#
# D1, eps = 1e-3:
#   act=[1,1] est=[0,0]  err=1, ref=1  -> 10*log10(1.001/1.001) = 10*log10(1)         = 0.0
#   act=[1,1] est=[1,1]  err=0, ref=1  -> 10*log10(0.001/1.001) = 10*(-3.00043408)    = -30.0043408
#   act=[2,-2] est=[1,-1] err=1, ref=2 -> 10*log10(1.001/2.001) = 10*(-0.30081376)    = -3.0081376
#   act=[0,0] est=[0,0]  err=0, ref=0  -> 10*log10(0.001/0.001) = 10*log10(1)         = 0.0
#
# dbrms, eps = 1e-8:
#   x=[1,1] -> sqrt(1+1e-8)=1.000000005 -> 20*log10(1.000000015)  = 1.303e-7  (~0 dB)
#   x=[0,0] -> sqrt(1e-8)=1e-4          -> 20*log10(1.0001e-4)    = -79.9991314
#   x=[2,2] -> sqrt(4+1e-8)=2.0         -> 20*log10(2.0000000125)  = 6.0205999
#
HAND_CASES_D1 = [
    ([[1.0, 1.0]], [[0.0, 0.0]], 0.0),
    ([[1.0, 1.0]], [[1.0, 1.0]], -30.0043408),
    ([[2.0, -2.0]], [[1.0, -1.0]], -3.0081376),
    ([[0.0, 0.0]], [[0.0, 0.0]], 0.0),
]

HAND_CASES_DBRMS = [
    ([[1.0, 1.0]], 1.303e-7),
    ([[0.0, 0.0]], -79.9991314),
    ([[2.0, 2.0]], 6.0205999),
]
