import ast
import math
import pathlib

import torch
import pytest
from typing import Optional

import reference
from torch_l1_snr import (
    dbrms,
    L1SNRLoss,
    L1SNRDBLoss,
    STFTL1SNRDBLoss,
    MultiL1SNRDBLoss,
)

# --- Test Helper: Stem Wrapper ---
class StemWrappedLoss(torch.nn.Module):
    """Test helper matching user's pipL1SNRLoss wrapper pattern."""
    def __init__(self, base_loss, stem_dimension: Optional[int] = None):
        super().__init__()
        self.base_loss = base_loss
        self.stem_dimension = stem_dimension
    
    def forward(self, estimates, actuals, *args, **kwargs):
        if self.stem_dimension is not None:
            # Handle both [B,S,T] and [B,S,C,T] shapes
            if estimates.ndim == 3:  # [B, S, T]
                est_source = estimates[:, self.stem_dimension, :]
                act_source = actuals[:, self.stem_dimension, :]
            else:  # [B, S, C, T]
                est_source = estimates[:, self.stem_dimension, :, :]
                act_source = actuals[:, self.stem_dimension, :, :]
            return self.base_loss(est_source, act_source, *args, **kwargs)
        else:
            return self.base_loss(estimates, actuals, *args, **kwargs)

# --- Test Fixtures ---
#
# Every fixture is seeded. Unseeded fixtures made detection of a real defect probabilistic: an adversarial
# reviewer changed the `lmin` default from -60 to -30 and the suite caught it in only 14 of 30 runs. A test
# that catches a bug half the time is worse than one that never does, because it lands green and then fails
# an unrelated later change.
@pytest.fixture
def dummy_audio():
    """A batch of dummy audio signals."""
    g = torch.Generator().manual_seed(11)
    estimates = torch.randn(2, 16000, generator=g)
    actuals = torch.randn(2, 16000, generator=g)
    # Ensure actuals are not all zero to avoid division by zero in loss
    actuals[0, :100] += 0.1
    return estimates, actuals

@pytest.fixture
def dummy_stems():
    """A batch of dummy multi-stem signals: [B, S, C, T]"""
    g = torch.Generator().manual_seed(12)
    estimates = torch.randn(2, 4, 1, 16000, generator=g)
    actuals = torch.randn(2, 4, 1, 16000, generator=g)
    actuals[:, 0, :, :100] += 0.1
    return estimates, actuals

# --- Test Functions ---

@pytest.mark.no_forward  # exercises dbrms(), which is not one of the four forward() methods
def test_dbrms():
    signal = torch.ones(2, 1000) * 0.1
    # RMS of 0.1 is -20 dB
    assert torch.allclose(dbrms(signal), torch.tensor([-20.0, -20.0]), atol=1e-4)
    
    zeros = torch.zeros(2, 1000)
    # dbrms of zero should be -80dB with default eps=1e-8
    assert torch.allclose(dbrms(zeros), torch.tensor([-80.0, -80.0]), atol=1e-4)

def test_l1snr_loss(dummy_audio):
    estimates, actuals = dummy_audio
    loss_fn = L1SNRLoss(name="test")
    loss = loss_fn(estimates, actuals)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    # A value assertion, not just isnan/isinf: those are satisfied by any constant.
    assert torch.allclose(loss, reference.l1snr(estimates, actuals), atol=1e-6)

def test_l1snrdb_loss_time(dummy_audio):
    estimates, actuals = dummy_audio
    
    # Test with default settings (L1SNR + Regularization)
    loss_fn = L1SNRDBLoss(name="test", use_regularization=True, l1_weight=0.0)
    loss = loss_fn(estimates, actuals)
    assert loss.ndim == 0 and not torch.isnan(loss)

    # Test without regularization
    loss_fn_no_reg = L1SNRDBLoss(name="test_no_reg", use_regularization=False, l1_weight=0.0)
    loss_no_reg = loss_fn_no_reg(estimates, actuals)
    assert loss_no_reg.ndim == 0 and not torch.isnan(loss_no_reg)

    # Test with L1 loss component
    loss_fn_l1 = L1SNRDBLoss(name="test_l1", l1_weight=0.2)
    loss_l1 = loss_fn_l1(estimates, actuals)
    assert loss_l1.ndim == 0 and not torch.isnan(loss_l1)
    
    # Test pure L1 loss mode
    loss_fn_pure_l1 = L1SNRDBLoss(name="test_pure_l1", l1_weight=1.0)
    pure_l1_loss = loss_fn_pure_l1(estimates, actuals)
    # Pure L1 mode uses torch.nn.L1Loss, so compare with manual L1 calculation
    l1_loss_manual = torch.nn.L1Loss()(
        estimates.reshape(estimates.shape[0], -1),
        actuals.reshape(actuals.shape[0], -1)
    )
    assert torch.allclose(pure_l1_loss, l1_loss_manual)

def test_stft_l1snrdb_loss(dummy_audio):
    estimates, actuals = dummy_audio
    
    # Test with default settings
    loss_fn = STFTL1SNRDBLoss(name="test", l1_weight=0.0)
    loss = loss_fn(estimates, actuals)
    assert loss.ndim == 0
    assert torch.allclose(loss, reference.multi_res_spec_d1(estimates, actuals), atol=1e-5)

    # Test pure L1 mode
    loss_fn_pure_l1 = STFTL1SNRDBLoss(name="test_pure_l1", l1_weight=1.0)
    l1_loss = loss_fn_pure_l1(estimates, actuals)
    assert torch.allclose(l1_loss, reference.multi_res_spec_d1(estimates, actuals, l1_weight=1.0),
                          atol=1e-6)

    # Below min_audio_length=512 the loss falls back to a time-domain computation, not zero (M16)
    short_estimates = estimates[:, :500]
    short_actuals = actuals[:, :500]
    loss_short = loss_fn(short_estimates, short_actuals)
    assert loss_short.item() != 0.0
    assert torch.allclose(loss_short,
                          reference.l1snr_db(short_estimates, short_actuals, use_regularization=False),
                          atol=1e-6)

def test_stem_multi_loss(dummy_stems):
    estimates, actuals = dummy_stems

    # Test with a specific stem - users now manage stems manually by slicing
    # Extract stem 1 (second stem) manually
    est_stem = estimates[:, 1, ...]  # Shape: [batch, channels, samples]
    act_stem = actuals[:, 1, ...]
    loss_fn_stem = MultiL1SNRDBLoss(
        name="test_loss_stem",
        spec_weight=0.5,
        l1_weight=0.1
    )
    loss = loss_fn_stem(est_stem, act_stem)
    assert loss.ndim == 0
    assert torch.allclose(
        loss,
        0.5 * reference.l1snr_db(est_stem, act_stem, l1_weight=0.1, ref_level=REF_LEVEL)
        + 0.5 * reference.multi_res_spec_d1(est_stem, act_stem, l1_weight=0.1,
                                            spec_ref_level=SPEC_REF_LEVEL),
        atol=1e-5)

    # Test with all stems jointly - flatten all stems together
    # Reshape to [batch, -1] to process all stems at once
    est_all = estimates.reshape(estimates.shape[0], -1)
    act_all = actuals.reshape(actuals.shape[0], -1)
    loss_fn_all = MultiL1SNRDBLoss(
        name="test_loss_all",
        spec_weight=0.5,
        l1_weight=0.1
    )
    loss_all = loss_fn_all(est_all, act_all)
    assert torch.allclose(
        loss_all,
        0.5 * reference.l1snr_db(est_all, act_all, l1_weight=0.1, ref_level=REF_LEVEL)
        + 0.5 * reference.multi_res_spec_d1(est_all, act_all, l1_weight=0.1,
                                            spec_ref_level=SPEC_REF_LEVEL),
        atol=1e-5)

    # Pure L1 mode on all stems. The reference now covers the spectrogram L1 endpoint, so this is a value
    # assertion rather than the isnan check it used to be.
    loss_fn_l1 = MultiL1SNRDBLoss(name="l1_only", l1_weight=1.0)
    l1_loss = loss_fn_l1(est_all, act_all)
    assert torch.allclose(
        l1_loss,
        0.5 * reference.l1snr_db(est_all, act_all, l1_weight=1.0, ref_level=REF_LEVEL)
        + 0.5 * reference.multi_res_spec_d1(est_all, act_all, l1_weight=1.0,
                                            spec_ref_level=SPEC_REF_LEVEL),
        atol=1e-5)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
def test_loss_variants(dummy_audio, l1_weight):
    """Test L1SNRDBLoss and STFTL1SNRDBLoss with different l1_weights."""
    estimates, actuals = dummy_audio
    
    time_loss_fn = L1SNRDBLoss(name=f"test_time_{l1_weight}", l1_weight=l1_weight)
    time_loss = time_loss_fn(estimates, actuals)
    assert torch.allclose(time_loss, reference.l1snr_db(estimates, actuals, l1_weight=l1_weight,
                                                       ref_level=REF_LEVEL), atol=1e-6)

    spec_loss_fn = STFTL1SNRDBLoss(name=f"test_spec_{l1_weight}", l1_weight=l1_weight)
    spec_loss = spec_loss_fn(estimates, actuals)
    assert torch.allclose(spec_loss, reference.multi_res_spec_d1(
        estimates, actuals, l1_weight=l1_weight, spec_ref_level=SPEC_REF_LEVEL), atol=1e-5)

# --- Wrapper-Paradigm Tests ---

# --- Wrapper Tests: compared against an independent reference, never against base_loss ---
#
# Audit finding M2. These tests previously asserted
#     wrapped_loss(est, act) == base_loss(est, act)                       (stem_dimension=None)
#     wrapped_loss(est, act) == base_loss(est[:, k], act[:, k])           (stem_dimension=k)
# Both are structurally f(x) == f(x), because that is exactly what StemWrappedLoss.forward does. They could
# not fail for any implementation of f, and 112 of the suite's 127 cases were of this form: stubbing all four
# forward() methods to a constant left 122 of 127 passing.
#
# They now compare against tests/reference.py, which is derived from the definitions, imports nothing from
# torch_l1_snr, and is itself validated against hand-computed values (see test_reference_matches_hand_arithmetic).

# A14 replaced the batch statistic with module constants, so the reference is given the same constants the
# library derives: ref_level for the time domain, 0.19 * ref_level for the spectrogram domain (the measured
# STFT-to-time reference ratio, P0_SPEC_REF_LEVEL.md).
REF_LEVEL = 0.05
SPEC_REF_LEVEL = 0.19 * REF_LEVEL

REF_BY_CLASS = {
    "L1SNRLoss": lambda est, act, **kw: reference.l1snr_blended(
        est, act, l1_weight=kw["l1_weight"], ref_level=REF_LEVEL),
    "L1SNRDBLoss": lambda est, act, **kw: reference.l1snr_db(
        est, act, l1_weight=kw["l1_weight"], use_regularization=kw["use_reg"],
        ref_level=REF_LEVEL),
    "STFTL1SNRDBLoss": lambda est, act, **kw: reference.multi_res_spec_d1(
        est, act, l1_weight=kw["l1_weight"], spec_ref_level=SPEC_REF_LEVEL),
    "MultiL1SNRDBLoss": lambda est, act, **kw: (
        0.5 * reference.l1snr_db(est, act, l1_weight=kw["l1_weight"],
                                 use_regularization=kw["use_reg"], ref_level=REF_LEVEL)
        + 0.5 * reference.multi_res_spec_d1(est, act, l1_weight=kw["l1_weight"],
                                            spec_ref_level=SPEC_REF_LEVEL)),
}


def _build(cls_name, l1_weight, use_reg):
    if cls_name == "L1SNRLoss":
        return L1SNRLoss(name="t", weight=1.0, l1_weight=l1_weight)
    if cls_name == "L1SNRDBLoss":
        return L1SNRDBLoss(name="t", weight=1.0, l1_weight=l1_weight, use_regularization=use_reg)
    if cls_name == "STFTL1SNRDBLoss":
        return STFTL1SNRDBLoss(name="t", weight=1.0, l1_weight=l1_weight, use_regularization=False)
    return MultiL1SNRDBLoss(name="t", weight=1.0, spec_weight=0.5, l1_weight=l1_weight,
                            use_time_regularization=use_reg, use_spec_regularization=False)


@pytest.mark.no_forward  # structural AST check, calls no library code
def test_reference_module_does_not_import_the_library():
    """A reference that calls the code under test is not a reference. Guard it structurally.

    Walks the AST rather than grepping the source: the module's own docstring says it must not import
    torch_l1_snr, and a substring check flags that sentence as a violation.
    """
    tree = ast.parse((pathlib.Path(__file__).parent / "reference.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch_l1_snr" not in imported, (
        f"tests/reference.py imports the library under test; it imports {sorted(imported)}")


@pytest.mark.no_forward  # validates the reference itself, not the library
@pytest.mark.parametrize("act,est,expected", reference.HAND_CASES_D1)
def test_reference_d1_matches_hand_arithmetic(act, est, expected):
    """The reference is trustworthy because it reproduces arithmetic that can be redone on paper."""
    got = reference.d1_per_element(torch.tensor(est), torch.tensor(act))
    assert got.shape == (1,)
    assert abs(got.item() - expected) < 1e-4, f"reference D1 = {got.item()}, hand-derived {expected}"


@pytest.mark.no_forward  # validates the reference itself, not the library
@pytest.mark.parametrize("x,expected", reference.HAND_CASES_DBRMS)
def test_reference_dbrms_matches_hand_arithmetic(x, expected):
    got = reference.dbrms(torch.tensor(x))
    assert abs(got.item() - expected) < 1e-4, f"reference dbrms = {got.item()}, hand-derived {expected}"


@pytest.mark.parametrize("cls_name", ["L1SNRLoss", "L1SNRDBLoss", "STFTL1SNRDBLoss", "MultiL1SNRDBLoss"])
@pytest.mark.parametrize("ndim", [3, 4])
@pytest.mark.parametrize("stem_idx", [None, 0, 3])
@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("use_reg", [True, False])
def test_wrapper_matches_independent_reference(cls_name, ndim, stem_idx, l1_weight, use_reg):
    """StemWrappedLoss output must equal an independently computed value, not base_loss's own output."""
    torch.manual_seed(0)
    shape = (2, 4, 4096) if ndim == 3 else (2, 4, 1, 4096)
    actuals = torch.randn(*shape) * 0.05
    estimates = actuals + torch.randn(*shape) * 0.005

    # Argument order matters: D1's denominator is mean|act|, so the reference is not symmetric in its
    # two arguments. Passing them the wrong way round is a mistake this test caught during development.
    est_in = estimates[:, stem_idx] if stem_idx is not None else estimates
    act_in = actuals[:, stem_idx] if stem_idx is not None else actuals
    expected = REF_BY_CLASS[cls_name](est_in, act_in, l1_weight=l1_weight, use_reg=use_reg)

    wrapped = StemWrappedLoss(_build(cls_name, l1_weight, use_reg), stem_dimension=stem_idx)
    got = wrapped(estimates, actuals)

    assert got.ndim == 0
    assert torch.allclose(got, expected, atol=1e-5), (
        f"{cls_name} ndim={ndim} stem={stem_idx} w={l1_weight} reg={use_reg}: "
        f"library {got.item():.7f} vs independent reference {expected.item():.7f}")


@pytest.mark.parametrize("ndim", [3, 4])
def test_wrapper_distinguishes_between_stems(ndim):
    """A wrapper that ignored stem_dimension would pass every test above; this one catches it.

    Each stem is given a deliberately different level, so slicing the wrong index gives a different loss.
    """
    torch.manual_seed(0)
    shape = (2, 4, 4096) if ndim == 3 else (2, 4, 1, 4096)
    actuals = torch.randn(*shape) * 0.05
    for k in range(4):
        actuals[:, k] *= (10.0 ** k)      # 0, 20, 40, 60 dB apart
    estimates = actuals + torch.randn(*shape) * 0.005

    base = L1SNRLoss(name="t")
    per_stem = [StemWrappedLoss(base, stem_dimension=k)(estimates, actuals).item() for k in range(4)]
    assert len(set(round(v, 5) for v in per_stem)) == 4, (
        f"per-stem losses are not distinct, so stem_dimension may be ignored: {per_stem}")


def test_stft_wrapper_short_audio_falls_back_not_zero():
    """M16: below min_audio_length the STFT loss returns a time-domain fallback, not zero."""
    torch.manual_seed(0)
    actuals = torch.randn(2, 4, 400) * 0.05
    estimates = actuals + torch.randn(2, 4, 400) * 0.005
    loss = STFTL1SNRDBLoss(name="t", l1_weight=0.0)(estimates, actuals)
    assert loss.ndim == 0
    assert loss.item() != 0.0, "short audio must not return a zero loss"
    # the fallback is a time-domain L1SNRDBLoss with regularization off
    expected = reference.l1snr_db(estimates, actuals, use_regularization=False)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_multi_short_audio_collapses_to_time_domain():
    """M21: below min_audio_length both branches of MultiL1SNRDBLoss are the same time-domain quantity."""
    torch.manual_seed(0)
    actuals = torch.randn(2, 400) * 0.05
    estimates = actuals + torch.randn(2, 400) * 0.005
    loss_fn = MultiL1SNRDBLoss(name="t", spec_weight=0.5, use_time_regularization=False,
                               use_spec_regularization=False)
    combined = loss_fn(estimates, actuals)
    time_only = loss_fn.time_loss(estimates, actuals)
    spec_only = loss_fn.spec_loss(estimates, actuals)
    assert torch.allclose(time_only, spec_only, atol=1e-6), (
        "documented behaviour: the spectral branch falls back to the same time-domain loss")
    assert torch.allclose(combined, time_only, atol=1e-6)
    # Cross-branch equality alone is satisfied by any constant, so pin the absolute value too.
    assert torch.allclose(combined, reference.l1snr_db(estimates, actuals, use_regularization=False),
                          atol=1e-6)


# --- Gradient Behavior Tests ---
def test_gradient_distinction_l1snr_vs_l1():
    """
    Verify L1SNR and L1 have distinct gradient behaviors.
    L1SNR: inverse-error scaling (larger updates for small errors)
    L1: uniform gradients regardless of error magnitude
    """
    torch.manual_seed(42)

    actuals = torch.tensor([[1.0] * 100, [1.0] * 100])
    estimates = actuals.clone()
    estimates[0] += 0.01  # small error
    estimates[1] += 0.5   # large error

    # Pure L1SNR (l1_weight=0)
    est_snr = estimates.clone().requires_grad_(True)
    loss_snr = L1SNRLoss("test", l1_weight=0.0)(est_snr, actuals)
    loss_snr.backward()
    ratio_snr = est_snr.grad[0].abs().mean() / est_snr.grad[1].abs().mean()

    # Pure L1 (l1_weight=1)
    est_l1 = estimates.clone().requires_grad_(True)
    loss_l1 = L1SNRLoss("test", l1_weight=1.0)(est_l1, actuals)
    loss_l1.backward()
    ratio_l1 = est_l1.grad[0].abs().mean() / est_l1.grad[1].abs().mean()

    # L1SNR: larger gradient for small error sample (ratio >> 1)
    assert ratio_snr > 10.0, f"L1SNR gradient ratio should be >> 1, got {ratio_snr}"
    # L1: uniform gradients (ratio ~ 1)
    assert 0.9 < ratio_l1 < 1.1, f"L1 gradient ratio should be ~1, got {ratio_l1}"


def test_l1_weight_interpolation():
    """
    Verify l1_weight actually affects gradient behavior.
    Gradient ratio should decrease as l1_weight increases (from inverse-error toward uniform).
    """
    torch.manual_seed(42)

    actuals = torch.tensor([[1.0] * 100, [1.0] * 100])
    estimates = actuals.clone()
    estimates[0] += 0.01  # small error
    estimates[1] += 0.5   # large error

    ratios = []
    for w in [0.0, 0.5, 1.0]:
        est = estimates.clone().requires_grad_(True)
        loss = L1SNRLoss("test", l1_weight=w)(est, actuals)
        loss.backward()
        ratio = (est.grad[0].abs().mean() / est.grad[1].abs().mean()).item()
        ratios.append(ratio)

    # Gradient ratio should monotonically decrease as l1_weight increases
    assert ratios[0] > ratios[1] > ratios[2], \
        f"Gradient ratios should decrease with l1_weight: {ratios}"


def test_stft_gradient_distinction():
    """
    Same gradient distinction test for STFTL1SNRDBLoss.
    """
    torch.manual_seed(42)

    # Need longer audio for STFT
    actuals = torch.tensor([[1.0] * 4096, [1.0] * 4096])
    estimates = actuals.clone()
    estimates[0] += 0.01  # small error
    estimates[1] += 0.5   # large error

    # Pure L1SNR (l1_weight=0)
    est_snr = estimates.clone().requires_grad_(True)
    loss_fn_snr = STFTL1SNRDBLoss("test", l1_weight=0.0, n_ffts=[512], hop_lengths=[128], win_lengths=[512])
    loss_snr = loss_fn_snr(est_snr, actuals)
    loss_snr.backward()
    ratio_snr = est_snr.grad[0].abs().mean() / est_snr.grad[1].abs().mean()

    # Pure L1 (l1_weight=1)
    est_l1 = estimates.clone().requires_grad_(True)
    loss_fn_l1 = STFTL1SNRDBLoss("test", l1_weight=1.0, n_ffts=[512], hop_lengths=[128], win_lengths=[512])
    loss_l1 = loss_fn_l1(est_l1, actuals)
    loss_l1.backward()
    ratio_l1 = est_l1.grad[0].abs().mean() / est_l1.grad[1].abs().mean()

    # STFT processing smooths out per-sample differences, so ratios are smaller
    # Key check: L1SNR ratio > L1 ratio (gradient behaviors differ)
    assert ratio_snr > ratio_l1, f"STFT L1SNR ratio ({ratio_snr}) should be > L1 ratio ({ratio_l1})"
    # L1: more uniform gradients (ratio closer to 1)
    assert ratio_l1 < ratio_snr, f"STFT L1 should have more uniform gradients"


# --- MPS fallback (T1-9) ---
#
# The previous tests here could not detect the bug they guard, for two independent reasons (G2, and a second
# found while re-verifying the upstream reproducer):
#   1. On CPU they compared mps_cpu_fallback=True against False. The fallback is a no-op off MPS, so that is
#      a CPU-vs-CPU comparison and passes whatever MPS does.
#   2. Their MPS branch used 4096 samples. torch.stft backward on MPS is *correct* at that size -- it only
#      fails above 65,536 -- so even on Apple silicon the test exercised a working path.
#
# These use a size where MPS is measurably wrong (264,600 samples, i.e. 6 s at 44.1 kHz, at batch 2) and skip
# entirely without MPS rather than degenerating into a tautology. See PYTORCH_BUG_REPORT.md.

MPS_FAILING_LENGTH = 264600      # not a multiple of 65536, so MPS backward is wrong here
MPS_FAILING_BATCH = 2


requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="needs MPS hardware; skipped rather than degenerating to a CPU-vs-CPU comparison",
)


@requires_mps
def test_mps_fallback_gradients_match_cpu_at_a_failing_size():
    """With the fallback on, MPS gradients must match CPU at a size where raw MPS stft backward is wrong."""
    torch.manual_seed(42)
    actuals = torch.randn(MPS_FAILING_BATCH, MPS_FAILING_LENGTH) * 0.1
    estimates = actuals + 0.01 * torch.randn_like(actuals)

    def run(device, fallback):
        est = estimates.clone().to(device).requires_grad_(True)
        act = actuals.clone().to(device)
        loss_fn = STFTL1SNRDBLoss("t", n_ffts=[512], hop_lengths=[128], win_lengths=[512],
                                  mps_cpu_fallback=fallback).to(device)
        loss = loss_fn(est, act)
        loss.backward()
        return loss.detach().cpu(), est.grad.detach().cpu()

    cpu_loss, cpu_grad = run("cpu", False)
    fb_loss, fb_grad = run("mps", True)

    assert torch.allclose(cpu_loss, fb_loss, atol=1e-4), (
        f"fallback loss {fb_loss.item():.7f} != CPU {cpu_loss.item():.7f}")
    ratio = fb_grad.norm().item() / cpu_grad.norm().item()
    assert abs(ratio - 1.0) < 1e-3, (
        f"fallback gradient norm ratio vs CPU is {ratio:.6f}, expected 1.0")


@requires_mps
@pytest.mark.no_forward  # probes torch.stft directly; deliberately independent of this library
def test_raw_mps_is_actually_wrong_at_this_size():
    """The fallback test above is only meaningful if MPS is genuinely broken here. Assert that it is.

    This is the guard against the failure mode that made the old tests useless: a fallback test passes
    trivially if the path it bypasses was never broken. If PyTorch fixes the bug this test starts failing,
    which is the signal to re-enable MPS (T4-6) rather than a regression.
    """
    torch.manual_seed(42)
    actuals = torch.randn(MPS_FAILING_BATCH, MPS_FAILING_LENGTH) * 0.1
    estimates = actuals + 0.01 * torch.randn_like(actuals)

    def raw(device):
        est = estimates.clone().to(device).requires_grad_(True)
        w = torch.hann_window(512, device=device)
        S = torch.stft(est.to(device), 512, 128, 512, w, center=True, pad_mode="reflect",
                       normalized=False, onesided=True, return_complex=True)
        S.real.abs().mean().backward()
        return est.grad.detach().cpu()

    ratio = raw("mps").norm().item() / raw("cpu").norm().item()
    assert abs(ratio - 1.0) > 1e-2, (
        f"raw MPS stft backward now agrees with CPU (ratio {ratio:.6f}). If upstream fixed this, "
        f"mps_cpu_fallback can be version-gated -- see T4-6 and PYTORCH_BUG_REPORT.md.")


def test_mps_fallback_is_a_noop_off_mps():
    """Off MPS the flag must not change anything. Pinned against the reference, not against itself."""
    torch.manual_seed(42)
    actuals = torch.randn(2, 4096) * 0.1
    estimates = actuals + 0.01 * torch.randn_like(actuals)
    expected = reference.spec_blended(estimates, actuals, 512, 128, 512)
    for fallback in (True, False):
        est = estimates.clone().requires_grad_(True)
        loss = STFTL1SNRDBLoss("t", n_ffts=[512], hop_lengths=[128], win_lengths=[512],
                               mps_cpu_fallback=fallback)(est, actuals)
        loss.backward()
        assert torch.allclose(loss, expected, atol=1e-6), f"fallback={fallback} changed the CPU result"
        assert est.grad is not None


@pytest.mark.no_forward  # inspects a constructor attribute; never calls forward()
def test_multi_passes_mps_flag_to_spectrogram_branch():
    """MultiL1SNRDBLoss must thread mps_cpu_fallback through to the branch that needs it."""
    for flag in (True, False):
        loss_fn = MultiL1SNRDBLoss("t", mps_cpu_fallback=flag, n_ffts=[256], hop_lengths=[64],
                                   win_lengths=[256], min_audio_length=256)
        assert loss_fn.spec_loss.mps_cpu_fallback is flag


def test_stft_l1_weight_interpolation():
    """l1_weight must move the spectrogram loss's cross-row gradient profile monotonically toward L1.

    The metric is max/min of the per-row gradient magnitude, which is what "how much L1 behaviour" means:
    pure D1 weights rows by their own error magnitude (inverse-error), pure L1 weights them equally, so the
    profile runs from a spread down to 1.0.

    This replaces an earlier version that used a fixed-order ratio grad[0]/grad[1] and asserted every value
    was > 1.0. That assertion was not a property of the loss. Pure L1 gives uniform gradient magnitudes, so
    its ratio is ~1.0 rather than above it, and on random signals even the pure-SNR ratio came out below 1.0
    (measured 0.9669, 0.9599, 0.9943 across three seeds). It held only for the one constant-signal
    configuration it was written against, and passed by luck rather than by testing anything.
    """
    profiles_by_spread = {}
    for label, quiet_level in (("20 dB", 0.005), ("40 dB", 0.0005)):
        actuals = torch.stack([
            torch.randn(4096, generator=torch.Generator().manual_seed(1)) * 0.05,
            torch.randn(4096, generator=torch.Generator().manual_seed(2)) * quiet_level,
        ])
        # a fixed *relative* error on each row, so the rows differ only in level
        estimates = actuals + 0.1 * actuals.abs().mean(dim=-1, keepdim=True) * torch.randn(
            actuals.shape, generator=torch.Generator().manual_seed(9))

        profile = []
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            est = estimates.clone().requires_grad_(True)
            loss_fn = STFTL1SNRDBLoss("t", l1_weight=w, n_ffts=[512], hop_lengths=[128],
                                      win_lengths=[512])
            loss_fn(est, actuals).backward()
            per_row = [est.grad[i].abs().mean().item() for i in range(actuals.shape[0])]
            profile.append(max(per_row) / min(per_row))

        assert all(profile[i] >= profile[i + 1] - 1e-6 for i in range(len(profile) - 1)), (
            f"{label}: profile must decrease monotonically as l1_weight rises, got {profile}")
        assert profile[-1] < 1.05, (
            f"{label}: pure L1 should give near-uniform gradients, got a profile of {profile[-1]:.4f}")
        assert profile[0] > profile[-1] + 0.5, (
            f"{label}: pure SNR should be markedly more level-dependent than pure L1, got {profile}")
        profiles_by_spread[label] = profile[0]

    # a wider level spread means a stronger inverse-error weighting under pure SNR
    assert profiles_by_spread["40 dB"] > profiles_by_spread["20 dB"], (
        f"a wider level spread should give a larger pure-SNR profile: {profiles_by_spread}")


# --- L2SNRLoss: the metric-matched (energy-ratio) sibling of L1SNRLoss ---
#
# Added for the CHRIS-394 A/B. uSDR is an energy ratio, so the derivation says the matched time-domain
# objective is a log of floored error ENERGY rather than of mean-absolute error. Whether that actually
# beats D1 on held-out uSDR is an open empirical question -- these gates only pin the arithmetic.

def test_l2snr_is_the_hand_computed_floored_energy_ratio():
    from torch_l1_snr import L2SNRLoss
    torch.manual_seed(0)
    est = torch.randn(4, 2, 8000)
    act = torch.randn(4, 2, 8000)
    tau, eps = 1e-3, 1e-6

    e = est.reshape(4, -1)
    a = act.reshape(4, -1)
    err = (e - a).pow(2).mean(dim=-1)
    ref = a.pow(2).mean(dim=-1)
    expected = (10.0 * torch.log10((err + tau * ref + eps) / (ref + eps))).mean()

    assert torch.allclose(L2SNRLoss("t")(est, act), expected, atol=0, rtol=1e-6)


def test_l2snr_bottoms_out_at_the_tau_implied_snr_cap():
    """tau=1e-3 caps the attainable SNR at 30 dB, so a perfect estimate scores -30, not -inf."""
    from torch_l1_snr import L2SNRLoss
    torch.manual_seed(0)
    # Scaled up so the absolute eps is negligible against tau*mean(y^2). At unit level eps shifts the cap
    # by a visible 0.004 dB, which is correct behaviour but not what this test is pinning.
    act = torch.randn(2, 4000) * 10
    assert torch.allclose(L2SNRLoss("t")(act.clone(), act), torch.tensor(-30.0), atol=1e-3)
    assert torch.allclose(L2SNRLoss("t", tau=1e-4)(act.clone(), act), torch.tensor(-40.0), atol=1e-3)


def test_l2snr_stays_bounded_on_a_silent_target():
    """The tau floor is relative to the target, so it vanishes when the target is silent.

    Without an absolute eps as well, a silent target leaves 10*log10(mean(e^2)) unfloored and the loss
    runs to -inf with a gradient going as 1/mean(e^2). Silent chunks are routine in stem training.
    """
    from torch_l1_snr import L2SNRLoss
    eps = 1e-6
    loss_fn = L2SNRLoss("t", eps=eps)
    for dtype in (torch.float32, torch.float64):
        silent = torch.zeros(2, 4000, dtype=dtype)
        for scale in (1e-2, 1e-5, 1e-10, 0.0):
            out = loss_fn(silent + scale, silent)
            assert torch.isfinite(out), f"{dtype}: non-finite loss at estimate scale {scale}"
            assert out >= -1e-6, f"{dtype}: loss below the floor at scale {scale}: {out.item()}"

            # With y=0 the target-relative tau floor vanishes, so the whole floor is eps and the loss is
            # exactly 10*log10((scale^2 + eps) / eps). Pinning the value, not just finiteness: an
            # isfinite/isnan pair is satisfied by any constant. Checked in float64 only -- the ratio sits
            # a hair above 1 for small scales, where float32 cancellation in log10 costs four digits and
            # would force a tolerance too loose to mean anything.
            if dtype is torch.float64:
                expected = 10.0 * math.log10((scale ** 2 + eps) / eps)
                assert out.item() == pytest.approx(expected, rel=1e-12, abs=1e-15), (
                    f"at estimate scale {scale}: got {out.item()}, expected {expected}")


def test_l2snrs_peak_gradient_matches_d1s_on_a_silent_target():
    """eps=1e-6 is the power-domain analogue of D1's amplitude-domain eps=1e-3.

    log(x^2 + eps) peaks in gradient at x=sqrt(eps) with value c/sqrt(eps); log(|x| + eps') peaks at
    c/eps'. Setting eps = eps'^2 makes the two ceilings equal, so swapping norms does not silently
    change how hard the loss pushes near convergence.
    """
    from torch_l1_snr import L2SNRLoss

    def peak_grad(loss_fn, amplitudes):
        best = 0.0
        for amp in amplitudes:
            a = torch.tensor(float(amp), requires_grad=True)
            loss_fn(a * torch.ones(1, 2000), torch.zeros(1, 2000)).backward()
            best = max(best, abs(a.grad.item()))
        return best

    sweep = [10 ** (-k / 4) for k in range(2, 33)]
    l2_peak = peak_grad(L2SNRLoss("t"), sweep)
    d1_peak = peak_grad(L1SNRLoss("t"), sweep)
    # Absolute value first. Asserting only that the two agree is an f(x) == f(x) comparison: it holds
    # just as well when both losses are broken in the same way.
    ceiling = (10.0 / math.log(10.0)) / math.sqrt(1e-6)
    assert l2_peak == pytest.approx(ceiling, rel=0.02), (
        f"L2SNR peak gradient should be c/sqrt(eps) = {ceiling:.1f}, got {l2_peak:.1f}")
    assert d1_peak == pytest.approx(ceiling, rel=0.02), (
        f"D1 peak gradient should be c/eps' = {ceiling:.1f}, got {d1_peak:.1f}")
    assert l2_peak == pytest.approx(d1_peak, rel=0.02), (
        f"peak gradients should match by construction: L2 {l2_peak:.1f} vs D1 {d1_peak:.1f}")


@pytest.mark.parametrize("shape", [(3, 4000), (3, 4, 4000), (3, 4, 2, 4000)])
def test_l2snr_flattens_every_batch_first_shape_to_one_value_per_example(shape):
    """Value-asserted per shape, so it also proves the flatten is over ALL non-batch dims.

    An ndim==0/isfinite pair would pass against a constant, and against a loss that reduced over the
    wrong axis.
    """
    from torch_l1_snr import L2SNRLoss
    torch.manual_seed(0)
    est, act = torch.randn(*shape), torch.randn(*shape)
    tau, eps = 1e-3, 1e-6

    e, a = est.reshape(shape[0], -1), act.reshape(shape[0], -1)
    err, ref = (e - a).pow(2).mean(dim=-1), a.pow(2).mean(dim=-1)
    expected = (10.0 * torch.log10((err + tau * ref + eps) / (ref + eps))).mean()

    out = L2SNRLoss("t")(est, act)
    assert out.ndim == 0
    assert torch.allclose(out, expected, atol=0, rtol=1e-6)


# --- MultiL1SNRDBLoss: injectable time-domain sub-loss ---

def test_the_multi_domain_time_branch_can_be_replaced_by_an_injected_module():
    from torch_l1_snr import L2SNRLoss
    torch.manual_seed(0)
    est, act = torch.randn(2, 2, 16000), torch.randn(2, 2, 16000)

    injected = L2SNRLoss("l2")
    loss_fn = MultiL1SNRDBLoss("m", time_loss_module=injected)
    assert loss_fn.time_loss is injected

    # The injection must actually change the number. Comparing only against a recombination of the same
    # sub-losses is f(x) == f(x) and holds even when every forward returns a constant.
    default = MultiL1SNRDBLoss("m")
    assert not torch.allclose(loss_fn(est, act), default(est, act)), (
        "injecting L2SNRLoss left the combined loss unchanged, so the injection did nothing")

    w = loss_fn.spec_weight
    expected = (1 - w) * injected(est, act) + w * loss_fn.spec_loss(est, act)
    assert torch.allclose(loss_fn(est, act), expected * loss_fn.weight)


@pytest.mark.no_forward
def test_injecting_a_time_loss_warns_that_the_time_only_parameters_are_orphaned():
    """Only `time_loss_params` and `use_time_regularization` are time-exclusive.

    lambda0, l1snr_eps, l1_weight and the rest are shared with the spectrogram branch, so they keep
    working and must NOT warn -- warning on them would be noise on an ordinary call.
    """
    from torch_l1_snr import L2SNRLoss
    with pytest.warns(UserWarning, match="time_loss_module"):
        MultiL1SNRDBLoss("m", time_loss_module=L2SNRLoss("l2"), time_loss_params={"weight": 2.0})
    with pytest.warns(UserWarning, match="time_loss_module"):
        MultiL1SNRDBLoss("m", time_loss_module=L2SNRLoss("l2"), use_time_regularization=False)

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        MultiL1SNRDBLoss("m", time_loss_module=L2SNRLoss("l2"), lambda0=0.5, l1snr_eps=1e-2)


@pytest.mark.no_forward
def test_injecting_a_non_module_is_rejected():
    with pytest.raises(ValueError, match="time_loss_module"):
        MultiL1SNRDBLoss("m", time_loss_module="L2SNRLoss")


def test_the_multi_domain_default_is_untouched_by_the_injection_parameter():
    """Regression guard, not a red-green gate: this must hold before and after the change."""
    torch.manual_seed(0)
    est, act = torch.randn(2, 2, 16000), torch.randn(2, 2, 16000)
    loss_fn = MultiL1SNRDBLoss("m")
    assert isinstance(loss_fn.time_loss, L1SNRDBLoss)
    assert loss_fn(est, act).item() == pytest.approx(2.230844497680664, abs=0, rel=1e-6)


# --- L2SNRLoss: the cross-class gates it does not inherit ---
#
# tests/test_edge_cases.py parametrizes ~21 gates over its own ALL_CLASSES, but every one of them assumes
# an `l1_weight` parameter, which L2SNRLoss deliberately does not have. Adding it to that list would fail
# for the wrong reason, so the universal contracts are pinned here instead. An adversarial reviewer found
# all three of these mutations surviving the whole numerical suite.

@pytest.mark.parametrize("weight", [0.0, 0.5, 2.0, 3.7])
def test_l2snr_weight_multiplier_scales_the_loss(weight):
    """Mutation that survived: deleting `* self.weight` from forward()."""
    from torch_l1_snr import L2SNRLoss
    torch.manual_seed(0)
    est, act = torch.randn(3, 8000), torch.randn(3, 8000)
    base = L2SNRLoss("t").forward(est, act)
    assert torch.allclose(L2SNRLoss("t", weight=weight)(est, act), base * weight, atol=0, rtol=1e-6)


@pytest.mark.no_forward  # constructor validation only; verified by tests/_forward_counter.py
@pytest.mark.parametrize("kwargs, field", [
    ({"weight": -1.0}, "weight"),      # negates the objective
    ({"weight": -1e-9}, "weight"),
    ({"eps": 0.0}, "eps"),             # un-floors a silent target
    ({"eps": -1e-6}, "eps"),
    ({"tau": -1e-9}, "tau"),
])
def test_l2snr_rejects_out_of_range_constructor_values(kwargs, field):
    """Mutation that survived: deleting the _validate_* calls from __init__."""
    from torch_l1_snr import L2SNRLoss
    with pytest.raises(ValueError, match=field):
        L2SNRLoss("t", **kwargs)


def test_l2snr_rejects_the_input_shapes_and_dtypes_every_other_loss_rejects():
    """Mutation that survived: deleting both _validate_* calls from forward().

    Each of these returned a plausible number or a silent NaN instead of raising.
    """
    from torch_l1_snr import L2SNRLoss
    loss_fn = L2SNRLoss("t")
    with pytest.raises(ValueError):                                   # rank-1: no batch dimension
        loss_fn(torch.randn(4000), torch.randn(4000))
    with pytest.raises(ValueError):                                   # empty -> NaN
        loss_fn(torch.zeros(2, 0, 4000), torch.zeros(2, 0, 4000))
    with pytest.raises(ValueError):                                   # integer dtype
        loss_fn(torch.zeros(2, 400, dtype=torch.int16), torch.zeros(2, 400, dtype=torch.int16))
    with pytest.raises(ValueError):                                   # mismatched shapes
        loss_fn(torch.randn(2, 4000), torch.randn(2, 8000))


def test_l2snrs_gradient_ceiling_rises_with_the_target_level():
    """The eps-matching derivation holds at silence ONLY, and the docs must not overstate it.

    On a non-silent target the floor inside the log is tau*mean(y^2) + eps, so the ceiling is
    c/sqrt(tau*mean(y^2) + eps) rather than c/sqrt(eps). At 0 dBFS that is ~32x below D1's, which is
    target-independent. Pinning the direction so the claim cannot silently drift back.
    """
    from torch_l1_snr import L2SNRLoss
    c = 10.0 / math.log(10.0)
    loss_fn = L2SNRLoss("t")

    def peak_grad(ref_energy):
        best = 0.0
        for amp in (10 ** (-k / 4) for k in range(0, 33)):
            a = torch.tensor(float(amp), requires_grad=True, dtype=torch.float64)
            act = torch.full((1, 2000), math.sqrt(ref_energy), dtype=torch.float64)
            loss_fn(act + a, act).backward()
            best = max(best, abs(a.grad.item()))
        return best

    silent, loud = peak_grad(0.0), peak_grad(1.0)
    assert silent == pytest.approx(c / math.sqrt(1e-6), rel=0.02)
    assert loud == pytest.approx(c / math.sqrt(1e-3 + 1e-6), rel=0.02)
    assert silent > 20 * loud, (
        f"ceiling must fall as the target gets louder: silent {silent:.1f} vs 0 dBFS {loud:.1f}")


@pytest.mark.no_forward  # construction-time warning only; verified by tests/_forward_counter.py
def test_injecting_a_time_loss_with_pure_l1_mode_warns_about_the_asymmetry():
    """pure_l1_mode reads True while the injected time branch is not an L1 loss.

    Found by an adversarial reviewer: this combination silently produced a public attribute that was
    simply wrong, with no warning of any kind.
    """
    from torch_l1_snr import L2SNRLoss
    with pytest.warns(UserWarning, match="pure_l1_mode"):
        m = MultiL1SNRDBLoss("m", l1_weight=1.0, time_loss_module=L2SNRLoss("l2"))
    assert m.pure_l1_mode is True  # the asymmetry the warning exists to announce

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        MultiL1SNRDBLoss("m", l1_weight=0.0, time_loss_module=L2SNRLoss("l2"))


def test_an_injected_modules_own_weight_multiplies_on_top_of_the_branch_weight():
    """Documented asymmetry: the built-in branch is forced to weight=1.0, an injected one is not."""
    from torch_l1_snr import L2SNRLoss
    torch.manual_seed(0)
    est, act = torch.randn(2, 16000), torch.randn(2, 16000)
    plain = MultiL1SNRDBLoss("m", time_loss_module=L2SNRLoss("l2"))
    scaled = MultiL1SNRDBLoss("m", time_loss_module=L2SNRLoss("l2", weight=3.0))
    w = plain.spec_weight
    delta = 2.0 * (1 - w) * L2SNRLoss("l2")(est, act)   # the extra 2x on the time branch only
    assert torch.allclose(scaled(est, act) - plain(est, act), delta, atol=0, rtol=1e-6)


def test_l2snr_reports_about_twice_the_decibels_d1_does():
    """D1 is 10log10 of an AMPLITUDE ratio (bandit's convention); L2SNR is 10log10 of a POWER ratio.

    Pinned because the consequence is easy to miss and expensive: inside MultiL1SNRDBLoss the branches
    are summed as (1-spec_weight)*time + spec_weight*spec, so a time term of twice the magnitude takes
    twice the share of the objective at an unchanged spec_weight. An A/B that swaps only the norm is
    confounded with a domain-balance change unless spec_weight is adjusted.
    """
    from torch_l1_snr import L2SNRLoss
    torch.manual_seed(0)
    for r in (0.3, 0.1, 0.03, 0.01):
        y = torch.randn(8, 40000, dtype=torch.float64) * 0.05
        est = y + r * torch.randn_like(y) * 0.05
        d1 = L1SNRLoss("t")(est, y).item()
        # eps tiny so the comparison is of the norms, not of the two floors
        d2 = L2SNRLoss("t", eps=1e-30)(est, y).item()
        assert 1.8 < d2 / d1 < 2.3, f"at relative error {r}: ratio {d2/d1:.3f}, expected ~2"


def test_swapping_the_time_norm_shifts_the_multi_domain_balance():
    """The confound itself, measured end to end, so the A/B design cannot forget it."""
    from torch_l1_snr import L2SNRLoss
    torch.manual_seed(0)
    y = torch.randn(4, 40000) * 0.05
    est = y + 0.1 * torch.randn_like(y) * 0.05

    spec = STFTL1SNRDBLoss("s", use_regularization=False)(est, y).abs().item()
    share_d1 = L1SNRLoss("t")(est, y).abs().item() / spec
    share_l2 = L2SNRLoss("t")(est, y).abs().item() / spec

    assert share_l2 > 1.8 * share_d1, (
        f"the time branch should roughly double its share: D1 {share_d1:.3f} vs L2 {share_l2:.3f}")


@pytest.mark.parametrize("eps,fp16_safe", [(1e-6, True), (1e-8, False)])
def test_l2snr_eps_below_float16s_subnormal_floor_gives_inf_on_a_silent_target(eps, fp16_safe):
    """float16's smallest subnormal is 5.96e-08, so eps=1e-8 rounds to zero and the floor disappears.

    Numerator and denominator then both collapse on a silent target and the loss returns +inf, which
    kills a training run. This is the concrete reason not to lower the eps default. bfloat16 is fine:
    it has float32's exponent range, so it represents 1e-8 exactly despite having FEWER mantissa bits.
    """
    from torch_l1_snr import L2SNRLoss
    loss_fn = L2SNRLoss("t", eps=eps)

    assert (torch.tensor(eps, dtype=torch.float16).item() != 0.0) is fp16_safe

    # With a silent target the whole floor is eps, so the exact value is 10*log10((scale^2 + eps)/eps).
    # Asserting the VALUE, not just finiteness: a stubbed constant forward() is finite too.
    scale = 1e-2
    expected = 10.0 * math.log10((scale ** 2 + eps) / eps)

    silent16 = torch.zeros(2, 4000, dtype=torch.float16)
    out16 = loss_fn(silent16 + scale, silent16)
    assert bool(torch.isfinite(out16)) is fp16_safe, (
        f"float16 at eps={eps:.0e} gave {out16.item()}, expected "
        f"{'a finite value' if fp16_safe else '+inf (eps flushed to zero)'}")
    if fp16_safe:
        assert out16.item() == pytest.approx(expected, abs=0.1)

    # bfloat16 survives either way: float32's exponent range, despite fewer mantissa bits.
    silentbf = torch.zeros(2, 4000, dtype=torch.bfloat16)
    outbf = loss_fn(silentbf + scale, silentbf)
    assert outbf.item() == pytest.approx(expected, abs=0.1), (
        f"bfloat16 at eps={eps:.0e} gave {outbf.item()}, expected {expected:.4f}")

    # autocast promotes the reduction and the log to float32, so it survives too.
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        silent32 = torch.zeros(2, 4000)
        out = loss_fn(silent32 + scale, silent32)
    assert out.item() == pytest.approx(expected, abs=0.1), (
        f"autocast(float16) at eps={eps:.0e} gave {out.item()}, expected {expected:.4f}")
