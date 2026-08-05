import ast
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
@pytest.fixture
def dummy_audio():
    """Provides a batch of dummy audio signals."""
    estimates = torch.randn(2, 16000)
    actuals = torch.randn(2, 16000)
    # Ensure actuals are not all zero to avoid division by zero in loss
    actuals[0, :100] += 0.1 
    return estimates, actuals

@pytest.fixture
def dummy_stems():
    """Provides a batch of dummy multi-stem signals."""
    estimates = torch.randn(2, 4, 1, 16000) # batch, stems, channels, samples
    actuals = torch.randn(2, 4, 1, 16000)
    actuals[:, 0, :, :100] += 0.1 # Ensure not all zero
    return estimates, actuals

@pytest.fixture
def dummy_stems_3d():
    """Multi-stem signals: [B, S, T]"""
    estimates = torch.randn(2, 4, 16000)
    actuals = torch.randn(2, 4, 16000)
    actuals[:, 0, :100] += 0.1  # Ensure not all zero
    return estimates, actuals

@pytest.fixture
def dummy_stems_4d():
    """Multi-stem signals: [B, S, C, T]"""
    estimates = torch.randn(2, 4, 1, 16000)
    actuals = torch.randn(2, 4, 1, 16000)
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
