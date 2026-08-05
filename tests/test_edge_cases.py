"""Validation, warnings, and edge cases.

Audit finding M11: the suite had zero `pytest.raises` and zero `pytest.warns` in 753 lines, so all five
validations and all six warning paths were unasserted, and silence -- the dominant failure mode of SNR-style
losses -- had no test at all. Coverage was 79%, and the uncovered set was almost exactly the defensive code
plus the novel spectrogram regularizer.

Validation raises `ValueError`, not `AssertionError`, so these tests run under `python -O` too. That is
finding M15: bare `assert` is removed by the optimizer, and out-of-range values then produce a silently wrong
loss instead of an error. The suite's `python -O` job is what holds this.
"""
import ast
import pathlib
import warnings

import pytest
import torch

import reference
from torch_l1_snr import (
    dbrms,
    L1SNRLoss,
    L1SNRDBLoss,
    STFTL1SNRDBLoss,
    MultiL1SNRDBLoss,
)

ALL_CLASSES = [L1SNRLoss, L1SNRDBLoss, STFTL1SNRDBLoss, MultiL1SNRDBLoss]


def audio(*shape, level=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * level


# ---------------------------------------------------------------------------------------------
# T1-5 -- validation paths
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("bad", [-0.5, -1e-9, 1.5, 2.0])
def test_l1_weight_out_of_range_is_rejected(cls, bad):
    """M14/A2: all four classes now validate. L1SNRLoss was the only one that did not, and silently took
    the pure-SNR branch for -0.5 and the pure-L1 branch for 2.0."""
    with pytest.raises(ValueError, match="l1_weight"):
        cls(name="t", l1_weight=bad)


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("good", [0.0, 0.5, 1.0])
def test_l1_weight_endpoints_are_accepted(cls, good):
    """The range is inclusive. Guards against validation that is too strict."""
    assert cls(name="t", l1_weight=good) is not None


@pytest.mark.parametrize("bad", [-0.1, 1.0001, 1.5, 2.0])
def test_spec_weight_out_of_range_is_rejected(bad):
    """M19, the highest-priority correctness item.

    forward computes (1 - spec_weight) * time_loss + spec_weight * spec_loss. Above 1 the time coefficient
    goes negative, so the optimizer is rewarded for degrading waveform reconstruction: measured
    spec_weight=1.5 giving a time coefficient of -0.5, with the loss becoming more negative as the
    time-domain estimate got worse. The docstring said "set higher to emphasize spectral accuracy" with no
    stated bound, actively inviting it.
    """
    with pytest.raises(ValueError, match="spec_weight"):
        MultiL1SNRDBLoss(name="t", spec_weight=bad)


@pytest.mark.parametrize("good", [0.0, 0.5, 1.0])
def test_spec_weight_endpoints_are_accepted(good):
    assert MultiL1SNRDBLoss(name="t", spec_weight=good) is not None


def test_time_coefficient_never_goes_negative():
    """M19 stated as the invariant rather than the symptom: no accepted spec_weight may invert the sign."""
    for sw in [0.0, 0.25, 0.5, 0.75, 1.0]:
        loss_fn = MultiL1SNRDBLoss(name="t", spec_weight=sw)
        assert (1.0 - loss_fn.spec_weight) >= 0.0


def test_mismatched_stft_parameter_list_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        STFTL1SNRDBLoss(name="t", n_ffts=[512, 1024], hop_lengths=[128], win_lengths=[512, 1024])


def test_win_length_larger_than_n_fft_is_rejected():
    with pytest.raises(ValueError, match="win_length"):
        STFTL1SNRDBLoss(name="t", n_ffts=[512], hop_lengths=[128], win_lengths=[1024])


def test_unknown_window_fn_is_rejected_with_the_valid_options():
    """Q25/A12: surfaced as an opaque AttributeError about a mangled attribute name."""
    with pytest.raises(ValueError) as exc:
        STFTL1SNRDBLoss(name="t", window_fn="blackman_harris")
    message = str(exc.value)
    for valid in ["hann", "hamming", "blackman", "bartlett", "kaiser"]:
        assert valid in message, f"the error message does not name {valid!r} as a valid option"


def test_validation_survives_python_O():
    """M15: the checks must not be bare asserts, which python -O strips.

    Asserted structurally as well as through the -O CI job, so the requirement is visible in the source.
    """
    src = (pathlib.Path(__file__).parent.parent / "torch_l1_snr" / "l1snr.py").read_text()
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("__init__",):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assert):
                    offenders.append(sub.lineno)
    assert not offenders, (
        f"__init__ still validates with bare assert at line(s) {offenders}; python -O would remove it")


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
def test_valid_window_names_all_construct(cls):
    """The five reachable window factories. Guards against a validation fix that is too strict."""
    for name in ["hann", "hamming", "blackman", "bartlett", "kaiser"]:
        if cls in (STFTL1SNRDBLoss, MultiL1SNRDBLoss):
            cls(name="t", window_fn=name)


# ---------------------------------------------------------------------------------------------
# T1-5 -- warning paths
# ---------------------------------------------------------------------------------------------

def test_spectrogram_failure_warns_and_names_the_resolution():
    """A resolution whose reflect-pad cannot be satisfied warns rather than failing silently."""
    loss_fn = STFTL1SNRDBLoss(name="t", n_ffts=[512, 8192], hop_lengths=[128, 8],
                              win_lengths=[512, 8192], min_audio_length=16)
    est, act = audio(2, 600), audio(2, 600, seed=1)
    with pytest.warns(UserWarning, match="resolution"):
        loss = loss_fn(est, act)
    assert loss.ndim == 0


def test_all_resolutions_failing_warns():
    """M6: currently returns a graph-detached zero. A6 makes it graph-connected; the warning stays."""
    loss_fn = STFTL1SNRDBLoss(name="t", n_ffts=[8192], hop_lengths=[8], win_lengths=[8192],
                              min_audio_length=16)
    est, act = audio(2, 600), audio(2, 600, seed=1)
    with pytest.warns(UserWarning, match="All spectrogram transforms failed"):
        loss = loss_fn(est, act)
    assert loss.item() == 0.0


def test_all_resolutions_failing_returns_a_detached_zero_today():
    """M6, pinned as current behaviour: .backward() raises because the zero carries no grad_fn.

    A6 fixes this. When it lands, this test inverts to assert backward() succeeds.
    """
    loss_fn = STFTL1SNRDBLoss(name="t", n_ffts=[8192], hop_lengths=[8], win_lengths=[8192],
                              min_audio_length=16)
    est = audio(2, 600).requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = loss_fn(est, audio(2, 600, seed=1))
    assert loss.grad_fn is None
    with pytest.raises(RuntimeError, match="does not require grad"):
        loss.backward()


def test_nan_input_is_scrubbed_without_warning_today():
    """M7: an all-NaN estimate yields exactly 0.0 with an all-zero gradient and no warning at all.

    A diverged run therefore reports as healthy. A7 adds a one-shot warning; this test then becomes a
    pytest.warns assertion. Pinned now so the silence is on the record.
    """
    loss_fn = STFTL1SNRDBLoss(name="t")
    est = torch.full((2, 4096), float("nan"), requires_grad=True)
    act = audio(2, 4096)
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # any warning at all would fail this
        loss = loss_fn(est, act)
    loss.backward()
    assert loss.item() == 0.0
    assert est.grad.abs().max().item() == 0.0


def test_time_domain_losses_do_not_scrub_nan():
    """Q12: only the STFT path sanitizes. A NaN estimate in the time domain gives a NaN loss, visibly."""
    act = audio(2, 4096)
    est = torch.full((2, 4096), float("nan"))
    for cls in (L1SNRLoss, L1SNRDBLoss):
        assert torch.isnan(cls(name="t")(est, act)), f"{cls.__name__} unexpectedly scrubbed NaN"


# ---------------------------------------------------------------------------------------------
# T1-7 -- silence, the dominant failure mode of SNR-style losses
# ---------------------------------------------------------------------------------------------

def test_silent_estimate_gives_d1_exactly_zero():
    """S11: a silent estimate makes |0-y| = |y|, so the ratio is 1 and D1 is exactly 0.

    This is the pathology the level-matching regularizer exists to fix: a model that outputs silence scores
    the same as one with error equal to the signal.
    """
    act = audio(2, 4096, level=0.1)
    est = torch.zeros_like(act).requires_grad_(True)
    loss = L1SNRLoss(name="t")(est, act)
    assert abs(loss.item()) < 1e-5, f"D1 for a silent estimate should be 0, got {loss.item()}"
    loss.backward()
    assert est.grad.abs().sum() > 0, "a silent estimate must still carry gradient"


def test_regularizer_penalizes_a_silent_estimate_asymmetrically():
    """S11: under-level is penalized far harder than over-level, which is the intent of eq. (10)."""
    act = audio(2, 4096, level=0.1)
    silent = torch.zeros_like(act)
    loud = act * 10.0
    reg_on = L1SNRDBLoss(name="t", use_regularization=True)
    under = reg_on(silent, act).item()
    over = reg_on(loud, act).item()
    assert under > over, f"silencing a real target ({under}) should cost more than over-levelling it ({over})"


@pytest.mark.parametrize("case", ["target", "estimate", "both"])
def test_silence_does_not_produce_nan(case):
    a = audio(2, 4096, level=0.1)
    e = a + audio(2, 4096, level=0.01, seed=2)
    if case in ("target", "both"):
        a = torch.zeros_like(a)
    if case in ("estimate", "both"):
        e = torch.zeros_like(e)
    for cls in ALL_CLASSES:
        loss = cls(name="t")(e, a)
        assert not torch.isnan(loss), f"{cls.__name__} produced NaN on silent {case}"
        assert not torch.isinf(loss), f"{cls.__name__} produced Inf on silent {case}"


def test_dbrms_floor_is_below_lmin_for_silence():
    """S10: dbrms_eps=1e-8 puts the floor near -80 dB, deliberately below the lmin=-60 threshold, so a
    digitally silent target is recognized as silent and eta correctly goes to 0."""
    silent = torch.zeros(2, 4096)
    level = dbrms(silent)
    assert level.max().item() < -60.0, f"silence reads {level.max().item()} dB, not below lmin=-60"
    assert abs(level.max().item() - (-79.9991)) < 1e-3


# ---------------------------------------------------------------------------------------------
# T1-7 -- shapes, dtypes, boundaries
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
def test_batch_of_one_works(cls):
    est, act = audio(1, 4096), audio(1, 4096, seed=1)
    assert cls(name="t")(est, act).ndim == 0


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
def test_empty_batch_raises(cls):
    """M18: B=0 raises a cryptic reshape error. Raising is right; the message is not actionable."""
    with pytest.raises(RuntimeError):
        cls(name="t")(torch.zeros(0, 4096), torch.zeros(0, 4096))


@pytest.mark.parametrize("length", [4095, 4097, 999])
def test_odd_lengths_work(length):
    est, act = audio(2, length), audio(2, length, seed=1)
    for cls in ALL_CLASSES:
        assert cls(name="t")(est, act).ndim == 0


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_min_audio_length_boundary(delta):
    """M5: the fallback boundary is discontinuous, and 512-1024 samples silently lose resolutions."""
    n = 512 + delta
    est, act = audio(2, n), audio(2, n, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = STFTL1SNRDBLoss(name="t", min_audio_length=512)(est, act)
    assert loss.ndim == 0 and not torch.isnan(loss)


def test_fallback_boundary_is_discontinuous():
    """M5, quantified: one extra sample roughly doubles the value, because the STFT path sums Re+Im
    while the time-domain fallback returns a single D1. Pinned so A5 documents rather than hides it."""
    g = torch.Generator().manual_seed(0)
    full = torch.randn(2, 513, generator=g) * 0.05
    est_full = full + torch.randn(2, 513, generator=g) * 0.005
    loss_fn = STFTL1SNRDBLoss(name="t", min_audio_length=512)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        below = loss_fn(est_full[:, :511], full[:, :511]).item()   # time fallback
        above = loss_fn(est_full[:, :512], full[:, :512]).item()   # STFT path
    assert 1.5 < above / below < 2.5, f"expected roughly a 2x jump across the boundary, got {above/below:.3f}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved_in_the_time_domain(dtype):
    est = audio(2, 4096).to(dtype)
    act = audio(2, 4096, seed=1).to(dtype)
    for cls in (L1SNRLoss, L1SNRDBLoss):
        assert cls(name="t")(est, act).dtype == dtype


def test_stft_downcasts_float64_today():
    """M13: STFTL1SNRDBLoss silently returns float32 from float64 input while its siblings do not.

    Pinned as current behaviour; A9 fixes it and this test inverts.
    """
    est = audio(2, 4096).double()
    act = audio(2, 4096, seed=1).double()
    assert STFTL1SNRDBLoss(name="t")(est, act).dtype == torch.float32


def test_non_contiguous_input_works():
    base = audio(2, 8192)
    est = base[:, ::2]                     # a strided view
    act = audio(2, 8192, seed=1)[:, ::2]
    assert not est.is_contiguous()
    for cls in ALL_CLASSES:
        assert cls(name="t")(est, act).ndim == 0


# ---------------------------------------------------------------------------------------------
# T1-8 -- the spectrogram regularizer, never executed by any test before now
# ---------------------------------------------------------------------------------------------

def test_spectrogram_regularizer_runs_and_changes_the_loss():
    """M11: use_regularization=True was never passed to STFTL1SNRDBLoss anywhere in 127 tests, leaving the
    entire novel spectrogram regularizer at 0% coverage."""
    est, act = audio(2, 8192), audio(2, 8192, seed=1)
    off = STFTL1SNRDBLoss(name="t", use_regularization=False)(est, act)
    on = STFTL1SNRDBLoss(name="t", use_regularization=True)(est, act)
    assert on.item() != off.item(), "enabling the spectrogram regularizer changed nothing"
    assert on.item() > off.item(), "the regularizer is a penalty, so it should increase the loss"


def test_spec_reg_coef_scales_the_regularizer():
    est, act = audio(2, 8192), audio(2, 8192, seed=1)
    base = STFTL1SNRDBLoss(name="t", use_regularization=False)(est, act).item()
    small = STFTL1SNRDBLoss(name="t", use_regularization=True, spec_reg_coef=0.1)(est, act).item()
    large = STFTL1SNRDBLoss(name="t", use_regularization=True, spec_reg_coef=1.0)(est, act).item()
    assert large - base > small - base > 0
    # the coefficient is linear in the regularizer contribution
    assert abs((large - base) / (small - base) - 10.0) < 0.1


def test_spectrogram_regularizer_penalizes_a_silent_estimate():
    est = torch.zeros(2, 8192)
    act = audio(2, 8192, level=0.1)
    off = STFTL1SNRDBLoss(name="t", use_regularization=False)(est, act).item()
    on = STFTL1SNRDBLoss(name="t", use_regularization=True)(est, act).item()
    assert on > off, "the spectral regularizer should penalize a silent estimate"


def test_spectrogram_regularizer_is_reachable_from_multi():
    """Q1: spec_reg_coef was unreachable from MultiL1SNRDBLoss except through spec_loss_params."""
    est, act = audio(2, 8192), audio(2, 8192, seed=1)
    a = MultiL1SNRDBLoss(name="t", use_spec_regularization=True, spec_reg_coef=0.1)
    b = MultiL1SNRDBLoss(name="t", use_spec_regularization=True, spec_reg_coef=1.0)
    assert a.spec_loss.spec_reg_coef == 0.1 and b.spec_loss.spec_reg_coef == 1.0
    assert a(est, act).item() != b(est, act).item()


@pytest.mark.parametrize("level,expected_floor", [
    # D1's floor at perfect reconstruction is 10*log10(eps / (mean|y| + eps)) with eps=1e-3. These are the
    # values documented in the README's Limitations section, computed from a constant-amplitude signal so
    # mean|y| equals the level exactly.
    (1.0, -30.0043),
    (0.1, -20.0432),
    (0.01, -10.4139),
    (0.001, -3.0103),
])
def test_d1_floor_collapses_for_quiet_targets(level, expected_floor):
    """Section 5: the usable dynamic range shrinks with target level, so a quiet stem has almost no range.

    This is the mechanism behind Q27's observation that the objective is regularizer-dominated exactly where
    reconstruction has least room. Q27's own claim is about *gradient* magnitudes (lane D measured ~810x) and
    is not asserted here: a loss-value version of it is not equivalent, and I got the direction wrong when I
    first tried to state one. T3-2 measures the gradient balance properly.
    """
    act = torch.full((1, 4096), level)
    est = act.clone()                              # perfect reconstruction, so error is exactly 0
    floor = L1SNRLoss(name="t")(est, act).item()
    assert abs(floor - expected_floor) < 1e-3, (
        f"at mean|y|={level} the floor should be {expected_floor} dB, got {floor:.4f}")
    assert torch.allclose(L1SNRLoss(name="t")(est, act), reference.l1snr(est, act), atol=1e-6)


# ---------------------------------------------------------------------------------------------
# T1-10 -- remaining reachable branches
# ---------------------------------------------------------------------------------------------

def test_frame_count_validation_rejects_audio_shorter_than_the_hop():
    """_validate_audio_length's second check: fewer than 2 frames means the STFT cannot run.

    Reachable only when min_audio_length is set below the hop length, so the length check passes and the
    frame-count check is what rejects it.
    """
    loss_fn = STFTL1SNRDBLoss(name="t", n_ffts=[2048], hop_lengths=[1024], win_lengths=[2048],
                              min_audio_length=1)
    assert loss_fn._validate_audio_length(500) is False      # 500 // 1024 + 1 == 1 frame
    assert loss_fn._validate_audio_length(4096) is True
    # and the forward path takes the time-domain fallback rather than failing
    est, act = audio(2, 500), audio(2, 500, seed=1)
    assert loss_fn(est, act).ndim == 0


def test_time_and_spec_loss_params_override_the_shared_defaults():
    """Q1/M11: the two override dicts were never exercised. They allow, e.g., a different lmin per domain."""
    loss_fn = MultiL1SNRDBLoss(
        name="t", lmin=-60.0,
        time_loss_params={"lmin": -40.0, "lambda0": 0.5},
        spec_loss_params={"lmin": -20.0},
    )
    assert loss_fn.time_loss.lmin == -40.0, "time_loss_params was ignored"
    assert loss_fn.time_loss.lambda0 == 0.5
    assert loss_fn.spec_loss.lmin == -20.0, "spec_loss_params was ignored"
    # the override must actually change the computed loss, not just the attribute
    est, act = audio(2, 8192), audio(2, 8192, seed=1)
    plain = MultiL1SNRDBLoss(name="t", lmin=-60.0, use_time_regularization=True)
    assert loss_fn(est, act).item() != plain(est, act).item()
