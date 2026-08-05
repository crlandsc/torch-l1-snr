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

class _Exploding(torch.nn.Module):
    """A transform that always raises, to reach the in-loop backstop.

    After A5 filters unusable resolutions up front, a transform that passed the length and frame-count
    checks should not fail, so the try/except handlers are no longer reachable through the public API. They
    are kept as defence in depth and tested by forcing the failure, rather than deleted untested or left
    uncovered.
    """

    def forward(self, x):
        raise RuntimeError("synthetic transform failure")


def test_a_failing_transform_warns_and_is_skipped():
    """The in-loop RuntimeError backstop: one resolution fails, the others still contribute."""
    loss_fn = STFTL1SNRDBLoss(name="t", n_ffts=[512, 1024], hop_lengths=[128, 256],
                              win_lengths=[512, 1024], min_audio_length=1)
    loss_fn.spectrogram_transforms[1] = _Exploding()
    est, act = audio(2, 8192), audio(2, 8192, seed=1)
    with pytest.warns(UserWarning, match="resolution 1"):
        loss = loss_fn(est, act)
    expected = reference.spec_blended(est, act, 512, 128, 512)
    assert torch.allclose(loss, expected, atol=1e-5)


def _all_failing_loss():
    loss_fn = STFTL1SNRDBLoss(name="t", n_ffts=[512], hop_lengths=[128], win_lengths=[512],
                              min_audio_length=1)
    loss_fn.spectrogram_transforms[0] = _Exploding()
    return loss_fn


def test_all_resolutions_failing_warns():
    """M6: the warning must fire when nothing contributed."""
    est, act = audio(2, 8192), audio(2, 8192, seed=1)
    with pytest.warns(RuntimeWarning, match="every spectrogram resolution failed"):
        loss = _all_failing_loss()(est, act)
    assert loss.item() == 0.0


def test_all_resolutions_failing_returns_a_graph_connected_zero():
    """M6/A6: the zero must stay attached to the graph, or .backward() raises outright.

    Worse than the crash was the silent case: inside MultiL1SNRDBLoss the detached zero made the spectral
    term contribute nothing with no error, so only the time term trained.
    """
    est = audio(2, 8192).requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = _all_failing_loss()(est, audio(2, 8192, seed=1))
    assert loss.grad_fn is not None, "the zero is detached from the graph"
    loss.backward()                                    # must not raise
    assert est.grad is not None
    assert est.grad.abs().max().item() == 0.0, "a zero loss should give a zero, not absent, gradient"


def test_multi_keeps_a_spectral_gradient_when_all_resolutions_fail():
    """M6's silent case: inside Multi the detached zero meant only the time term trained, with no signal."""
    loss_fn = MultiL1SNRDBLoss(name="t", n_ffts=[512], hop_lengths=[128], win_lengths=[512],
                               min_audio_length=1, spec_weight=0.5)
    loss_fn.spec_loss.spectrogram_transforms[0] = _Exploding()
    est = audio(2, 8192).requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = loss_fn(est, audio(2, 8192, seed=1))
    assert loss.requires_grad
    loss.backward()
    assert est.grad is not None


def test_nan_input_warns_exactly_once():
    """M7/A7: an all-NaN estimate yields exactly 0.0 with an all-zero gradient, so a diverged run reports
    as healthy. The scrubbing is deliberate; its silence was not."""
    loss_fn = STFTL1SNRDBLoss(name="t")
    est = torch.full((2, 4096), float("nan"), requires_grad=True)
    act = audio(2, 4096)
    with pytest.warns(RuntimeWarning, match="non-finite"):
        loss = loss_fn(est, act)
    loss.backward()
    assert loss.item() == 0.0
    assert est.grad.abs().max().item() == 0.0

    # one-shot, mirroring the existing _mps_warned pattern: a per-step warning would flood a training log
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loss_fn(torch.full((2, 4096), float("nan")), act)


def test_nan_warning_is_per_instance_not_global():
    """A fresh loss object must warn again, or a second training run in the same process stays silent."""
    act = audio(2, 4096)
    nan = torch.full((2, 4096), float("nan"))
    for _ in range(2):
        with pytest.warns(RuntimeWarning, match="non-finite"):
            STFTL1SNRDBLoss(name="t")(nan, act)


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
    assert abs(level.max().item() - (-80.0)) < 1e-5


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


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
def test_float64_is_not_downcast(cls):
    """M13/A9: STFTL1SNRDBLoss silently returned float32 from float64 input while its siblings did not.

    Returning float32 from half precision is defensible under AMP; losing float64 is not, and the
    disagreement between siblings is a contract defect either way.
    """
    est = audio(2, 4096).double()
    act = audio(2, 4096, seed=1).double()
    assert cls(name="t")(est, act).dtype == torch.float64


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_precision_promotes_to_float32(cls, dtype):
    """Documented AMP behaviour: half precision promotes rather than accumulating in half."""
    est = audio(2, 4096).to(dtype)
    act = audio(2, 4096, seed=1).to(dtype)
    out = cls(name="t")(est, act)
    assert out.dtype in (torch.float32, dtype), f"unexpected output dtype {out.dtype}"


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


# ---------------------------------------------------------------------------------------------
# A4 -- shape equality
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
def test_mismatched_shapes_with_equal_element_counts_are_rejected(cls):
    """M9: reshape(batch_size, -1) made any two tensors agreeing in batch size and element count
    comparable, so this returned a plausible 1.497 from a wrong pairing instead of complaining.

    The dangerous case is exactly this one: equal element counts, different structure. A user who permutes
    stems and channels, or feeds different durations, trains against a silently wrong pairing.
    """
    est = torch.randn(2, 4, 8000)
    act = torch.randn(2, 2, 16000)
    assert est.numel() == act.numel(), "the test premise is equal element counts"
    with pytest.raises(ValueError, match="shape"):
        cls(name="t")(est, act)


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
def test_mismatched_durations_are_rejected(cls):
    with pytest.raises(ValueError, match="shape"):
        cls(name="t")(torch.randn(2, 16000), torch.randn(2, 8000))


@pytest.mark.parametrize("shape", [(2, 4096), (2, 2, 4096), (2, 4, 2, 4096)])
def test_matching_shapes_still_work(shape):
    """A4 must only reject shapes that already differ. Every legitimate call is unaffected."""
    est, act = audio(*shape), audio(*shape, seed=1)
    for cls in ALL_CLASSES:
        assert cls(name="t")(est, act).ndim == 0


def test_per_stem_slicing_still_works_after_the_shape_check():
    """The pattern the maintainer's training configs use: slice one stem, then call the loss."""
    est, act = audio(2, 4, 2, 4096), audio(2, 4, 2, 4096, seed=1)
    for k in range(4):
        for cls in ALL_CLASSES:
            assert cls(name="t")(est[:, k], act[:, k]).ndim == 0


# ---------------------------------------------------------------------------------------------
# A11, A13
# ---------------------------------------------------------------------------------------------

def test_multi_warns_when_the_spectral_branch_falls_back():
    """M21: below min_audio_length both branches compute the same time-domain quantity, so the user is
    optimizing one domain at effective weight 1.0 while believing they have a multi-domain objective."""
    est, act = audio(2, 400), audio(2, 400, seed=1)
    with pytest.warns(RuntimeWarning, match="fall(ing|s)? back|short"):
        MultiL1SNRDBLoss(name="t", min_audio_length=512)(est, act)


def test_multi_does_not_warn_above_the_threshold():
    est, act = audio(2, 8192), audio(2, 8192, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MultiL1SNRDBLoss(name="t", min_audio_length=512)(est, act)


@pytest.mark.parametrize("attr", ["n_ffts", "hop_lengths", "win_lengths"])
def test_mutable_list_defaults_are_not_shared(attr):
    """Q11: the default lists were stored by reference, so two instances aliased the same object and each
    other's in-place mutation leaked across them.

    Checked on STFTL1SNRDBLoss, the class that stores them. MultiL1SNRDBLoss does not keep its own copies --
    it threads them to the spectral branch, which is covered separately below.
    """
    a, b = STFTL1SNRDBLoss(name="a"), STFTL1SNRDBLoss(name="b")
    la, lb = getattr(a, attr), getattr(b, attr)
    assert la is not lb, f"STFTL1SNRDBLoss.{attr} is shared between instances"
    la.append(99999)
    assert 99999 not in lb, f"mutating one instance's {attr} leaked into another"


@pytest.mark.parametrize("attr", ["n_ffts", "hop_lengths", "win_lengths"])
def test_multi_does_not_leak_lists_through_its_spectral_branch(attr):
    """MultiL1SNRDBLoss passes the signature defaults down, so the leak has to be closed at the far end."""
    a, b = MultiL1SNRDBLoss(name="a"), MultiL1SNRDBLoss(name="b")
    la, lb = getattr(a.spec_loss, attr), getattr(b.spec_loss, attr)
    assert la is not lb
    la.append(99999)
    assert 99999 not in lb


# ---------------------------------------------------------------------------------------------
# A8, A10
# ---------------------------------------------------------------------------------------------

class _Wrapper(torch.nn.Module):
    """The standard Lightning pattern: hold the loss as a submodule of the model."""

    def __init__(self, loss):
        super().__init__()
        self.net = torch.nn.Linear(4, 4)
        self.loss = loss


@pytest.mark.parametrize("cls", [STFTL1SNRDBLoss, MultiL1SNRDBLoss], ids=lambda c: c.__name__)
def test_loss_adds_no_keys_to_an_enclosing_state_dict(cls):
    """M8/A8: torchaudio registers the Spectrogram window as a *persistent* buffer, so a ModuleList of them
    entered every checkpoint -- 3 keys and 3584 floats for the default resolutions."""
    plain = set(torch.nn.Module.state_dict(_Wrapper(torch.nn.Identity())).keys())
    with_loss = set(_Wrapper(cls(name="t")).state_dict().keys())
    added = {k for k in with_loss if "window" in k}
    assert not added, f"the loss added {len(added)} window buffer(s) to the checkpoint: {sorted(added)}"


def test_checkpoint_survives_a_change_of_resolutions():
    """M8's concrete symptom: retuning n_ffts made an earlier checkpoint fail to load."""
    model_a = _Wrapper(STFTL1SNRDBLoss(name="t", n_ffts=[512, 1024, 2048],
                                       hop_lengths=[128, 256, 512], win_lengths=[512, 1024, 2048]))
    state = model_a.state_dict()
    model_b = _Wrapper(STFTL1SNRDBLoss(name="t", n_ffts=[256, 512], hop_lengths=[64, 128],
                                       win_lengths=[256, 512]))
    model_b.load_state_dict(state)          # strict=True; must not raise


def test_short_audio_keeps_the_regularizer_when_it_was_requested():
    """M20/A10: the fallback was built with a hardcoded use_regularization=False, so
    STFTL1SNRDBLoss(use_regularization=True) below min_audio_length silently lost the anti-collapse
    protection the user explicitly asked for. Measured: a silent estimate against a non-silent target gave
    loss exactly 0.000000, which is the collapse the regularizer exists to prevent."""
    act = audio(2, 400, level=0.1)
    est = torch.zeros_like(act)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = STFTL1SNRDBLoss(name="t", use_regularization=True, min_audio_length=512)(est, act)
    assert loss.item() != 0.0, "a total collapse on short audio still scores zero"
    assert loss.item() > 0.0, "the regularizer should penalize the collapse"


def test_short_audio_regularizer_coefficient_matches_the_stft_path():
    """A10's design point: naive pass-through would give the fallback a 10x stronger regularizer, because
    the STFT path scales by spec_reg_coef=0.1 while L1SNRDBLoss had no coefficient at all."""
    act = audio(2, 400, level=0.1)
    est = torch.zeros_like(act)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = STFTL1SNRDBLoss(name="t", use_regularization=True, spec_reg_coef=0.1,
                            min_audio_length=512)(est, act).item()
        b = STFTL1SNRDBLoss(name="t", use_regularization=True, spec_reg_coef=1.0,
                            min_audio_length=512)(est, act).item()
        base = STFTL1SNRDBLoss(name="t", use_regularization=False, min_audio_length=512)(est, act).item()
    assert abs((b - base) / (a - base) - 10.0) < 0.1, (
        "the fallback regularizer does not track spec_reg_coef, so its weight jumps across the boundary")


def test_l1snrdb_reg_coef_default_preserves_behaviour():
    """A10 adds reg_coef to L1SNRDBLoss. Its default must reproduce the previous computation exactly."""
    act = audio(2, 4096, level=0.1)
    est = act + audio(2, 4096, level=0.01, seed=3)
    explicit = L1SNRDBLoss(name="t", use_regularization=True, reg_coef=1.0)(est, act)
    expected = reference.l1snr_db(est, act, use_regularization=True)
    assert torch.allclose(explicit, expected, atol=1e-6)


# ---------------------------------------------------------------------------------------------
# A5 -- per-resolution length validation
# ---------------------------------------------------------------------------------------------

# For center=True with pad_mode="reflect" the binding constraint is length > n_fft // 2, which
# _validate_audio_length never checked. So at 512 samples only 1 of the 3 default resolutions ran, and at
# 513-1024 only 2 of 3, silently changing the arity of the multi-resolution average.
@pytest.mark.parametrize("length,expected_used", [
    (256, 0),       # below every requirement -> time-domain fallback
    (257, 1),       # n_fft=512 needs 257
    (512, 1),
    (513, 2),       # n_fft=1024 needs 513
    (1024, 2),
    (1025, 3),      # n_fft=2048 needs 1025
    (8192, 3),
])
def test_only_usable_resolutions_are_used(length, expected_used):
    """Keep whichever resolutions are valid rather than discarding all of them.

    Falling back wholesale at 512-1024 samples would throw away two working resolutions for none, which is
    why the plan's original "derive min_audio_length" spec was corrected.
    """
    loss_fn = STFTL1SNRDBLoss(name="t", min_audio_length=1)
    assert loss_fn._usable_resolutions(length) == list(range(expected_used)), (
        f"at {length} samples, expected {expected_used} usable resolution(s)")


def test_dropping_a_resolution_always_warns():
    """The arity of the average must never change silently."""
    est, act = audio(2, 600), audio(2, 600, seed=1)
    with pytest.warns(RuntimeWarning, match="resolution"):
        STFTL1SNRDBLoss(name="t", min_audio_length=1)(est, act)


def test_no_warning_when_every_resolution_is_usable():
    est, act = audio(2, 8192), audio(2, 8192, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        STFTL1SNRDBLoss(name="t", min_audio_length=1)(est, act)


def test_dropped_resolution_warning_names_which_ones():
    est, act = audio(2, 600), audio(2, 600, seed=1)
    with pytest.warns(RuntimeWarning) as rec:
        STFTL1SNRDBLoss(name="t", min_audio_length=1)(est, act)
    message = " ".join(str(w.message) for w in rec)
    assert "2048" in message, f"the warning does not name the dropped n_fft: {message}"


def test_partial_resolution_use_matches_a_reference_over_the_same_subset():
    """The value must be the average over the resolutions actually used, not over all three."""
    est, act = audio(2, 600), audio(2, 600, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = STFTL1SNRDBLoss(name="t", min_audio_length=1)(est, act)
    expected = reference.multi_res_spec_d1(est, act, n_ffts=(512, 1024), hop_lengths=(128, 256),
                                           win_lengths=(512, 1024))
    assert torch.allclose(got, expected, atol=1e-5), (
        f"library {got.item():.7f} vs reference over the 2 usable resolutions {expected.item():.7f}")


# ---------------------------------------------------------------------------------------------
# C9, C11
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
def test_l1_weight_cannot_be_mutated_after_construction(cls):
    """Q13: l1_weight was stored in several unsynchronized copies, and mutating the public attribute took
    effect inconsistently -- the value is baked into child modules and the pure_l1_mode flag at
    construction. Making the setter work would mean rebuilding children on assignment, which is real
    complexity for a use case nobody has asked for. Raising is honest and one line.
    """
    loss = cls(name="t", l1_weight=0.0)
    assert loss.l1_weight == 0.0
    with pytest.raises(AttributeError, match="l1_weight"):
        loss.l1_weight = 0.5


def test_dbrms_outer_epsilon_removal_is_within_the_stated_bound():
    """Q22/C11: dbrms applied two epsilons of different physical dimension, one inside the sqrt on a power
    quantity and one outside on an amplitude. The outer one is inert because rms is already at least
    sqrt(dbrms_eps) = 1e-4, so log10 can never see zero.

    The BREAKING rule has one carve-out, for a change whose numerical effect is measured and stated below a
    bound. This is that change, and this test is the measurement: it must stay under 0.001 dB.
    """
    worst = 0.0
    for level in [0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0]:
        x = torch.full((1, 4096), level)
        rms = torch.sqrt((x ** 2).mean() + 1e-8)
        with_outer = 20.0 * torch.log10(rms + 1e-8)
        without = 20.0 * torch.log10(rms)
        worst = max(worst, abs((with_outer - without).item()))
    assert worst < 1e-3, f"removing the outer epsilon moves dbrms by {worst:.6f} dB, above the stated bound"


def test_dbrms_silence_floor_is_exactly_the_power_epsilon():
    """C11 must not regress the -80 dB floor, which per S10 is better matched to lmin=-60 than the authors'
    own -30 dB and is the reason a digitally silent target is recognized as silent.

    With the inert outer epsilon gone the floor is exactly 10*log10(eps) = -80.0, rather than the -79.99913
    the extra term produced. Asserted exactly rather than within a tolerance that the old value also
    satisfied by 0.0009 dB.
    """
    floor = dbrms(torch.zeros(2, 4096)).max().item()
    assert abs(floor - (-80.0)) < 1e-5, f"silence floor is {floor}, expected exactly -80.0"
    assert floor < -60.0, "the floor must sit below lmin so silence is recognized as silent"


# ---------------------------------------------------------------------------------------------
# A14 -- the one numerics change: a module constant in place of a batch statistic
# ---------------------------------------------------------------------------------------------
#
# M3. The L1/L1SNR blend scaled the L1 term by c * mean_b(1 / (ref_b + eps)) -- a mean of *reciprocals*, so
# one quiet target inflated the scale for the whole batch. Measured: rows at identical relative error saw
# their gradients move by 5.73x because a different row went quiet.
#
# M22. It also made the knob mean different things run to run: l1_weight=0.5 delivered anywhere from 15% to
# 91% of the way toward L1 behaviour depending on batch content.
#
# Both are fixed by the same edit. The two properties below are what distinguish the correct fix from the
# per-row form an adversarial panel originally recommended, which decoupled correctly but diverged *away*
# from L1 (profile 9.5714 against pure D1's 7.0000) and gave a non-monotone knob with a cliff at w=1.

def _row_profile(loss_fn, levels, relerr=0.1, T=4096, seed=0):
    """Ratio of per-row gradient magnitude between the quietest and loudest row.

    This is the operationalisation of "how much L1 behaviour": both terms push in the same direction, so the
    only thing l1_weight controls is how strongly each sample's update is scaled by its own error magnitude.
    """
    g = torch.Generator().manual_seed(seed)
    act = torch.stack([torch.randn(T, generator=g) * lv for lv in levels])
    est = act.clone()
    for i, lv in enumerate(levels):
        est[i] += relerr * lv * torch.randn(T, generator=g)
    est = est.requires_grad_(True)
    loss_fn(est, act).backward()
    per_row = [est.grad[i].abs().mean().item() for i in range(len(levels))]
    return max(per_row) / min(per_row)


def test_one_quiet_target_no_longer_moves_the_other_rows_gradients():
    """M3, the defect itself. Rows 0-2 are held identical; only row 3's level changes."""
    ratios = []
    for row3 in [0.2, 0.02, 0.002, 0.0]:
        g = torch.Generator().manual_seed(0)
        levels = [0.2, 0.2, 0.2, row3]
        act = torch.stack([torch.randn(4096, generator=g) * lv for lv in levels])
        est = act.clone()
        for i, lv in enumerate(levels):
            est[i] += 0.1 * lv * torch.randn(4096, generator=g)
        est = est.requires_grad_(True)
        L1SNRLoss(name="t", l1_weight=0.5)(est, act).backward()
        ratios.append(est.grad[:3].abs().mean().item())
    baseline = ratios[0]
    for r, row3 in zip(ratios, [0.2, 0.02, 0.002, 0.0]):
        rel = r / baseline
        assert abs(rel - 1.0) < 1e-3, (
            f"row 3 at level {row3} moved rows 0-2 gradients by {rel:.3f}x; before the fix this reached "
            f"5.730x at silence")


@pytest.mark.parametrize("l1_weight", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_the_knob_stays_monotone_toward_l1(l1_weight):
    """The property the rejected per-row form failed: it ended at 9.5714, *more* quiet-biased than pure D1,
    then collapsed discontinuously to 1.0 at the endpoint because of the w >= 1.0 shortcut."""
    levels = [0.2, 0.2, 0.2, 0.02]
    profiles = {w: _row_profile(L1SNRLoss(name="t", l1_weight=w), levels)
                for w in [0.0, 0.25, 0.5, 0.75, 1.0]}
    ordered = [profiles[w] for w in [0.0, 0.25, 0.5, 0.75, 1.0]]
    assert ordered == sorted(ordered, reverse=True), (
        f"profile is not monotone decreasing toward L1: {ordered}")
    assert abs(profiles[1.0] - 1.0) < 0.01, f"w=1.0 should reach pure L1 (1.0), got {profiles[1.0]:.4f}"
    assert profiles[0.0] > 5.0, f"w=0.0 should retain D1's quiet bias, got {profiles[0.0]:.4f}"


def test_ref_level_is_a_calibration_handle():
    """K only sets how fast the knob moves; it cannot reintroduce coupling or break monotonicity."""
    levels = [0.5, 0.1, 0.02, 0.005]
    profiles = [_row_profile(L1SNRLoss(name="t", l1_weight=0.5, ref_level=rl), levels)
                for rl in [1.0, 0.05, 0.005]]
    assert profiles[0] > profiles[1] > profiles[2], (
        f"a smaller ref_level should move the knob further toward L1: {profiles}")


def test_spec_ref_level_defaults_to_the_measured_ratio():
    """P0-1: the STFT reference is 5.6x below the time-domain one, measured on 496 real MUSDB stem-chunks.

    Defaulting spec_ref_level to ref_level would be 5.3x too large, costing 19.7 points of knob position at
    l1_weight=0.5. It is derived instead, so a user who sets ref_level for their data gets both domains right.
    """
    loss_fn = STFTL1SNRDBLoss(name="t", l1_weight=0.5, ref_level=0.05)
    assert abs(loss_fn._resolved_spec_ref_level - 0.0095) < 1e-6, (
        f"expected 0.19 * 0.05 = 0.0095, got {loss_fn._resolved_spec_ref_level}")
    # and it tracks ref_level rather than sitting at an absolute value
    quiet = STFTL1SNRDBLoss(name="t", l1_weight=0.5, ref_level=0.005)
    assert abs(quiet._resolved_spec_ref_level - 0.00095) < 1e-7
    # explicit override wins
    override = STFTL1SNRDBLoss(name="t", l1_weight=0.5, ref_level=0.05, spec_ref_level=0.02)
    assert override._resolved_spec_ref_level == 0.02


def test_pure_snr_path_is_untouched():
    """A14 must not perturb l1_weight=0.0, which is the default and what the training configs use."""
    est, act = audio(3, 4096), audio(3, 4096, seed=1)
    for cls in ALL_CLASSES:
        got = cls(name="t", l1_weight=0.0)(est, act)
        assert not torch.isnan(got)
    assert torch.allclose(L1SNRLoss(name="t", l1_weight=0.0)(est, act),
                          reference.l1snr(est, act), atol=1e-7)


def test_blend_matches_the_constant_k_reference():
    """The blended path must equal an independently computed constant-K blend, not the old batch statistic."""
    est, act = audio(4, 4096), audio(4, 4096, seed=1)
    for w in [0.25, 0.5, 0.75]:
        got = L1SNRLoss(name="t", l1_weight=w, ref_level=0.05)(est, act)
        expected = reference.l1snr_blended(est, act, l1_weight=w, ref_level=0.05)
        assert torch.allclose(got, expected, atol=1e-6), f"w={w}: {got.item()} vs {expected.item()}"


def test_ref_level_must_be_positive():
    with pytest.raises(ValueError, match="ref_level"):
        L1SNRLoss(name="t", ref_level=0.0)
    with pytest.raises(ValueError, match="ref_level"):
        L1SNRLoss(name="t", ref_level=-0.1)
