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


def expected(cls, est, act, **kw):
    """The value `cls` should return, computed from tests/reference.py rather than from the library.

    Exists so that shape, dtype and edge-case tests can assert a *number*. An adversarial reviewer found 40
    tests here that called a loss and then checked only `ndim == 0`, `not isnan`, or `.dtype` -- all of which
    a constant satisfies. That is audit finding M2 reappearing in the file written to prevent it.
    """
    name = cls.__name__
    if name == "L1SNRLoss":
        return reference.l1snr_blended(est, act, l1_weight=kw.get("l1_weight", 0.0),
                                       ref_level=kw.get("ref_level", 0.05))
    if name == "L1SNRDBLoss":
        return reference.l1snr_db(est, act, l1_weight=kw.get("l1_weight", 0.0),
                                  use_regularization=kw.get("use_regularization", True),
                                  ref_level=kw.get("ref_level", 0.05))
    if name == "STFTL1SNRDBLoss":
        return reference.multi_res_spec_d1(est, act, l1_weight=kw.get("l1_weight", 0.0),
                                           spec_ref_level=kw.get("spec_ref_level", 0.19 * 0.05))
    sw = kw.get("spec_weight", 0.5)
    return ((1.0 - sw) * reference.l1snr_db(est, act, l1_weight=kw.get("l1_weight", 0.0),
                                            use_regularization=kw.get("use_time_regularization", True),
                                            ref_level=kw.get("ref_level", 0.05))
            + sw * reference.multi_res_spec_d1(est, act, l1_weight=kw.get("l1_weight", 0.0),
                                                spec_ref_level=kw.get("spec_ref_level", 0.19 * 0.05)))



# ---------------------------------------------------------------------------------------------
# T1-5 -- validation paths
# ---------------------------------------------------------------------------------------------

@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("bad", [-0.5, -1e-9, 1.5, 2.0])
def test_l1_weight_out_of_range_is_rejected(cls, bad):
    """M14/A2: all four classes now validate. L1SNRLoss was the only one that did not, and silently took
    the pure-SNR branch for -0.5 and the pure-L1 branch for 2.0."""
    with pytest.raises(ValueError, match="l1_weight"):
        cls(name="t", l1_weight=bad)


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("good", [0.0, 0.5, 1.0])
def test_l1_weight_endpoints_are_accepted(cls, good):
    """The range is inclusive. Guards against validation that is too strict."""
    assert cls(name="t", l1_weight=good) is not None


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
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


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
@pytest.mark.parametrize("good", [0.0, 0.5, 1.0])
def test_spec_weight_endpoints_are_accepted(good):
    assert MultiL1SNRDBLoss(name="t", spec_weight=good) is not None


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
def test_time_coefficient_never_goes_negative():
    """M19 stated as the invariant rather than the symptom: no accepted spec_weight may invert the sign."""
    for sw in [0.0, 0.25, 0.5, 0.75, 1.0]:
        loss_fn = MultiL1SNRDBLoss(name="t", spec_weight=sw)
        assert (1.0 - loss_fn.spec_weight) >= 0.0


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
def test_mismatched_stft_parameter_list_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        STFTL1SNRDBLoss(name="t", n_ffts=[512, 1024], hop_lengths=[128], win_lengths=[512, 1024])


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
def test_win_length_larger_than_n_fft_is_rejected():
    with pytest.raises(ValueError, match="win_length"):
        STFTL1SNRDBLoss(name="t", n_ffts=[512], hop_lengths=[128], win_lengths=[1024])


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
def test_unknown_window_fn_is_rejected_with_the_valid_options():
    """Q25/A12: surfaced as an opaque AttributeError about a mangled attribute name."""
    with pytest.raises(ValueError) as exc:
        STFTL1SNRDBLoss(name="t", window_fn="blackman_harris")
    message = str(exc.value)
    for valid in ["hann", "hamming", "blackman", "bartlett", "kaiser"]:
        assert valid in message, f"the error message does not name {valid!r} as a valid option"


@pytest.mark.no_forward  # reads source; never calls forward()
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


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
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
    assert loss.item() == 0.0, f"the loss should be exactly zero, got {loss.item()}"
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
    # with the spectral branch contributing exactly zero, the value is the weighted time term alone
    assert torch.allclose(loss, 0.5 * reference.l1snr_db(est, audio(2, 8192, seed=1)), atol=1e-5)
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
        # and on finite input the same loss must still be correct, so this test cannot pass against a
        # constant that merely happens to propagate NaN
        finite = act + audio(2, 4096, level=0.005, seed=2)
        assert torch.allclose(cls(name="t")(finite, act), expected(cls, finite, act), atol=1e-5)


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
        assert torch.allclose(loss, expected(cls, e, a), atol=1e-5), (
            f"{cls.__name__} on silent {case}: {loss.item():.7f} vs reference "
            f"{expected(cls, e, a).item():.7f}")


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
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
    got = cls(name="t")(est, act)
    assert got.ndim == 0
    assert torch.allclose(got, expected(cls, est, act), atol=1e-5)


@pytest.mark.parametrize("shape", [(0, 4096), (2, 0), (2, 1, 0), (2, 0, 4096)],
                         ids=["empty-batch", "empty-time", "empty-trailing", "empty-stem"])
@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
def test_empty_input_raises_actionably(cls, shape):
    """M18 established that empty input should raise; only an empty *batch* did.

    A zero-size non-batch dimension slipped through to `torch.mean` over an empty reduction, which is NaN --
    and `[B, 0, T]`, an empty stem selection, gave NaN from the time-domain classes but 0.0 from the
    spectrogram one. The empty batch did raise, but with a cryptic reshape RuntimeError this test's earlier
    version accepted while noting the message was not actionable. Both halves are fixed: every empty shape
    raises the same way, and the message names the offending tensor and its shape.
    """
    with pytest.raises(ValueError, match="empty"):
        cls(name="t")(torch.zeros(*shape), torch.zeros(*shape))


@pytest.mark.parametrize("length", [4095, 4097, 999])
def test_odd_lengths_work(length):
    est, act = audio(2, length), audio(2, length, seed=1)
    for cls in ALL_CLASSES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # 999 samples drops the largest resolution
            got = cls(name="t")(est, act)
        assert got.ndim == 0
        if length > 1024:                        # all resolutions usable, so the reference matches
            assert torch.allclose(got, expected(cls, est, act), atol=1e-5)
        else:
            # only the usable resolutions apply, so compare against the reference over that subset
            ref = (expected(cls, est, act) if cls in (L1SNRLoss, L1SNRDBLoss)
                   else reference.multi_res_spec_d1(est, act, n_ffts=(512, 1024),
                                                    hop_lengths=(128, 256), win_lengths=(512, 1024)))
            if cls is MultiL1SNRDBLoss:
                ref = 0.5 * reference.l1snr_db(est, act) + 0.5 * reference.multi_res_spec_d1(
                    est, act, n_ffts=(512, 1024), hop_lengths=(128, 256), win_lengths=(512, 1024))
            assert torch.allclose(got, ref, atol=1e-5), f"{cls.__name__} at length {length}"


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_min_audio_length_boundary(delta):
    """M5: the fallback boundary is discontinuous, and 512-1024 samples silently lose resolutions."""
    n = 512 + delta
    est, act = audio(2, n), audio(2, n, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = STFTL1SNRDBLoss(name="t", min_audio_length=512)(est, act)
    assert loss.ndim == 0 and not torch.isnan(loss)
    # which resolutions apply is a function of length: n_fft=512 needs 257 samples, 1024 needs 513,
    # 2048 needs 1025. So 511 falls back entirely, 512 uses one resolution, and 513 uses two.
    if n < 512:
        ref = reference.l1snr_db(est, act, use_regularization=False)
    elif n < 513:
        ref = reference.spec_blended(est, act, 512, 128, 512)
    else:
        ref = reference.multi_res_spec_d1(est, act, n_ffts=(512, 1024), hop_lengths=(128, 256),
                                          win_lengths=(512, 1024))
    assert torch.allclose(loss, ref, atol=1e-5), f"length {n}: {loss.item():.6f} vs {ref.item():.6f}"


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
        got = cls(name="t")(est, act)
        assert got.dtype == dtype
        assert torch.allclose(got, expected(cls, est, act).to(dtype), atol=1e-5)


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
def test_float64_is_not_downcast(cls):
    """M13/A9: STFTL1SNRDBLoss silently returned float32 from float64 input while its siblings did not.

    Returning float32 from half precision is defensible under AMP; losing float64 is not, and the
    disagreement between siblings is a contract defect either way.
    """
    est = audio(2, 4096).double()
    act = audio(2, 4096, seed=1).double()
    got = cls(name="t")(est, act)
    assert got.dtype == torch.float64
    assert torch.allclose(got, expected(cls, est, act).double(), atol=1e-5), (
        "float64 is preserved in dtype but the value is wrong")


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_precision_promotes_to_float32(cls, dtype):
    """Documented AMP behaviour: half precision promotes rather than accumulating in half."""
    # a *correlated* estimate, so the loss sits near -20 dB rather than near 0 dB. With independent
    # tensors the loss lands at about 1.47, close enough to a stubbed constant of 1.0 to slip through a
    # tolerance loose enough for half precision.
    act = audio(2, 4096, seed=1).to(dtype)
    est = (act.float() + audio(2, 4096, level=0.005, seed=2)).to(dtype)
    out = cls(name="t")(est, act)
    assert out.dtype in (torch.float32, dtype), f"unexpected output dtype {out.dtype}"
    # half precision is lossy, so the tolerance is loose -- but it still separates a real value from a
    # constant, which is what this needs to do
    ref = expected(cls, est.float(), act.float())
    assert torch.allclose(out.float(), ref, rtol=0.05, atol=0.5), (
        f"{cls.__name__} at {dtype}: {out.item():.4f} vs float32 reference {ref.item():.4f}")


def test_non_contiguous_input_works():
    base = audio(2, 8192)
    est = base[:, ::2]                     # a strided view
    act = audio(2, 8192, seed=1)[:, ::2]
    assert not est.is_contiguous()
    for cls in ALL_CLASSES:
        got = cls(name="t")(est, act)
        assert torch.allclose(got, expected(cls, est, act), atol=1e-5), (
            f"{cls.__name__} gives a different answer on a strided view than the reference")


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
    # hop must exceed n_fft // 2 for the frame count to be the deciding clause: otherwise the length
    # requirement (n_fft // 2 + 1) excludes the resolution first and the frame check is never reached.
    # With n_fft=512 the length requirement is 257, so 300 samples passes it, and hop=512 gives
    # 300 // 512 + 1 == 1 frame, which is what must reject it.
    loss_fn = STFTL1SNRDBLoss(name="t", n_ffts=[512], hop_lengths=[512], win_lengths=[512],
                              min_audio_length=1)
    assert loss_fn._usable_resolutions(300) == [], "the frame-count clause is not rejecting a 1-frame input"
    assert loss_fn._usable_resolutions(2048) == [0]
    # _usable_resolutions is the live selector. An earlier version of this test called
    # _validate_audio_length, which forward had stopped using, so it exercised dead code while the real
    # frame-count clause went unasserted. That method is now removed.
    est, act = audio(2, 300), audio(2, 300, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = loss_fn(est, act)
    assert torch.allclose(got, reference.l1snr_db(est, act, use_regularization=False), atol=1e-5), (
        "below every resolution's requirement the loss must be the time-domain fallback")


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
        got = cls(name="t")(est, act)
        assert got.ndim == 0
        assert torch.allclose(got, expected(cls, est, act), atol=1e-5)


def test_per_stem_slicing_still_works_after_the_shape_check():
    """The pattern the maintainer's training configs use: slice one stem, then call the loss."""
    est, act = audio(2, 4, 2, 4096), audio(2, 4, 2, 4096, seed=1)
    for k in range(4):
        for cls in ALL_CLASSES:
            got = cls(name="t")(est[:, k], act[:, k])
            assert torch.allclose(got, expected(cls, est[:, k], act[:, k]), atol=1e-5)


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
        got = MultiL1SNRDBLoss(name="t", min_audio_length=512)(est, act)
    assert torch.allclose(got, expected(MultiL1SNRDBLoss, est, act), atol=1e-5)


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
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


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
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


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
@pytest.mark.parametrize("cls", [STFTL1SNRDBLoss, MultiL1SNRDBLoss], ids=lambda c: c.__name__)
def test_loss_adds_no_keys_to_an_enclosing_state_dict(cls):
    """M8/A8: torchaudio registers the Spectrogram window as a *persistent* buffer, so a ModuleList of them
    entered every checkpoint -- 3 keys and 3584 floats for the default resolutions."""
    plain = set(torch.nn.Module.state_dict(_Wrapper(torch.nn.Identity())).keys())
    with_loss = set(_Wrapper(cls(name="t")).state_dict().keys())
    added = {k for k in with_loss if "window" in k}
    assert not added, f"the loss added {len(added)} window buffer(s) to the checkpoint: {sorted(added)}"


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
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
    # the fallback is a time-domain L1SNRDBLoss whose regularizer is scaled by spec_reg_coef
    ref = reference.l1snr_db(est, act, use_regularization=True) * 1.0
    reg_only = ref - reference.l1snr_db(est, act, use_regularization=False)
    expected_val = reference.l1snr_db(est, act, use_regularization=False) + 0.1 * reg_only
    assert torch.allclose(loss, expected_val, atol=1e-4), (
        f"{loss.item():.6f} vs expected {expected_val.item():.6f}")


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
@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
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
        got = STFTL1SNRDBLoss(name="t", min_audio_length=1)(est, act)
    assert torch.allclose(got, reference.multi_res_spec_d1(est, act), atol=1e-5)


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

@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
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


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
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


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
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


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
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


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
def test_ref_level_must_be_positive():
    with pytest.raises(ValueError, match="ref_level"):
        L1SNRLoss(name="t", ref_level=0.0)
    with pytest.raises(ValueError, match="ref_level"):
        L1SNRLoss(name="t", ref_level=-0.1)


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_ref_level_is_rejected(bad):
    """A NaN ref_level passes a `<= 0` check because NaN fails every comparison, and then yields a NaN
    scale. On the spectrogram path that surfaces as a loss of -0.0 with a warning blaming the FFT sizes,
    which sends the user to the wrong place entirely.

    Inf is rejected for the opposite reason: it makes the scale exactly zero, so `l1_weight` silently stops
    mixing in any L1 while appearing to be set.
    """
    for cls in ALL_CLASSES:
        with pytest.raises(ValueError, match="ref_level"):
            cls(name="t", l1_weight=0.5, ref_level=bad)


def test_inf_ref_level_would_have_silently_disabled_the_l1_term():
    """Pins why Inf must be rejected rather than merely discouraged: it is indistinguishable from working."""
    a = audio(2, 4096)
    e = a + audio(2, 4096, level=0.005, seed=1)
    pure_snr = L1SNRLoss(name="t", l1_weight=0.0)(e, a)
    blended = L1SNRLoss(name="t", l1_weight=0.5, ref_level=0.05)(e, a)
    assert torch.allclose(pure_snr, reference.l1snr_blended(e, a, l1_weight=0.0), atol=1e-6)
    assert torch.allclose(blended, reference.l1snr_blended(e, a, l1_weight=0.5, ref_level=0.05),
                          atol=1e-6)
    assert not torch.allclose(blended, 0.5 * pure_snr, atol=1e-6), (
        "with a valid ref_level the L1 term must contribute something")


@pytest.mark.no_forward  # constructs or inspects only; verified by tests/_forward_counter.py
def test_window_normalization_is_computed_in_double_precision():
    """The window fold must not degrade float64 accuracy.

    torchaudio builds the window in float32. Dividing it by its own float32 norm rounds twice, and a float64
    input then inherits that error -- which measurably worsens the float64 path, directly contradicting A9
    whose purpose is preserving float64. Normalizing in double and casting back avoids the second rounding.

    Reverting that detail passed all 356 tests when this was written, so the fix had no gate. This is it.

    Thresholds here are measured, not assumed: the achievable target is the exact normalization *of the
    float32 Hann window* (the float32 window itself is torchaudio's choice and not something this fixes).
    Against that target the stored window is off by 1.86e-09 and a float32 division by 2.06e-09.
    """
    stored = STFTL1SNRDBLoss(name="t", n_ffts=[2048], hop_lengths=[512],
                             win_lengths=[2048]).spectrogram_transforms[0].window
    w32 = torch.hann_window(2048)
    target = w32.double() / w32.double().pow(2).sum().sqrt()
    naive = (w32 / w32.pow(2).sum().sqrt()).double()

    err_stored = (stored.double() - target).abs().max().item()
    err_naive = (naive - target).abs().max().item()
    assert err_stored < err_naive, (
        f"the stored window is no closer to the exact normalized window than a plain float32 division "
        f"would be ({err_stored:.3e} vs {err_naive:.3e}); the double-precision normalization is gone")


def test_window_fold_preserves_float64_accuracy():
    """The same requirement end to end, against a fully-float64 reference.

    Catches the case where the double-precision normalization survives but a float32 rounding is
    reintroduced elsewhere in the window path. Threshold measured: the shipped code lands at 5.19e-10 and a
    float32 division at 9.6e-10, so 7e-10 separates them with margin on both sides.
    """
    g = torch.Generator().manual_seed(0)
    act = (torch.randn(2, 2, 8192, generator=g) * 0.05).double()
    est = act + (torch.randn(2, 2, 8192, generator=g) * 0.005).double()

    got = STFTL1SNRDBLoss(name="t", n_ffts=[512], hop_lengths=[128], win_lengths=[512])(est, act)

    w = torch.hann_window(512, dtype=torch.float64)
    B = est.shape[0]

    def spec(v):
        S = torch.stft(v.reshape(-1, v.shape[-1]), 512, 128, 512, w, center=True, pad_mode="reflect",
                       normalized=False, onesided=True, return_complex=True) / w.pow(2).sum().sqrt()
        return S.reshape(B, -1, *S.shape[-2:])

    Se, Sa = spec(est), spec(act)
    terms = []
    for pe, pa in ((Se.real, Sa.real), (Se.imag, Sa.imag)):
        pe, pa = pe.reshape(B, -1), pa.reshape(B, -1)
        terms.append(10 * torch.log10(((pe - pa).abs().mean(dim=1) + 1e-3)
                                      / (pa.abs().mean(dim=1) + 1e-3)))
    truth = (terms[0] + terms[1]).mean()

    rel = abs((got.double() - truth) / truth).item()
    assert rel < 7e-10, (
        f"float64 relative error against a fully-float64 reference is {rel:.3e}, above the 7e-10 bound; "
        "the window path has lost double precision somewhere")


# ---------------------------------------------------------------------------------------------
# Public parameter wiring. Every test below was added because a mutation campaign showed the
# parameter could be ignored, mis-wired or retuned with the whole suite still green.
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("spec_weight", [0.0, 0.25, 0.75, 1.0])
def test_spec_weight_applies_to_the_right_domain(spec_weight):
    """Swapping the two coefficients in MultiL1SNRDBLoss.forward survived the entire suite.

    Every existing value test used spec_weight=0.5, which is exactly the fixed point of that swap. A user
    setting 0.8 to emphasize spectral accuracy would have had 0.8 applied to the time domain instead. This is
    the immediate neighbour of M19, the audit's highest-priority finding.
    """
    est, act = audio(2, 8192), audio(2, 8192, seed=1)
    got = MultiL1SNRDBLoss(name="t", spec_weight=spec_weight)(est, act)
    time_ref = reference.l1snr_db(est, act, use_regularization=True)
    spec_ref = reference.multi_res_spec_d1(est, act)
    assert torch.allclose(got, (1.0 - spec_weight) * time_ref + spec_weight * spec_ref, atol=1e-5), (
        f"spec_weight={spec_weight} is not being applied to the spectrogram branch")
    # and the two domains must be distinguishable, or the assertion above proves nothing
    assert abs(time_ref.item() - spec_ref.item()) > 1.0


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("weight", [0.5, 2.0])
@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
def test_weight_multiplier_scales_the_loss(cls, weight, l1_weight):
    """`weight` is a documented public attribute on all four classes and was never passed as anything but
    1.0 by any test, so dropping it from any return statement survived the suite.

    Parametrized over l1_weight because each class has *several* return paths -- a pure-SNR branch, a blended
    branch and a pure-L1 shortcut -- each with its own `* self.weight`. A first version of this test used the
    default l1_weight only and still missed the multiplier being dropped from the pure-L1 shortcut.
    """
    est, act = audio(2, 8192), audio(2, 8192, seed=1)
    base = cls(name="t", l1_weight=l1_weight)(est, act)
    scaled = cls(name="t", l1_weight=l1_weight, weight=weight)(est, act)
    assert torch.allclose(scaled, weight * base, atol=1e-5), (
        f"{cls.__name__}(weight={weight}, l1_weight={l1_weight}) returned {scaled.item():.6f}, expected "
        f"{weight * base.item():.6f}")


def test_spectrogram_regularizer_is_asymmetric_like_the_time_domain_one():
    """Swapping L_pred and L_true inside _compute_spec_level_matching survived the suite.

    That swap reverses the anti-collapse direction, which is the regularizer's entire purpose: silencing a
    real target went from costing 30x more than over-levelling it to costing 0.75x. The time domain has this
    test; it had never been ported to the spectrogram path, where the existing test only asserted that the
    regularizer changes the loss at all.
    """
    act = audio(2, 8192, level=0.1)
    silent = torch.zeros_like(act)
    loud = act * 10.0
    off = STFTL1SNRDBLoss(name="t", use_regularization=False)
    on = STFTL1SNRDBLoss(name="t", use_regularization=True)
    under = (on(silent, act) - off(silent, act)).item()
    over = (on(loud, act) - off(loud, act)).item()
    assert under > 0 and over > 0
    assert under > 5.0 * over, (
        f"silencing a real target should cost far more than over-levelling it, got {under:.5f} vs "
        f"{over:.5f}; the anti-collapse asymmetry may be inverted")


@pytest.mark.parametrize("cls", [L1SNRDBLoss, MultiL1SNRDBLoss], ids=lambda c: c.__name__)
def test_ref_level_reaches_the_child_losses(cls):
    """Dropping `ref_level=ref_level` from L1SNRDBLoss's child constructor survived the suite: every test
    that varied ref_level used L1SNRLoss directly, and the wrappers were only ever given the default."""
    act = audio(2, 8192)
    est = act + audio(2, 8192, level=0.005, seed=1)
    a = cls(name="t", l1_weight=0.5, ref_level=0.05)(est, act)
    b = cls(name="t", l1_weight=0.5, ref_level=0.005)(est, act)
    assert not torch.allclose(a, b, atol=1e-4), (
        f"{cls.__name__} gives the same loss at ref_level 0.05 and 0.005, so the parameter is not reaching "
        "the component that uses it")


def test_lambda_is_stop_gradient():
    """The `.detach()` on the adaptive weight survived removal.

    An earlier probe missed it because the clamp saturates when the estimate is silent. In the unsaturated
    regime it is a 41% change in the regularizer's gradient, entirely silent. lambda is a *weight* derived
    from the levels; letting gradient flow through it changes what the regularizer optimizes.
    """
    act = audio(2, 8192, level=0.1)              # about -20 dBFS
    est = act * 0.316                            # about -30 dBFS: R/denom lands mid-range, no clamp
    e = est.clone().requires_grad_(True)
    L1SNRDBLoss(name="t", use_regularization=True)(e, act).backward()
    with_detach = e.grad.norm().item()

    # The inputs must require grad, or `lam.requires_grad` is False whether or not detach() is there and the
    # assertion proves nothing. That was the flaw in the first version of this test.
    live = est.clone().requires_grad_(True)
    L_pred = reference.dbrms(live)
    L_true = reference.dbrms(act)
    assert L_pred.requires_grad, "the test premise is a graph-connected L_pred"
    lam = L1SNRDBLoss.compute_adaptive_weight(L_pred, L_true, -60.0, 0.1, 0.9,
                                              (L_pred - L_true).abs())
    assert not lam.requires_grad, (
        "compute_adaptive_weight must return a detached weight; lambda is derived from the levels and "
        "letting gradient flow through it changes what the regularizer optimizes")
    assert with_detach > 0


@pytest.mark.no_forward  # calls the compute_adaptive_weight staticmethod directly
def test_adaptive_weight_is_bounded_by_lambda0_plus_delta():
    """Widening the clamp from (0, 1) to (0, 10) survived, multiplying the silence penalty by 1.45.

    lambda must stay within [lambda0, lambda0 + delta_lambda] by construction, or those two parameters stop
    describing the regularizer's range.
    """
    for scale in (1.0, 0.5, 0.1, 0.001, 0.0):
        act = audio(2, 4096, level=0.1)
        est = act * scale
        lam = L1SNRDBLoss.compute_adaptive_weight(
            reference.dbrms(est), reference.dbrms(act), -60.0, 0.1, 0.9,
            (reference.dbrms(est) - reference.dbrms(act)).abs())
        assert lam.min().item() >= 0.1 - 1e-6, f"lambda fell below lambda0 at scale {scale}: {lam.min()}"
        assert lam.max().item() <= 1.0 + 1e-6, (
            f"lambda exceeded lambda0 + delta_lambda at scale {scale}: {lam.max().item()}")


def test_spectrogram_level_uses_magnitude_not_the_real_part():
    """Replacing torch.abs(est_spec) with est_spec.real in the spectrogram regularizer survived.

    The level of a complex spectrogram is |S|. Using Re S alone discards the imaginary component and
    understates the level, which shifts every level-matching penalty.
    """
    act = audio(2, 8192, level=0.1)
    silent = torch.zeros_like(act)
    penalty = (STFTL1SNRDBLoss(name="t", use_regularization=True)(silent, act)
               - STFTL1SNRDBLoss(name="t", use_regularization=False)(silent, act)).item()
    # computed from |S|, independently of the library
    Sa = reference.normalized_stft(act.reshape(-1, act.shape[-1]), 512, 128, 512)
    mag = Sa.abs().reshape(act.shape[0], -1)
    L_true = 20 * torch.log10(torch.sqrt((mag ** 2).mean(dim=-1) + 1e-8) + 1e-8)
    L_pred = 20 * torch.log10(torch.sqrt(torch.zeros(act.shape[0]) + 1e-8) + 1e-8)
    R = (L_pred - L_true).abs()
    lam = reference.adaptive_lambda(L_pred, L_true)
    one_res = 0.1 * (lam * R).mean().item()
    assert abs(penalty - one_res) / one_res < 0.35, (
        f"spectrogram penalty {penalty:.5f} is not consistent with a magnitude-based level "
        f"(|S| gives about {one_res:.5f} for the n_fft=512 resolution)")


@pytest.mark.no_forward  # inspects defaults only
@pytest.mark.parametrize("param,expected_default,cls", [
    ("lambda0", 0.1, L1SNRDBLoss),
    ("delta_lambda", 0.9, L1SNRDBLoss),
    ("lmin", -60.0, L1SNRDBLoss),
    ("l1snr_eps", 1e-3, L1SNRDBLoss),
    ("dbrms_eps", 1e-8, L1SNRDBLoss),
    ("reg_coef", 1.0, L1SNRDBLoss),
    ("ref_level", 0.05, L1SNRLoss),
    ("eps", 1e-3, L1SNRLoss),
    ("spec_reg_coef", 0.1, STFTL1SNRDBLoss),
    ("min_audio_length", 512, STFTL1SNRDBLoss),
    ("spec_weight", 0.5, MultiL1SNRDBLoss),
    ("spec_reg_coef", 0.1, MultiL1SNRDBLoss),
])
def test_documented_defaults_are_pinned(param, expected_default, cls):
    """Six of these could be retuned with the whole suite green, because every test that used them passed
    an explicit value and never read the default.

    They are not arbitrary. lmin=-60 at -30 turns the anti-collapse protection almost off for quiet stems
    (penalty 0.1x); spec_reg_coef=0.1 is load-bearing for A10's claim that the regularizer's weight is
    continuous across the short-audio fallback boundary; min_audio_length=512 at 1024 silently switches a
    700-sample input to a different objective. Changing one is a deliberate act that should have to edit
    this list.
    """
    import inspect as _inspect
    actual = _inspect.signature(cls.__init__).parameters[param].default
    assert actual == expected_default, (
        f"{cls.__name__}.{param} default is {actual}, expected {expected_default}. If this change is "
        "intended, update this test and say why in the commit message.")


def test_lmin_default_keeps_anti_collapse_active_for_quiet_targets():
    """Why lmin=-60 rather than -30, stated as behaviour rather than as a number.

    At -30 a target at -50 dBFS reads as silent, eta goes to zero, and a collapsed estimate is penalized at
    lambda0 only: measured 0.1x the correct penalty.
    """
    act = audio(2, 8192, level=0.003)            # about -50 dBFS
    silent = torch.zeros_like(act)
    strict = (L1SNRDBLoss(name="t", use_regularization=True, lmin=-60.0)(silent, act)
              - L1SNRDBLoss(name="t", use_regularization=False)(silent, act)).item()
    loose = (L1SNRDBLoss(name="t", use_regularization=True, lmin=-30.0)(silent, act)
             - L1SNRDBLoss(name="t", use_regularization=False)(silent, act)).item()
    assert strict > 5.0 * loose, (
        f"at lmin=-60 a collapsed quiet target should be penalized far harder than at -30, got "
        f"{strict:.4f} vs {loose:.4f}")


def test_non_finite_check_covers_actuals_and_inf():
    """Both halves of check_finite were untested: every NaN test put the NaN in `estimates` only.

    Dropping the sanitization of `actuals` reintroduces M7 exactly for the target tensor -- an all-NaN
    target yields a finite-looking loss of -0.0 -- and dropping `isinf` lets an Inf through to a NaN loss.
    """
    act = audio(2, 4096)
    good = act + audio(2, 4096, level=0.005, seed=1)
    for label, e, a in [
        ("NaN in actuals", good, torch.full_like(act, float("nan"))),
        ("Inf in estimates", torch.full_like(good, float("inf")), act),
        ("Inf in actuals", good, torch.full_like(act, float("inf"))),
    ]:
        with pytest.warns(RuntimeWarning, match="non-finite"):
            loss = STFTL1SNRDBLoss(name="t")(e, a)
        assert torch.isfinite(loss), f"{label}: loss is {loss.item()}, expected a finite sanitized value"


@pytest.mark.parametrize("w", [0.99, 0.999])
def test_pure_l1_mode_is_exactly_at_one(w):
    """Loosening the `l1_weight == 1.0` test for pure-L1 mode to `>= 0.99` survived, and at l1_weight=0.99
    it changes the spectrogram loss from 4.39 to 0.0056 -- a different objective entirely."""
    est, act = audio(2, 8192), audio(2, 8192, seed=1)
    for cls in (L1SNRDBLoss, STFTL1SNRDBLoss):
        assert cls(name="t", l1_weight=w).l1_loss is None, (
            f"{cls.__name__} entered pure-L1 mode at l1_weight={w}; that mode is only for exactly 1.0")
        got = cls(name="t", l1_weight=w)(est, act)
        assert torch.allclose(got, expected(cls, est, act, l1_weight=w), atol=1e-4)


def test_spectrogram_regularizer_is_scaled_by_one_minus_l1_weight():
    """No test combined l1_weight > 0 with use_regularization=True on the STFT path, so dropping the
    (1 - l1_weight) factor on the spectral regularizer survived."""
    est, act = audio(2, 8192), audio(2, 8192, seed=1)
    recon = STFTL1SNRDBLoss(name="t", l1_weight=0.5, use_regularization=False)(est, act)
    total = STFTL1SNRDBLoss(name="t", l1_weight=0.5, use_regularization=True)(est, act)
    at_zero_recon = STFTL1SNRDBLoss(name="t", l1_weight=0.0, use_regularization=False)(est, act)
    at_zero_total = STFTL1SNRDBLoss(name="t", l1_weight=0.0, use_regularization=True)(est, act)
    reg_at_half = (total - recon).item()
    reg_at_zero = (at_zero_total - at_zero_recon).item()
    assert abs(reg_at_half / reg_at_zero - 0.5) < 0.02, (
        f"the spectral regularizer should be scaled by (1 - l1_weight): at l1_weight=0.5 it contributes "
        f"{reg_at_half:.6f} against {reg_at_zero:.6f} at 0.0, a ratio of "
        f"{reg_at_half / reg_at_zero:.3f} rather than 0.5")


def test_spectrogram_regularizer_averages_over_usable_resolutions_only():
    """Dividing the accumulated spectral regularizer by len(n_ffts) instead of valid_transforms survived,
    because no test combined a partial resolution set with the spectrogram regularizer."""
    est, act = audio(2, 600), audio(2, 600, seed=1)          # only 2 of 3 resolutions are usable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recon = STFTL1SNRDBLoss(name="t", min_audio_length=1, use_regularization=False)(est, act)
        total = STFTL1SNRDBLoss(name="t", min_audio_length=1, use_regularization=True)(est, act)
    reg = (total - recon).item()
    per_res = []
    for n, h in ((512, 128), (1024, 256)):
        Sa = reference.normalized_stft(act.reshape(-1, act.shape[-1]), n, h, n)
        Se = reference.normalized_stft(est.reshape(-1, est.shape[-1]), n, h, n)
        B = act.shape[0]
        La = reference.dbrms(Sa.abs().reshape(B, -1))
        Lp = reference.dbrms(Se.abs().reshape(B, -1))
        R = (Lp - La).abs()
        per_res.append((reference.adaptive_lambda(Lp, La) * R).mean().item())
    expected_reg = 0.1 * sum(per_res) / len(per_res)          # divided by 2 usable, not 3 configured
    assert abs(reg - expected_reg) / expected_reg < 0.02, (
        f"spectral regularizer is {reg:.6f}; averaging over the 2 usable resolutions gives "
        f"{expected_reg:.6f}, over all 3 configured it would give {expected_reg * 2 / 3:.6f}")


@pytest.mark.parametrize("latch,make,call", [
    ("_dropped_warned",
     lambda: STFTL1SNRDBLoss(name="t", min_audio_length=1),
     lambda f: f(audio(2, 600), audio(2, 600, seed=1))),
    ("_fallback_warned",
     lambda: STFTL1SNRDBLoss(name="t", min_audio_length=512),
     lambda f: f(audio(2, 400), audio(2, 400, seed=1))),
])
def test_one_shot_warnings_really_fire_only_once(latch, make, call):
    """Every warning docstring promises "Warned once per loss instance", but only the NaN latch was tested.
    Making the others fire every call survived the suite, which would flood a training log."""
    loss_fn = make()
    with pytest.warns(RuntimeWarning):
        call(loss_fn)
    assert getattr(loss_fn, latch) is True, f"{latch} was not set"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        call(loss_fn)                     # a second call must be silent


@pytest.mark.parametrize("cls", [STFTL1SNRDBLoss, MultiL1SNRDBLoss], ids=lambda c: c.__name__)
def test_moving_the_loss_module_does_not_break_a_cpu_forward(cls):
    """A device guard cached in a plain attribute goes stale under nn.Module.to().

    T4-5 skipped the transform device move when a cached `_transforms_device` matched. `.to()` moves the
    window buffers but cannot update a Python attribute, so after `loss.to(device)` the cache still said
    "cpu" while the buffers had moved: the guard skipped a needed move, every resolution failed on a device
    mismatch, and the spectral loss became exactly 0.0 with a zero gradient. On Apple silicon that is the
    default path, since mps_cpu_fallback puts the input on CPU while the buffers are on MPS.

    Exercised with meta and CPU here so it runs everywhere. The essential property is that a forward must
    never depend on a cached belief about where the buffers are.
    """
    act = audio(2, 2, 8192)
    est = (act + audio(2, 2, 8192, level=0.005, seed=1)).requires_grad_(True)
    expected_value = cls(name="t")(est.detach(), act)

    loss_fn = cls(name="t")
    # simulate a module move: relocate the buffers without going through forward
    for tf in (loss_fn.spectrogram_transforms if cls is STFTL1SNRDBLoss
               else loss_fn.spec_loss.spectrogram_transforms):
        tf.window.data = tf.window.data.clone()
    loss_fn.to(torch.device("cpu"))

    got = loss_fn(est, act)
    assert got.item() != 0.0, (
        "the loss is exactly zero after a module move, which means the transforms were left on the wrong "
        "device and every resolution failed")
    assert torch.allclose(got, expected_value, atol=1e-5)
    got.backward()
    assert est.grad.abs().max().item() > 0.0, "a module move must not zero the gradient"


@pytest.mark.no_forward  # inspects instance attributes; never calls forward()
def test_no_cached_device_state_on_the_loss():
    """Structural guard on the mechanism, not just the symptom.

    The symptom test above cannot reproduce the accelerator case on a CPU-only machine, so this asserts the
    cause is absent: no attribute caching a device, because such a cache cannot be kept in step with
    nn.Module.to().
    """
    for cls in (STFTL1SNRDBLoss, MultiL1SNRDBLoss):
        loss_fn = cls(name="t")
        cached = [a for a in vars(loss_fn) if "device" in a.lower()]
        assert not cached, (
            f"{cls.__name__} caches device state in {cached}; nn.Module.to() cannot update a plain "
            "attribute, so such a cache goes stale and the forward acts on a stale belief")


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
def test_a_rank_1_waveform_raises_instead_of_returning_an_optimistic_number(cls):
    """`reshape(shape[0], -1)` read a [T] waveform as T batch rows of one sample each.

    The time-domain classes returned a mean of per-sample D1s: measured 0.16 to 1.9 dB optimistic across
    relative errors from 0.5 to 0.001, always flattering and always in a believable dB range, with no warning.
    The spectrogram classes raised a reshape error, so the four classes disagreed on the same input.
    """
    sig = torch.randn(16000) * 0.05
    est = sig + 0.0005 * torch.randn(16000)
    with pytest.raises(ValueError, match="batch-first"):
        cls(name="t")(est, sig)
    # the same data with a batch axis must work, and must not equal the old flattened answer
    batched = cls(name="t")(est[None], sig[None])
    assert torch.isfinite(batched)


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64, torch.uint8, torch.complex64])
@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
def test_non_floating_input_raises_instead_of_a_silent_zero(cls, dtype):
    """int16 PCM straight from a decoder made `STFTL1SNRDBLoss` return -0.0 forever.

    `torch.stft` raises on an integer tensor, the per-resolution handler swallowed it, all resolutions failed,
    and the spectral loss became a permanent 0.0 after a single warning that Python's default filter shows
    once. The time-domain classes raised on the same input, so this was also a four-way inconsistency.
    """
    if dtype.is_complex:
        t = torch.randn(2, 1, 4096, dtype=dtype)
    else:
        t = torch.randint(0, 99, (2, 1, 4096), dtype=dtype)
    with pytest.raises(ValueError, match="floating-point"):
        cls(name="t")(t, t.clone())


@pytest.mark.no_forward  # constructor validation; never reaches a forward
@pytest.mark.parametrize("param", ["weight", "lambda0", "delta_lambda", "spec_reg_coef"])
def test_a_negative_coefficient_raises(param):
    """`spec_weight` is confined to [0, 1] because a negative coefficient maximizes what it scales.

    Nothing else was checked. `weight=-1.0` negated the entire objective and `spec_reg_coef=-5.0` turned the
    anti-collapse regularizer into a reward for collapsing, both silently. Zero stays legal: it is how a term
    is switched off.
    """
    with pytest.raises(ValueError, match="non-negative"):
        MultiL1SNRDBLoss(name="m", **{param: -1.0})
    MultiL1SNRDBLoss(name="m", **{param: 0.0})  # zero must remain accepted


@pytest.mark.no_forward  # constructor validation; never reaches a forward
@pytest.mark.parametrize("param", ["l1snr_eps", "dbrms_eps"])
def test_a_non_positive_epsilon_raises(param):
    """An epsilon of zero or below defeats the stability it exists for; neither was checked."""
    for bad in (0.0, -1e-3):
        with pytest.raises(ValueError, match="positive"):
            MultiL1SNRDBLoss(name="m", **{param: bad})


@pytest.mark.no_forward  # constructor validation; never reaches a forward
@pytest.mark.parametrize("kwargs,match", [
    ({"hop_lengths": [0, 512, 256]}, "positive"),
    ({"hop_lengths": [-128, 512, 256]}, "positive"),
    ({"n_ffts": 512}, "list or tuple"),
    ({"n_ffts": [512.0], "hop_lengths": [128], "win_lengths": [512]}, "must be an int"),
    ({"n_ffts": [], "hop_lengths": [], "win_lengths": []}, "empty"),
])
def test_malformed_stft_params_raise_at_construction(kwargs, match):
    """These failed late, confusingly, or not at all.

    `hop_lengths=[0]` constructed fine and then raised ZeroDivisionError from `_usable_resolutions` on the
    first forward, mid-training, with no mention of hop_length. A bare `n_ffts=512` gave "object of type 'int'
    has no len()". A float from a YAML config died inside `hann_window`. A negative hop silently dropped its
    resolution while the warning claimed the input was too short. Three empty lists were accepted outright,
    making the spectrogram branch a permanent time-domain fallback reporting that same wrong reason.
    """
    with pytest.raises(ValueError, match=match):
        STFTL1SNRDBLoss(name="s", **kwargs)


def test_a_single_resolution_in_a_list_is_still_valid():
    """The bare-int rejection must not catch the legitimate one-resolution case."""
    est, act = audio(2, 8192), audio(2, 8192, seed=1)
    loss = STFTL1SNRDBLoss(name="s", n_ffts=[1024], hop_lengths=[256], win_lengths=[1024])
    out = loss(est, act)
    want = reference.multi_res_spec_d1(est, act, n_ffts=[1024], hop_lengths=[256], win_lengths=[1024])
    assert torch.allclose(out, want, atol=1e-4), (
        f"a single-resolution configuration returns {out.item()} against the reference's {want.item()}")


def test_the_spectrogram_loss_is_not_monotone_and_the_docs_say_so():
    """Pins the inversion itself, so the documented claim cannot quietly stop being true.

    A DC offset produces almost no imaginary error, so `D1_im` saturates at the eps floor and pays a fixed
    reward any near-purely-real error collects. The result is an estimate the time domain rates ~10 dB worse
    scoring ~5 dB better on the spectrogram loss. This is a property of the published Re+Im objective --
    confirmed against tests/reference.py -- so the gate exists to keep the documentation honest, not to
    forbid the behaviour. If a future change makes the loss monotone here, this test should fail and the
    README's Limitations entry should come out with it.
    """
    torch.manual_seed(7)
    act = torch.randn(4, 2, 44100) * 0.05
    dc = act + torch.full_like(act, 0.05)          # error equal to the signal amplitude
    noise = act + 0.005 * torch.randn_like(act)    # 10% relative error

    time_dc = L1SNRLoss("t")(dc, act).item()
    time_noise = L1SNRLoss("t")(noise, act).item()
    spec_dc = STFTL1SNRDBLoss("s")(dc, act).item()
    spec_noise = STFTL1SNRDBLoss("s")(noise, act).item()

    assert time_dc > time_noise, "the time domain should rate the DC error worse; the fixture has drifted"
    assert spec_dc < spec_noise, (
        f"the spectrogram loss no longer inverts the ordering (DC {spec_dc:.2f} vs noise {spec_noise:.2f}); "
        "if this is deliberate, remove the Limitations entry that documents it")
    # and the combined default must still order them correctly, which is why it is the recommended entry point
    multi_dc = MultiL1SNRDBLoss("m")(dc, act).item()
    multi_noise = MultiL1SNRDBLoss("m")(noise, act).item()
    assert multi_dc > multi_noise, (
        "MultiL1SNRDBLoss at spec_weight=0.5 no longer orders these correctly, which the README relies on")


def test_the_regularizer_has_no_gradient_at_exact_silence_but_the_loss_does():
    """Two halves, because only stating the first would read as "the model gets stuck", which it does not.

    `d/dx sqrt(mean(x^2) + eps)` is exactly 0 at `x == 0`, so the anti-collapse term exerts no force at the
    collapse point itself. The D1 term does, so the total gradient is nonzero and a saturated mask can still
    escape.
    """
    x = torch.zeros(2, 16000, requires_grad=True)
    torch.sqrt(torch.mean(x ** 2, dim=-1) + 1e-8).sum().backward()
    assert x.grad.abs().max().item() == 0.0, (
        "the RMS term's gradient at exact zero is no longer zero; the documented caveat is stale")

    est = torch.zeros(2, 16000, requires_grad=True)
    act = audio(2, 16000)
    out = MultiL1SNRDBLoss("m", use_time_regularization=True)(est, act)
    out.backward()
    assert est.grad.abs().max().item() > 0.0, (
        "the total gradient at exact silence is zero, so a collapsed model really would be stuck; the "
        "README says the D1 term supplies escape pressure")


@pytest.mark.no_forward  # exercises dbrms directly, not a loss forward
def test_dbrms_overflows_float32_on_finite_input_and_float64_does_not():
    """`mean(x**2)` squares before reducing, so the sum overflows around 2.5e16 at a realistic shape.

    The inputs are finite, so `check_finite` cannot catch it. Left as documented behaviour rather than
    re-tuned, because it needs a run already diverging through 1e16 and inf/NaN is loud. Gated so the README's
    figure stays attached to something measured, and so a future numerics change has to notice this claim.
    """
    # 529200 elements per row: a [8, 2, 264600] batch flattened as dbrms flattens it. The threshold scales
    # with elements per row (2.0e17 at 8192, 2.5e16 here), so the shape is load-bearing.
    x32 = torch.full((8, 2 * 264600), 3e16, dtype=torch.float32)
    assert not torch.isfinite(dbrms(x32)).all(), (
        "dbrms no longer overflows at 3e16 in float32; if the numerics were made overflow-safe, update the "
        "README's Limitations entry")
    assert torch.isfinite(x32).all(), "the premise is that the input itself is finite"
    x64 = torch.full((8, 2 * 264600), 3e16, dtype=torch.float64)
    assert torch.isfinite(dbrms(x64)).all(), "float64 is documented as unaffected"


def test_the_fallback_warning_names_the_constraint_that_actually_bit():
    """The fallback branch has two causes and used to report only one of them.

    1024 samples with `n_ffts=[2048, 4096]` and `min_audio_length=512` warned "input length 1024 is below
    min_audio_length (512)" -- a threshold the input clears, naming a knob that would not help. The real
    constraint is the per-resolution `n_fft // 2 + 1` requirement. Two high-resolution FFTs is an ordinary
    choice, and the neighbouring case emits a correct dropped-resolutions message, which made the wrong one
    look authoritative. Same defect class as the all-resolutions-failed warning fixed earlier in this branch.
    """
    loss = STFTL1SNRDBLoss("spec", n_ffts=[2048, 4096], hop_lengths=[512, 1024],
                           win_lengths=[2048, 4096], min_audio_length=512)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss(audio(2, 1024), audio(2, 1024, seed=1))
    msg = str(caught[0].message)
    assert "is below min_audio_length" not in msg, (
        f"the warning claims 1024 is below min_audio_length=512, which it is not: {msg}")
    assert "1025" in msg, "the warning should state the length the configured resolutions actually need"

    # the genuinely-too-short case must still name min_audio_length
    short = STFTL1SNRDBLoss("spec", min_audio_length=512)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        short(audio(2, 400), audio(2, 400, seed=1))
    assert "below min_audio_length" in str(caught[0].message), (
        "an input that really is below the threshold must still say so")


def test_check_finite_substitutes_full_scale_for_inf_not_zero():
    """Two docstrings, the runtime warning and the CHANGELOG all said "replaced with zeros".

    Only NaN becomes zero. `torch.nan_to_num(posinf=1.0, neginf=-1.0)` maps infinities to full-scale audio, so
    a corrupt Inf sample becomes the loudest possible click rather than silence -- a materially different
    thing to see when a level-matching regularizer starts moving. The substitution values had no test at all,
    so the docs could drift from them freely.
    """
    est = torch.zeros(1, 4096)
    est[0, 0] = float("inf")
    est[0, 1] = float("-inf")
    est[0, 2] = float("nan")
    act = torch.zeros(1, 4096)
    loss = STFTL1SNRDBLoss("spec", check_finite=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = loss(est, act)
    # what the loss would be if Inf really were zeroed: an all-silent estimate against a silent target
    zeroed = loss(torch.zeros(1, 4096), act)
    assert got.item() != zeroed.item(), (
        "sanitizing Inf to zero and to full scale gave the same loss, so this test cannot see the difference")
    sanitized = torch.nan_to_num(est, nan=0.0, posinf=1.0, neginf=-1.0)
    assert sanitized[0, 0].item() == 1.0 and sanitized[0, 1].item() == -1.0, (
        "the documented substitution no longer matches torch.nan_to_num's arguments in the source")
    assert torch.allclose(loss(sanitized, act), got), (
        "the loss does not match one computed on the substitution the docs now describe")


@pytest.mark.parametrize("corrupt", ["estimates", "actuals"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_check_finite_false_propagates_non_finite_from_either_input(corrupt, bad):
    """With check_finite=False a non-finite input must reach the loss, whichever tensor it is in.

    It did not. Every resolution went NaN, the isnan guard dropped them all, and the all-failed fallback was
    `est_source.sum() * 0.0` -- an expression that carries non-finiteness only from the estimate. A NaN in the
    *target* therefore produced exactly -0.0 with an all-zero gradient: a corrupt target reading as a
    perfectly healthy step, which is the same silent-zero failure as the cached-device bug and the detached
    zero before it. Both the README and the CHANGELOG recommend check_finite=False on CUDA, and a non-finite
    target is the ordinary corrupt-decode case, so this is the combination a user actually reaches.
    """
    torch.manual_seed(3)
    est = (torch.randn(2, 1, 8192) * 0.05).requires_grad_(True)
    act = est.detach() + 0.005 * torch.randn(2, 1, 8192)
    if corrupt == "actuals":
        act[0, 0, 100] = bad
    else:
        est.data[0, 0, 100] = bad
    loss = STFTL1SNRDBLoss("spec", check_finite=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = loss(est, act)
    assert not torch.isfinite(out), (
        f"a {bad} in {corrupt} produced the finite loss {out.item()!r} with check_finite=False, which "
        "documents itself as letting non-finite values propagate visibly")
    # Anchor the clean path to a reference number. Without this the `corrupt="estimates"` variants pass
    # against a stubbed forward, because a stub that touches `estimates` propagates NaN on its own -- so they
    # would be asserting a property of NaN arithmetic rather than of this library.
    clean_est = (torch.randn(2, 1, 8192) * 0.05)
    clean_act = clean_est + 0.005 * torch.randn(2, 1, 8192)
    clean = loss(clean_est, clean_act)
    assert torch.allclose(clean, expected(STFTL1SNRDBLoss, clean_est, clean_act), atol=1e-4), (
        "finite input no longer matches the independent reference")


def test_the_all_failed_fallback_is_still_a_graph_connected_zero():
    """The fix for the above must not cost the property the fallback exists for.

    When every resolution legitimately fails on finite input the return has to be zero, attached to the
    graph, and differentiable, or .backward() raises and the term silently contributes nothing inside
    MultiL1SNRDBLoss while the time term goes on training.
    """
    est = (torch.randn(2, 1, 4096) * 0.05).requires_grad_(True)
    act = est.detach() + 0.001
    loss = STFTL1SNRDBLoss("spec")

    class _AlwaysFails(torch.nn.Module):
        def forward(self, _x):
            raise RuntimeError("forced failure")

    # Reaching this branch with finite input on CPU takes a forced failure: _usable_resolutions admits only
    # resolutions torch.stft actually accepts, so every natural route to it involves a non-finite value or a
    # device mismatch, and MPS is not available everywhere this suite runs.
    loss.spectrogram_transforms = torch.nn.ModuleList([_AlwaysFails() for _ in loss.n_ffts])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = loss(est, act)
    out.backward()
    assert out.item() == 0.0, f"the fallback must be zero on finite input, got {out.item()!r}"
    assert out.grad_fn is not None, "the fallback must stay attached to the graph"
    assert est.grad is not None and torch.isfinite(est.grad).all(), (
        "the fallback must be differentiable, or .backward() raises and the spectral term silently "
        "contributes nothing inside MultiL1SNRDBLoss")


@pytest.mark.no_forward  # inspects signatures only
def test_new_constructor_params_are_appended_not_inserted():
    """A parameter inserted mid-signature silently reinterprets every positional call.

    `spec_reg_coef` was added at position 17 of MultiL1SNRDBLoss, before `time_loss_params`. A 17-argument
    positional call written against 0.1.3 then *constructed and ran without error* while the user's
    `time_loss_params` dict landed in `spec_reg_coef` and their overrides were silently discarded. This pins
    the 0.1.3 parameter order as a prefix, so additions have to go at the end.
    """
    import inspect as _inspect
    v013 = V013_PARAM_ORDER
    for cls in ALL_CLASSES:
        current = [p for p in _inspect.signature(cls.__init__).parameters if p != "self"]
        expected = v013[cls.__name__]
        assert current[:len(expected)] == expected, (
            f"{cls.__name__}'s 0.1.3 parameter order is no longer a prefix of its signature. A positional "
            f"call written against 0.1.3 would now mean something different.\n"
            f"  0.1.3:   {expected}\n  current: {current[:len(expected)]}")


# The published 0.1.3 constructor order, in full. TRANSCRIBED LISTS ARE THE HAZARD THESE TESTS EXIST FOR: an
# earlier version omitted `spec_reg_coef` from MultiL1SNRDBLoss, which made a real break look compliant and
# let a change that moved the parameter pass its own gate. So the lists are themselves checked against the
# 0.1.3 source in test_the_0_1_3_reference_order_is_itself_correct.
V013_PARAM_ORDER = {
        "MultiL1SNRDBLoss": ["name", "weight", "spec_weight", "l1_weight", "use_time_regularization",
                             "use_spec_regularization", "lambda0", "delta_lambda", "l1snr_eps",
                             "dbrms_eps", "lmin", "n_ffts", "hop_lengths", "win_lengths", "window_fn",
                             "min_audio_length", "spec_reg_coef", "time_loss_params", "spec_loss_params",
                             "mps_cpu_fallback"],
        "L1SNRLoss": ["name", "weight", "eps", "l1_weight"],
        "L1SNRDBLoss": ["name", "weight", "lambda0", "delta_lambda", "l1snr_eps", "dbrms_eps", "lmin",
                        "use_regularization", "l1_weight"],
        "STFTL1SNRDBLoss": ["name", "weight", "lambda0", "delta_lambda", "l1snr_eps", "dbrms_eps", "lmin",
                            "n_ffts", "hop_lengths", "win_lengths", "window_fn", "min_audio_length",
                            "use_regularization", "spec_reg_coef", "l1_weight", "mps_cpu_fallback"],
}

# Last commit at version 0.1.3, the only version published to PyPI when 0.2.0 was prepared.
_V013_COMMIT = "15b2458"


def _param_order_at_0_1_3():
    """Read the four constructor signatures out of the 0.1.3 source, or return None if git cannot reach it.

    Returns None rather than a partial answer: a check that silently compares against the wrong commit is
    worse than no check, and this whole family of tests exists because a wrong reference passed once.
    """
    import ast
    import subprocess
    def _show(path):
        return subprocess.run(["git", "show", f"{_V013_COMMIT}:{path}"], capture_output=True, text=True,
                              cwd=str(pathlib.Path(__file__).resolve().parent.parent))
    version = _show("torch_l1_snr/__init__.py")
    if version.returncode != 0 or '__version__ = "0.1.3"' not in version.stdout:
        return None  # not a git checkout, shallow clone, or not the commit we think it is
    src = _show("torch_l1_snr/l1snr.py")
    if src.returncode != 0:
        return None
    orders = {}
    for node in ast.walk(ast.parse(src.stdout)):
        if isinstance(node, ast.ClassDef) and node.name in V013_PARAM_ORDER:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    orders[node.name] = [a.arg for a in item.args.args][1:]
    return orders or None


@pytest.mark.no_forward  # reads signatures out of git history
def test_the_0_1_3_reference_order_is_itself_correct():
    """The reference list above is hand-written, and a hand-written reference can be wrong.

    It was: `spec_reg_coef` was omitted from MultiL1SNRDBLoss, so the prefix check above compared the
    signature against a 19-parameter fiction and passed while position 17 genuinely moved. Nothing in the
    suite could see it, because the fiction was the only stated ground truth. This gate makes the 0.1.3
    source the authority.
    """
    actual = _param_order_at_0_1_3()
    if actual is None:
        pytest.skip(f"cannot read {_V013_COMMIT} from git; the reference list is unverifiable here")
    for name, expected in V013_PARAM_ORDER.items():
        assert actual[name] == expected, (
            f"the transcribed 0.1.3 order for {name} does not match commit {_V013_COMMIT}.\n"
            f"  transcribed: {expected}\n  actual 0.1.3: {actual[name]}")


@pytest.mark.no_forward  # constructs only; the override lands in a child module, not in a loss value
def test_a_full_positional_call_written_against_0_1_3_still_means_the_same_thing():
    """The behavioural form of the prefix check, and the one that actually caught the break.

    A user who passed every 0.1.3 parameter positionally -- 20 of them on MultiL1SNRDBLoss, with
    `time_loss_params` at position 18 -- must still get their override applied. When `spec_reg_coef` moved
    from 17 to the end, this call raised `TypeError: 'float' object is not iterable`, because position 18
    stopped being a dict. A quieter version of the same bug discards the override without any error.
    """
    order = V013_PARAM_ORDER["MultiL1SNRDBLoss"]
    defaults = {
        "name": "vocals", "weight": 1.0, "spec_weight": 0.5, "l1_weight": 0.0,
        "use_time_regularization": True, "use_spec_regularization": False, "lambda0": 1.0,
        "delta_lambda": 1.0, "l1snr_eps": 1e-3, "dbrms_eps": 1e-8, "lmin": -60.0,
        "n_ffts": [4096, 2048, 1024], "hop_lengths": [1024, 512, 256],
        "win_lengths": [4096, 2048, 1024], "window_fn": "hann", "min_audio_length": 512,
        "spec_reg_coef": 0.1, "time_loss_params": {"lmin": -40.0}, "spec_loss_params": None,
        "mps_cpu_fallback": True,
    }
    assert set(defaults) == set(order), "the positional fixture has drifted from the 0.1.3 parameter list"
    loss = MultiL1SNRDBLoss(*[defaults[p] for p in order])
    assert loss.time_loss.lmin == -40.0, (
        "a 20-argument positional call written against 0.1.3 no longer applies the caller's "
        f"time_loss_params; lmin is {loss.time_loss.lmin}, not the -40.0 passed at position "
        f"{order.index('time_loss_params') + 1}")


def test_all_resolutions_failed_warning_is_one_shot():
    """The CHANGELOG lists this among the one-shot warnings, but it had no latch and fired every call,
    deduped only by Python's default filter. It is also the warning a device-mismatched run emits, so in
    the case where it matters most it would have repeated every step."""
    loss_fn = _all_failing_loss()
    est, act = audio(2, 8192), audio(2, 8192, seed=1)
    with pytest.warns(RuntimeWarning, match="every spectrogram resolution failed"):
        first = loss_fn(est, act)
    assert first.item() == 0.0
    # Scoped to the warning under test. The per-resolution "Error computing spectrogram" UserWarning is a
    # separate backstop that reports a genuine exception on every call and is deliberately not latched --
    # the CHANGELOG lists exactly three one-shot warnings and that is not one of them.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        second = loss_fn(est, act)
    repeats = [w for w in rec if "every spectrogram resolution failed" in str(w.message)]
    assert not repeats, f"the all-resolutions-failed warning fired again on call 2: {repeats}"
    assert second.item() == 0.0, "latching the warning must not change the returned value"
