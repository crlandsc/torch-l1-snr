import torch
import pytest
from typing import Optional
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
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

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
    assert loss.ndim == 0 and not torch.isnan(loss) and not torch.isinf(loss)
    
    # Test pure L1 mode
    loss_fn_pure_l1 = STFTL1SNRDBLoss(name="test_pure_l1", l1_weight=1.0)
    l1_loss = loss_fn_pure_l1(estimates, actuals)
    assert l1_loss.ndim == 0 and not torch.isnan(l1_loss) and not torch.isinf(l1_loss)

    # Test with very short audio
    short_estimates = estimates[:, :500]
    short_actuals = actuals[:, :500]
    loss_short = loss_fn(short_estimates, short_actuals)
    # min_audio_length is 512, so this should fallback to time-domain loss
    assert loss_short.ndim == 0 and not torch.isnan(loss_short)

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
    assert loss.ndim == 0 and not torch.isnan(loss)

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
    assert loss_all.ndim == 0 and not torch.isnan(loss_all)
    
    # Test pure L1 mode on all stems
    loss_fn_l1 = MultiL1SNRDBLoss(name="l1_only", l1_weight=1.0)
    l1_loss = loss_fn_l1(est_all, act_all)
    
    # Can't easily compute multi-res STFT L1 here, but can check it's not nan
    assert l1_loss.ndim == 0 and not torch.isnan(l1_loss)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
def test_loss_variants(dummy_audio, l1_weight):
    """Test L1SNRDBLoss and STFTL1SNRDBLoss with different l1_weights."""
    estimates, actuals = dummy_audio
    
    time_loss_fn = L1SNRDBLoss(name=f"test_time_{l1_weight}", l1_weight=l1_weight)
    time_loss = time_loss_fn(estimates, actuals)
    assert not torch.isnan(time_loss) and not torch.isinf(time_loss)

    spec_loss_fn = STFTL1SNRDBLoss(name=f"test_spec_{l1_weight}", l1_weight=l1_weight)
    spec_loss = spec_loss_fn(estimates, actuals)
    assert not torch.isnan(spec_loss) and not torch.isinf(spec_loss)

# --- Wrapper-Paradigm Tests ---

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
def test_l1snr_wrapper_all_stems_3d(dummy_stems_3d, l1_weight):
    """Test L1SNRLoss wrapper with stem_dimension=None on [B,S,T]."""
    estimates, actuals = dummy_stems_3d
    base_loss = L1SNRLoss(name="test", weight=1.0, l1_weight=l1_weight)
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=None)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    direct_result = base_loss(estimates, actuals)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
def test_l1snr_wrapper_all_stems_4d(dummy_stems_4d, l1_weight):
    """Test L1SNRLoss wrapper with stem_dimension=None on [B,S,C,T]."""
    estimates, actuals = dummy_stems_4d
    base_loss = L1SNRLoss(name="test", weight=1.0, l1_weight=l1_weight)
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=None)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    direct_result = base_loss(estimates, actuals)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("stem_idx", [0, 3])
def test_l1snr_wrapper_single_stem_3d(dummy_stems_3d, l1_weight, stem_idx):
    """Test L1SNRLoss wrapper with stem_dimension=k on [B,S,T]."""
    estimates, actuals = dummy_stems_3d
    base_loss = L1SNRLoss(name="test", weight=1.0, l1_weight=l1_weight)
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=stem_idx)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    est_slice = estimates[:, stem_idx, :]
    act_slice = actuals[:, stem_idx, :]
    direct_result = base_loss(est_slice, act_slice)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("stem_idx", [0, 3])
def test_l1snr_wrapper_single_stem_4d(dummy_stems_4d, l1_weight, stem_idx):
    """Test L1SNRLoss wrapper with stem_dimension=k on [B,S,C,T]."""
    estimates, actuals = dummy_stems_4d
    base_loss = L1SNRLoss(name="test", weight=1.0, l1_weight=l1_weight)
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=stem_idx)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    est_slice = estimates[:, stem_idx, :, :]
    act_slice = actuals[:, stem_idx, :, :]
    direct_result = base_loss(est_slice, act_slice)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("use_reg", [True, False])
def test_l1snrdb_wrapper_all_stems_3d(dummy_stems_3d, l1_weight, use_reg):
    """Test L1SNRDBLoss wrapper with stem_dimension=None on [B,S,T]."""
    estimates, actuals = dummy_stems_3d
    base_loss = L1SNRDBLoss(name="test", weight=1.0, l1_weight=l1_weight, use_regularization=use_reg)
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=None)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    direct_result = base_loss(estimates, actuals)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("use_reg", [True, False])
def test_l1snrdb_wrapper_all_stems_4d(dummy_stems_4d, l1_weight, use_reg):
    """Test L1SNRDBLoss wrapper with stem_dimension=None on [B,S,C,T]."""
    estimates, actuals = dummy_stems_4d
    base_loss = L1SNRDBLoss(name="test", weight=1.0, l1_weight=l1_weight, use_regularization=use_reg)
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=None)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    direct_result = base_loss(estimates, actuals)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("use_reg", [True, False])
@pytest.mark.parametrize("stem_idx", [0, 3])
def test_l1snrdb_wrapper_single_stem_3d(dummy_stems_3d, l1_weight, use_reg, stem_idx):
    """Test L1SNRDBLoss wrapper with stem_dimension=k on [B,S,T]."""
    estimates, actuals = dummy_stems_3d
    base_loss = L1SNRDBLoss(name="test", weight=1.0, l1_weight=l1_weight, use_regularization=use_reg)
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=stem_idx)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    est_slice = estimates[:, stem_idx, :]
    act_slice = actuals[:, stem_idx, :]
    direct_result = base_loss(est_slice, act_slice)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("use_reg", [True, False])
@pytest.mark.parametrize("stem_idx", [0, 3])
def test_l1snrdb_wrapper_single_stem_4d(dummy_stems_4d, l1_weight, use_reg, stem_idx):
    """Test L1SNRDBLoss wrapper with stem_dimension=k on [B,S,C,T]."""
    estimates, actuals = dummy_stems_4d
    base_loss = L1SNRDBLoss(name="test", weight=1.0, l1_weight=l1_weight, use_regularization=use_reg)
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=stem_idx)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    est_slice = estimates[:, stem_idx, :, :]
    act_slice = actuals[:, stem_idx, :, :]
    direct_result = base_loss(est_slice, act_slice)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
def test_stft_wrapper_all_stems_3d(dummy_stems_3d, l1_weight):
    """Test STFTL1SNRDBLoss wrapper with stem_dimension=None on [B,S,T]."""
    estimates, actuals = dummy_stems_3d
    base_loss = STFTL1SNRDBLoss(
        name="test", weight=1.0, l1_weight=l1_weight,
        n_ffts=[256], hop_lengths=[64], win_lengths=[256], min_audio_length=256
    )
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=None)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    direct_result = base_loss(estimates, actuals)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
def test_stft_wrapper_all_stems_4d(dummy_stems_4d, l1_weight):
    """Test STFTL1SNRDBLoss wrapper with stem_dimension=None on [B,S,C,T]."""
    estimates, actuals = dummy_stems_4d
    base_loss = STFTL1SNRDBLoss(
        name="test", weight=1.0, l1_weight=l1_weight,
        n_ffts=[256], hop_lengths=[64], win_lengths=[256], min_audio_length=256
    )
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=None)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    direct_result = base_loss(estimates, actuals)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("stem_idx", [0, 3])
def test_stft_wrapper_single_stem_3d(dummy_stems_3d, l1_weight, stem_idx):
    """Test STFTL1SNRDBLoss wrapper with stem_dimension=k on [B,S,T]."""
    estimates, actuals = dummy_stems_3d
    base_loss = STFTL1SNRDBLoss(
        name="test", weight=1.0, l1_weight=l1_weight,
        n_ffts=[256], hop_lengths=[64], win_lengths=[256], min_audio_length=256
    )
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=stem_idx)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    est_slice = estimates[:, stem_idx, :]
    act_slice = actuals[:, stem_idx, :]
    direct_result = base_loss(est_slice, act_slice)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("stem_idx", [0, 3])
def test_stft_wrapper_single_stem_4d(dummy_stems_4d, l1_weight, stem_idx):
    """Test STFTL1SNRDBLoss wrapper with stem_dimension=k on [B,S,C,T]."""
    estimates, actuals = dummy_stems_4d
    base_loss = STFTL1SNRDBLoss(
        name="test", weight=1.0, l1_weight=l1_weight,
        n_ffts=[256], hop_lengths=[64], win_lengths=[256], min_audio_length=256
    )
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=stem_idx)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    est_slice = estimates[:, stem_idx, :, :]
    act_slice = actuals[:, stem_idx, :, :]
    direct_result = base_loss(est_slice, act_slice)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

def test_stft_wrapper_short_audio_3d():
    """Test STFTL1SNRDBLoss wrapper fallback path with short audio [B,S,T]."""
    estimates = torch.randn(2, 4, 400)  # Short audio
    actuals = torch.randn(2, 4, 400)
    actuals[:, 0, :100] += 0.1
    
    base_loss = STFTL1SNRDBLoss(
        name="test", weight=1.0, l1_weight=0.0,
        n_ffts=[256], hop_lengths=[64], win_lengths=[256], min_audio_length=512
    )
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=None)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    direct_result = base_loss(estimates, actuals)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

def test_stft_wrapper_short_audio_4d():
    """Test STFTL1SNRDBLoss wrapper fallback path with short audio [B,S,C,T]."""
    estimates = torch.randn(2, 4, 1, 400)  # Short audio
    actuals = torch.randn(2, 4, 1, 400)
    actuals[:, 0, :, :100] += 0.1
    
    base_loss = STFTL1SNRDBLoss(
        name="test", weight=1.0, l1_weight=0.0,
        n_ffts=[256], hop_lengths=[64], win_lengths=[256], min_audio_length=512
    )
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=None)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    direct_result = base_loss(estimates, actuals)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("use_time_reg", [True, False])
def test_multi_wrapper_all_stems_3d(dummy_stems_3d, l1_weight, use_time_reg):
    """Test MultiL1SNRDBLoss wrapper with stem_dimension=None on [B,S,T]."""
    estimates, actuals = dummy_stems_3d
    base_loss = MultiL1SNRDBLoss(
        name="test", weight=1.0, l1_weight=l1_weight,
        use_time_regularization=use_time_reg, use_spec_regularization=False,
        n_ffts=[256], hop_lengths=[64], win_lengths=[256], min_audio_length=256
    )
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=None)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    direct_result = base_loss(estimates, actuals)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("use_time_reg", [True, False])
def test_multi_wrapper_all_stems_4d(dummy_stems_4d, l1_weight, use_time_reg):
    """Test MultiL1SNRDBLoss wrapper with stem_dimension=None on [B,S,C,T]."""
    estimates, actuals = dummy_stems_4d
    base_loss = MultiL1SNRDBLoss(
        name="test", weight=1.0, l1_weight=l1_weight,
        use_time_regularization=use_time_reg, use_spec_regularization=False,
        n_ffts=[256], hop_lengths=[64], win_lengths=[256], min_audio_length=256
    )
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=None)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    direct_result = base_loss(estimates, actuals)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("use_time_reg", [True, False])
@pytest.mark.parametrize("stem_idx", [0, 3])
def test_multi_wrapper_single_stem_3d(dummy_stems_3d, l1_weight, use_time_reg, stem_idx):
    """Test MultiL1SNRDBLoss wrapper with stem_dimension=k on [B,S,T]."""
    estimates, actuals = dummy_stems_3d
    base_loss = MultiL1SNRDBLoss(
        name="test", weight=1.0, l1_weight=l1_weight,
        use_time_regularization=use_time_reg, use_spec_regularization=False,
        n_ffts=[256], hop_lengths=[64], win_lengths=[256], min_audio_length=256
    )
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=stem_idx)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    est_slice = estimates[:, stem_idx, :]
    act_slice = actuals[:, stem_idx, :]
    direct_result = base_loss(est_slice, act_slice)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("use_time_reg", [True, False])
@pytest.mark.parametrize("stem_idx", [0, 3])
def test_multi_wrapper_single_stem_4d(dummy_stems_4d, l1_weight, use_time_reg, stem_idx):
    """Test MultiL1SNRDBLoss wrapper with stem_dimension=k on [B,S,C,T]."""
    estimates, actuals = dummy_stems_4d
    base_loss = MultiL1SNRDBLoss(
        name="test", weight=1.0, l1_weight=l1_weight,
        use_time_regularization=use_time_reg, use_spec_regularization=False,
        n_ffts=[256], hop_lengths=[64], win_lengths=[256], min_audio_length=256
    )
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=stem_idx)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    est_slice = estimates[:, stem_idx, :, :]
    act_slice = actuals[:, stem_idx, :, :]
    direct_result = base_loss(est_slice, act_slice)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

def test_multi_wrapper_short_audio_3d():
    """Test MultiL1SNRDBLoss wrapper fallback path with short audio [B,S,T]."""
    estimates = torch.randn(2, 4, 400)  # Short audio
    actuals = torch.randn(2, 4, 400)
    actuals[:, 0, :100] += 0.1
    
    base_loss = MultiL1SNRDBLoss(
        name="test", weight=1.0, l1_weight=0.0,
        use_time_regularization=True, use_spec_regularization=False,
        n_ffts=[256], hop_lengths=[64], win_lengths=[256], min_audio_length=512
    )
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=None)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    direct_result = base_loss(estimates, actuals)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result)

def test_multi_wrapper_short_audio_4d():
    """Test MultiL1SNRDBLoss wrapper fallback path with short audio [B,S,C,T]."""
    estimates = torch.randn(2, 4, 1, 400)  # Short audio
    actuals = torch.randn(2, 4, 1, 400)
    actuals[:, 0, :, :100] += 0.1
    
    base_loss = MultiL1SNRDBLoss(
        name="test", weight=1.0, l1_weight=0.0,
        use_time_regularization=True, use_spec_regularization=False,
        n_ffts=[256], hop_lengths=[64], win_lengths=[256], min_audio_length=512
    )
    wrapped_loss = StemWrappedLoss(base_loss, stem_dimension=None)
    
    wrapped_result = wrapped_loss(estimates, actuals)
    direct_result = base_loss(estimates, actuals)
    
    assert torch.allclose(wrapped_result, direct_result, atol=1e-6)
    assert wrapped_result.ndim == 0
    assert not torch.isnan(wrapped_result) and not torch.isinf(wrapped_result) 