import torch
import pytest
from torch_l1snr import (
    dbrms,
    L1SNRLoss,
    L1SNRDBLoss,
    STFTL1SNRDBLoss,
    MultiL1SNRDBLoss,
)

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
    loss_fn = L1SNRLoss()
    loss = loss_fn(estimates, actuals)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

def test_l1snrdb_loss_time(dummy_audio):
    estimates, actuals = dummy_audio
    
    # Test with default settings (L1SNR + Regularization)
    loss_fn = L1SNRDBLoss(use_regularization=True, l1_weight=0.0)
    loss = loss_fn(estimates, actuals)
    assert loss.ndim == 0 and not torch.isnan(loss)

    # Test without regularization
    loss_fn_no_reg = L1SNRDBLoss(use_regularization=False, l1_weight=0.0)
    loss_no_reg = loss_fn_no_reg(estimates, actuals)
    assert loss_no_reg.ndim == 0 and not torch.isnan(loss_no_reg)

    # Test with L1 loss component
    loss_fn_l1 = L1SNRDBLoss(l1_weight=0.2)
    loss_l1 = loss_fn_l1(estimates, actuals)
    assert loss_l1.ndim == 0 and not torch.isnan(loss_l1)
    
    # Test pure L1 loss mode
    loss_fn_pure_l1 = L1SNRDBLoss(l1_weight=1.0)
    pure_l1_loss = loss_fn_pure_l1(estimates, actuals)
    l1_loss_manual = torch.mean(torch.abs(estimates - actuals)) * loss_fn_pure_l1.l1_scale
    assert torch.allclose(pure_l1_loss, l1_loss_manual)

def test_stft_l1snrdb_loss(dummy_audio):
    estimates, actuals = dummy_audio
    
    # Test with default settings
    loss_fn = STFTL1SNRDBLoss(l1_weight=0.0)
    loss = loss_fn(estimates, actuals)
    assert loss.ndim == 0 and not torch.isnan(loss) and not torch.isinf(loss)
    
    # Test pure L1 mode
    loss_fn_pure_l1 = STFTL1SNRDBLoss(l1_weight=1.0)
    l1_loss = loss_fn_pure_l1(estimates, actuals)
    assert l1_loss.ndim == 0 and not torch.isnan(l1_loss) and not torch.isinf(l1_loss)

    # Test with very short audio
    short_estimates = estimates[:, :500]
    short_actuals = actuals[:, :500]
    loss_short = loss_fn(short_estimates, short_actuals)
    # min_audio_length is 512, so this should return 0
    assert loss_short.item() == 0.0

def test_stem_multi_loss(dummy_stems):
    estimates, actuals = dummy_stems

    # Test with a specific stem
    loss_fn_stem = MultiL1SNRDBLoss(
        name="test_loss",
        stem_dimension=1,
        spec_weight=0.5,
        l1_weight=0.1
    )
    loss = loss_fn_stem(estimates, actuals)
    assert loss.ndim == 0 and not torch.isnan(loss)

    # Test with all stems jointly
    loss_fn_all = MultiL1SNRDBLoss(
        name="test_loss_all",
        stem_dimension=None,
        spec_weight=0.5,
        l1_weight=0.1
    )
    loss_all = loss_fn_all(estimates, actuals)
    assert loss_all.ndim == 0 and not torch.isnan(loss_all)
    
    # Test pure L1 mode on all stems
    loss_fn_l1 = MultiL1SNRDBLoss(name="l1_only", l1_weight=1.0)
    l1_loss = loss_fn_l1(estimates, actuals)
    
    # Manual L1 calculation for comparison
    time_loss = torch.mean(torch.abs(estimates - actuals)) * 100.0 # default l1_scale_time
    
    # Can't easily compute multi-res STFT L1 here, but can check it's not nan
    assert l1_loss.ndim == 0 and not torch.isnan(l1_loss)

@pytest.mark.parametrize("l1_weight", [0.0, 0.5, 1.0])
def test_loss_variants(dummy_audio, l1_weight):
    """Test L1SNRDBLoss and STFTL1SNRDBLoss with different l1_weights."""
    estimates, actuals = dummy_audio
    
    time_loss_fn = L1SNRDBLoss(l1_weight=l1_weight)
    time_loss = time_loss_fn(estimates, actuals)
    assert not torch.isnan(time_loss) and not torch.isinf(time_loss)

    spec_loss_fn = STFTL1SNRDBLoss(l1_weight=l1_weight)
    spec_loss = spec_loss_fn(estimates, actuals)
    assert not torch.isnan(spec_loss) and not torch.isinf(spec_loss) 