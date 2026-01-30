# PyTorch implementation of L1SNR loss functions for audio source separation
# https://github.com/crlandsc/torch-l1-snr
# Copyright (c) 2026 crlandsc
# MIT License
#
# This implementation is based on and extends the loss functions described in:
# [1] "Separate This, and All of these Things Around It: Music Source Separation via Hyperellipsoidal Queries"
#     Karn N. Watcharasupat, Alexander Lerch
#     arXiv:2501.16171
# [2] "A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation"
#     Karn N. Watcharasupat, Chih-Wei Wu, Yiwei Ding, Iroro Orife, Aaron J. Hipple, Phillip A. Williams, Scott Kramer, Alexander Lerch, William Wolcott
#     IEEE Open Journal of Signal Processing, 2023
#     arXiv:2309.02539
# [3] "A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems"
#     Karn N. Watcharasupat, Alexander Lerch
#     Proceedings of the 25th International Society for Music Information Retrieval Conference, 2024
#     arXiv:2406.18747

import warnings

import torch
import torch.nn as nn
from torchaudio.transforms import Spectrogram
import math
from typing import Union, Dict, Tuple, Optional, List
    
def dbrms(x, eps=1e-8):
    """
    Compute RMS level in decibels for a batch of signals.
    Args:
        x: (batch, time) or (batch, ...) tensor
        eps: stability constant
    Returns:
        (batch,) tensor of dB RMS
    """
    x = x.reshape(x.shape[0], -1)
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1) + eps)
    return 20.0 * torch.log10(rms + eps)


class L1SNRLoss(torch.nn.Module):
    """
    Implements the L1 Signal-to-Noise Ratio (SNR) loss with optional weighted L1 loss 
    component to balance "all-or-nothing" behavior.

    Paper-aligned D1(ŷ; y) form:
      D1 = 10 * log10( (||ŷ - y||_1 + eps) / (||y||_1 + eps) )
    L1SNR_loss = mean(D1)

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
    """
    def __init__(
        self,
        name,
        weight: float = 1.0,
        eps: float = 1e-3,
        l1_weight: float = 0.0,
    ):
        super().__init__()
        self.name = name
        self.weight = weight
        self.eps = eps
        self.l1_weight = l1_weight

    def forward(self, estimates, actuals, *args, **kwargs):
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
        inv_mean = torch.mean(1.0 / (l1_error.detach() + self.eps))
        # w-independent scaling to match typical gradient magnitudes
        scale_time = c * inv_mean
        l1_term = torch.mean(l1_error) * scale_time

        loss = (1.0 - w) * l1snr_loss + w * l1_term
        return loss * self.weight


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
    ):
        super().__init__()
        self.name = name
        self.weight = weight
        self.lambda0 = lambda0          # minimum regularization weight
        self.delta_lambda = delta_lambda # range of extra weight
        self.l1snr_eps = l1snr_eps
        self.dbrms_eps = dbrms_eps
        self.lmin = lmin
        self.use_regularization = use_regularization
        
        # Validate l1_weight is between 0.0 and 1.0 inclusive
        assert 0.0 <= l1_weight <= 1.0, "l1_weight must be between 0.0 and 1.0 inclusive"
        self.l1_weight = l1_weight
        
        # Initialize component losses based on l1_weight
        if self.l1_weight == 1.0:
            # Pure L1 mode - only need L1 loss
            self.l1snr_loss = None
            self.l1_loss = torch.nn.L1Loss()
        else:
            # Standard mode with L1SNR (and optional weighted L1 if l1_weight > 0)
            self.l1snr_loss = L1SNRLoss(
                name="l1_snr",
                weight=1.0,  # We'll apply the weight at the end
                eps=l1snr_eps,
                l1_weight=l1_weight,
            )
            self.l1_loss = None
    
    @staticmethod
    def compute_adaptive_weight(L_pred, L_true, L_min, lambda0, delta_lambda, R):
        """
        Implements the adaptive weighting of the level-matching regularization term, per arXiv:2501.16171.
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
            total_loss = l1snr_loss + (l1snr_weight * torch.mean(reg_loss))
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
        window_fn (str): Window function for the STFT ('hann', 'hamming', etc.)
        min_audio_length (int): Minimum audio length required for processing.
            If audio is shorter, returns zero loss to avoid errors.
        use_regularization (bool): Whether to use level-matching regularization.
            If False, only the L1SNR (and optional L1) components are used.
        l1_weight (float): Weight for the L1 loss component. Default 0 (disabled).
            As this increases, the regularization term is also scaled down proportionally.
            When set to 1.0, efficiently computes only L1 loss.
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
    ):
        super().__init__()
        self.name = name
        self.weight = weight
        self.min_audio_length = min_audio_length
        
        # Validate STFT parameters
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths), "All STFT parameter lists must have the same length"
        
        # Store STFT parameters for validation during forward pass
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths
        self.window_fn_name = window_fn
        
        # Validate window sizes
        for n_fft, win_length in zip(n_ffts, win_lengths):
            assert n_fft >= win_length, f"FFT size ({n_fft}) must be greater than or equal to window length ({win_length})"
        
        # Pre-initialize Spectrogram transforms for maximum efficiency
        self.spectrogram_transforms = nn.ModuleList()
        window_fn_callable = getattr(torch, f"{window_fn}_window")
        
        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            # Create a spectrogram transform for each resolution
            transform = Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                pad_mode="reflect",
                center=True,
                window_fn=window_fn_callable,
                normalized=True,
                power=None,  # This ensures the output is complex
            )
            self.spectrogram_transforms.append(transform)
        
        # Parameters for spectrogram domain level-matching
        self.lambda0 = lambda0
        self.delta_lambda = delta_lambda
        self.lmin = lmin
        self.dbrms_eps = dbrms_eps
        self.l1snr_eps = l1snr_eps
        
        # Add L1 loss parameters and validate
        assert 0.0 <= l1_weight <= 1.0, "l1_weight must be between 0.0 and 1.0 inclusive"
        self.l1_weight = l1_weight
        
        # Flag for pure L1 mode
        self.pure_l1_mode = (self.l1_weight == 1.0)
        # Create L1 loss function for pure L1 mode
        if self.pure_l1_mode:
            self.l1_loss = torch.nn.L1Loss()
        else:
            self.l1_loss = None

        
        # Add this parameter to control regularization
        self.use_regularization = use_regularization
        # Coefficient to scale spectral regularization (disabled by default)
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
            use_regularization=False,  # regularizer belongs to TF for this class
            l1_weight=self.l1_weight,
        )

        # Simplified tracking
        self.nan_inf_counts = {"inputs": 0, "spec_loss": 0}

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
        err_re = torch.mean(torch.abs(est_re - act_re), dim=1)
        ref_re = torch.mean(torch.abs(act_re), dim=1)
        err_im = torch.mean(torch.abs(est_im - act_im), dim=1)
        ref_im = torch.mean(torch.abs(act_im), dim=1)

        # Paper-aligned D1 = 10*log10((||e||_1 + eps)/(||y||_1 + eps))
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
            inv_mean_comp = torch.mean(0.5 * (1.0 / (err_re.detach() + self.l1snr_eps) +
                                              1.0 / (err_im.detach() + self.l1snr_eps)))
            # w-independent scaling to match typical gradient magnitudes (factor 2.0 for Re/Im symmetry)
            scale_spec = 2.0 * c * inv_mean_comp
            l1_term = 0.5 * (torch.mean(err_re) + torch.mean(err_im)) * scale_spec

            loss = (1.0 - w) * d1_sum + w * l1_term
            return loss
        elif w >= 1.0:
            # Pure L1
            l1_term = 0.5 * (torch.mean(err_re) + torch.mean(err_im))
            return l1_term
        else:
            # Pure SNR (D1)
            return d1_sum
    
    def _compute_spec_level_matching(self, est_spec, act_spec):
        """
        Compute the level matching regularization term for a spectrogram.
        """
        batch_size = est_spec.shape[0]
        
        # Make sure dimensions match before operations
        if est_spec.shape != act_spec.shape:
            # Resize to match the smaller of the two
            min_freq = min(est_spec.shape[-2], act_spec.shape[-2])
            min_time = min(est_spec.shape[-1], act_spec.shape[-1])
            est_spec = est_spec[..., :min_freq, :min_time]
            act_spec = act_spec[..., :min_freq, :min_time]
        
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

    def _validate_audio_length(self, audio_length):
        """
        Validates that the audio is long enough for the STFT parameters.
        """
        if audio_length < self.min_audio_length:
            return False
            
        for n_fft, hop_length in zip(self.n_ffts, self.hop_lengths):
            n_frames = (audio_length // hop_length) + 1
            if n_frames < 2:
                return False
                
        return True

    def forward(self, estimates, actuals, *args, **kwargs):
        device = estimates.device
        batch_size = estimates.shape[0]
        
        # Basic NaN/Inf handling (simplified)
        if torch.isnan(estimates).any() or torch.isinf(estimates).any() or torch.isnan(actuals).any() or torch.isinf(actuals).any():
            self.nan_inf_counts["inputs"] += 1
            estimates = torch.nan_to_num(estimates, nan=0.0, posinf=1.0, neginf=-1.0)
            actuals = torch.nan_to_num(actuals, nan=0.0, posinf=1.0, neginf=-1.0)
        
        est_source = estimates.reshape(batch_size, -1, estimates.shape[-1])
        act_source = actuals.reshape(batch_size, -1, actuals.shape[-1])
        
        # Validate audio length
        audio_length = est_source.shape[-1]
        if not self._validate_audio_length(audio_length):
            # Fallback to time-domain L1SNR-style loss instead of zero
            return self.fallback_time_loss(estimates, actuals, *args, **kwargs) * self.weight
        
        # Track losses (initialize as tensors on the correct device for stability)
        total_spec_loss = torch.tensor(0.0, device=device)
        total_spec_reg_loss = torch.tensor(0.0, device=device)
        valid_transforms = 0
        
        # Ensure transforms are on the correct device
        self.spectrogram_transforms.to(device)
        
        # Process each resolution
        for i, transform in enumerate(self.spectrogram_transforms):
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
                    self.nan_inf_counts["spec_loss"] += 1
                    continue
                
                # Only compute regularization if not in pure L1 mode and regularization is enabled
                if not self.pure_l1_mode and self.use_regularization:
                    try:
                        spec_reg_loss = self._compute_spec_level_matching(est_spec, act_spec)
                        
                        # Check for numerical issues
                        if torch.isnan(spec_reg_loss) or torch.isinf(spec_reg_loss):
                            self.nan_inf_counts["spec_loss"] += 1
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
        
        # If all transforms failed, return zero loss
        if valid_transforms == 0:
            warnings.warn("All spectrogram transforms failed. Returning zero loss.")
            return torch.tensor(0.0, device=device)
        
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

        return final_loss * self.weight


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
        spec_weight (float): The weight for spectrogram domain loss relative to time domain.
            Default 0.5 (equal weighting). Set higher to emphasize spectral accuracy.
        use_time_regularization (bool): Whether to use level-matching regularization in time domain.
        use_spec_regularization (bool): Whether to use level-matching regularization in spectogram domain.
        l1_weight (float): Weight for the L1 loss component vs the L1SNR+reg components.
            Default 0 (disabled). As this increases, the regularization term is also scaled down.
            When set to 1.0, efficiently computes only L1 loss in both domains.
        lambda0 (float): Minimum regularization weight for both domains.
        delta_lambda (float): Range of extra weight for regularization in both domains.
        time_loss_params (dict): Optional additional parameters to pass to time domain loss.
        spec_loss_params (dict): Optional additional parameters to pass to spectrogram domain loss.
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
        # auto-balanced mixing used
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
        # Allow for separate parameter overrides (e.g. different delta_lambda for time and spec)
        time_loss_params: dict = None,
        spec_loss_params: dict = None,
    ):
        super().__init__()
        self.name = name
        self.weight = weight
        self.spec_weight = spec_weight
        
        # Validate l1_weight is in valid range
        assert 0.0 <= l1_weight <= 1.0, "l1_weight must be between 0.0 and 1.0 inclusive"
        self.l1_weight = l1_weight
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
            "l1_weight": l1_weight,

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
        self.time_loss = L1SNRDBLoss(**default_time_params)
        self.spec_loss = STFTL1SNRDBLoss(**default_spec_params)
        
        # For reference only, indicate if we're in pure L1 mode
        self.pure_l1_mode = (self.l1_weight == 1.0)

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
        # Compute time domain loss
        time_loss = self.time_loss(estimates, actuals, *args, **kwargs)
        
        # Compute spectrogram domain loss
        spec_loss = self.spec_loss(estimates, actuals, *args, **kwargs)
        
        # Combine with weighting
        combined_loss = (1 - self.spec_weight) * time_loss + self.spec_weight * spec_loss
        
        # Apply overall weight
        return combined_loss * self.weight
