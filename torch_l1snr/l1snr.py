# Separate This, and All of these Things Around It: Music Source Separation via Hyperellipsoidal Queries
# Karn N. Watcharasupat, Alexander Lerch
# arXiv:2501.16171

# A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation
# Karn N. Watcharasupat, Chih-Wei Wu, Yiwei Ding, Iroro Orife, Aaron J. Hipple, Phillip A. Williams, Scott Kramer, Alexander Lerch, William Wolcott
# IEEE Open Journal of Signal Processing, 2023
# arXiv:2309.02539

# A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems
# Karn N. Watcharasupat, Alexander Lerch
# Proceedings of the 25th International Society for Music Information Retrieval Conference, 2024
# arXiv:2406.18747

import torch
import torch.nn as nn
import torchaudio
from typing import Union, List, Optional, Dict, Tuple

def dbrms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute RMS level in decibels for a batch of signals.

    Args:
        x (torch.Tensor): Input tensor of shape `(batch, ...)`
        eps (float): Stability constant to avoid log(0).

    Returns:
        torch.Tensor: A tensor of shape `(batch,)` containing dBRMS values.
    """
    x = x.reshape(x.shape[0], -1)  # Shape: [batch, num_samples]
    rms = torch.sqrt(torch.mean(x**2, dim=-1) + eps)
    return 20.0 * torch.log10(rms + eps)


class L1SNRLoss(nn.Module):
    """
    Implements a standalone L1 Signal-to-Noise Ratio (SNR) loss in the time domain,
    as proposed in [1].

    The L1SNR loss is defined as:
    SNR = 20 * log10((L1_true + eps) / (L1_error + eps))
    L1SNR_loss = -mean(SNR)
    
    Where L1_true is the mean absolute value of the target signal and L1_error is the
    mean absolute error between the estimate and target.

    Attributes:
        eps (float): A small epsilon value to avoid division by zero and stabilize log.

    References:
        [1] K. N. Watcharasupat, C.-W. Wu, Y. Ding, I. Orife, A. J. Hipple,
            P. A. Williams, S. Kramer, A. Lerch, and W. Wolcott, "A Generalized
            Bandsplit Neural Network for Cinematic Audio Source Separation,"
            IEEE Open Journal of Signal Processing, 2023. (arXiv:2309.02539)
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, estimates: torch.Tensor, actuals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimates (torch.Tensor): Estimated signals.
                                      Shape: `(batch_size, num_samples)`
            actuals (torch.Tensor): Ground truth signals.
                                    Shape: `(batch_size, num_samples)`
        
        Returns:
            torch.Tensor: The calculated L1 SNR loss (scalar).
        """
        # Shape: [batch_size]
        l1_error = torch.mean(torch.abs(estimates - actuals), dim=-1)
        # Shape: [batch_size]
        l1_true = torch.mean(torch.abs(actuals), dim=-1)
        # Shape: [batch_size]
        snr = 20.0 * torch.log10((l1_true + self.eps) / (l1_error + self.eps))
        
        return -torch.mean(snr)


class L1SNRDBLoss(nn.Module):
    """
    This module extends the L1SNR concept by incorporating the adaptive
    level-matching regularization technique described in [1]. It also includes an
    optional L1 loss component to balance "all-or-nothing" behavior.

    The loss combines three components:
    1. L1SNR loss: -mean(20*log10((l1_true + eps) / (l1_error + eps)))
    2. Level-matching regularization: λ*|L_pred - L_true|
       Where λ is adaptively computed based on the signal levels.
    3. Optional L1 loss: mean(l1_error) * l1_scale

    Attributes:
        lambda0 (float): Minimum regularization weight (λ_min).
        delta_lambda (float): Range of extra weight for regularization (Δλ).
        l1snr_eps (float): Epsilon for the L1SNR component to avoid log(0).
        dbrms_eps (float): Epsilon for dBRMS calculation to avoid log(0).
        lmin (float): Minimum dBRMS considered non-silent for adaptive weighting.
        use_regularization (bool): Whether to use level-matching regularization.
        l1_weight (float): Weight for the L1 loss component (0.0 to 1.0).
                           When 1.0, this module computes only L1 loss.
        l1_scale (float): Scaling factor for L1 loss.

    References:
        [1] K. N. Watcharasupat and A. Lerch, "Separate This, and All of these
            Things Around It: Music Source Separation via Hyperellipsoidal Queries,"
            arXiv:2501.16171.
    """
    def __init__(
        self, 
        lambda0: float = 0.1,
        delta_lambda: float = 0.9,
        l1snr_eps: float = 1e-4,
        dbrms_eps: float = 1e-8,
        lmin: float = -60.0,
        use_regularization: bool = True,
        l1_weight: float = 0.0,
        l1_scale: float = 100.0,
    ):
        super().__init__()
        self.lambda0 = lambda0
        self.delta_lambda = delta_lambda
        self.l1snr_eps = l1snr_eps
        self.dbrms_eps = dbrms_eps
        self.lmin = lmin
        self.use_regularization = use_regularization
        
        assert 0.0 <= l1_weight <= 1.0, "l1_weight must be between 0.0 and 1.0"
        self.l1_weight = l1_weight
        self.l1_scale = l1_scale
        
        self.l1_loss = nn.L1Loss() if self.l1_weight == 1.0 else None

    @staticmethod
    def compute_adaptive_weight(
        L_pred: torch.Tensor, 
        L_true: torch.Tensor, 
        L_min: float, 
        lambda0: float, 
        delta_lambda: float, 
        R: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the adaptive weight for regularization.
        """
        max_val = torch.max(L_pred, torch.full_like(L_true, L_min))
        eta = (L_true > max_val).float()
        denom = (L_true - L_min).clamp(min=1e-6)
        clamp_arg = (R / denom).clamp(0.0, 1.0)
        lam = lambda0 + eta * delta_lambda * clamp_arg
        return lam.detach()
    
    def forward(self, estimates: torch.Tensor, actuals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimates (torch.Tensor): Estimated signals.
                                      Shape: `(batch_size, num_samples)`
            actuals (torch.Tensor): Ground truth signals.
                                    Shape: `(batch_size, num_samples)`
        
        Returns:
            torch.Tensor: The calculated loss (scalar).
        """
        # Efficient path for pure L1 loss
        if self.l1_loss is not None:
            return self.l1_loss(estimates, actuals) * self.l1_scale
        
        # --- L1SNR Component ---
        # Shape: [batch_size]
        l1_error = torch.mean(torch.abs(estimates - actuals), dim=-1)
        # Shape: [batch_size]
        l1_true = torch.mean(torch.abs(actuals), dim=-1)
        # Shape: [batch_size]
        snr = 20.0 * torch.log10((l1_true + self.l1snr_eps) / (l1_error + self.l1snr_eps))
        snr_loss = -torch.mean(snr)
        
        # --- L1 Component ---
        l1_loss_component = torch.mean(l1_error) * self.l1_scale
        
        # Combine L1SNR and L1 with weighting
        total_loss = ((1.0 - self.l1_weight) * snr_loss) + (self.l1_weight * l1_loss_component)

        # --- Regularization Component ---
        if self.use_regularization:
            # Shape: [batch_size]
            L_true = dbrms(actuals, self.dbrms_eps)
            # Shape: [batch_size]
            L_pred = dbrms(estimates, self.dbrms_eps)
            # Shape: [batch_size]
            R = torch.abs(L_pred - L_true)
            
            # Shape: [batch_size]
            lambda_weight = self.compute_adaptive_weight(L_pred, L_true, self.lmin, self.lambda0, self.delta_lambda, R)
            reg_loss = torch.mean(lambda_weight * R)
            
            # Regularization is only applied to the L1SNR part of the loss
            l1snr_component_weight = 1.0 - self.l1_weight
            total_loss = total_loss + (l1snr_component_weight * reg_loss)
            
        return total_loss


class STFTL1SNRDBLoss(nn.Module):
    """
    This module adapts the L1SNRDBLoss for the spectrogram domain, applying it
    across multiple STFT resolutions, based on the methods described in [1].

    This loss operates on complex spectrograms across multiple time-frequency resolutions.

    Attributes:
        lambda0, delta_lambda, lmin: Regularization parameters.
        l1snr_eps, dbrms_eps: Epsilon values for stability.
        n_ffts, hop_lengths, win_lengths: Lists of STFT parameters for multi-resolution analysis.
        window_fn (str): Name of the window function for STFT (e.g., 'hann').
        min_audio_length (int): Minimum audio length required for STFT processing.
        use_regularization (bool): Flag to enable/disable level-matching regularization.
        l1_weight (float): Weight for the L1 loss component (0.0 to 1.0). When 1.0, computes only L1 loss.
        l1_scale (float): Scaling factor for the spectrogram-domain L1 loss.
    
    References:
        [1] K. N. Watcharasupat and A. Lerch, "Separate This, and All of these
            Things Around It: Music Source Separation via Hyperellipsoidal Queries,"
            arXiv:2501.16171.
    """
    def __init__(
        self, 
        lambda0: float = 0.1,
        delta_lambda: float = 0.9,
        l1snr_eps: float = 1e-4,
        dbrms_eps: float = 1e-8,
        lmin: float = -60.0,
        n_ffts: List[int] = [512, 1024, 2048, 4096],
        hop_lengths: List[int] = [128, 256, 512, 1024],
        win_lengths: List[int] = [512, 1024, 2048, 4096],
        window_fn: str = 'hann',
        min_audio_length: int = 512,
        use_regularization: bool = True,
        l1_weight: float = 0.0,
        l1_scale: float = 10.0,
    ):
        super().__init__()
        self.min_audio_length = min_audio_length
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths), "STFT parameter lists must have the same length"
        
        self.n_ffts = n_ffts
        for n_fft, win_length in zip(n_ffts, win_lengths):
            assert n_fft >= win_length, f"FFT size ({n_fft}) must be >= window length ({win_length})"
        
        self.spectrogram_transforms = nn.ModuleList()
        window_fn_callable = getattr(torch, f"{window_fn}_window")
        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            transform = torchaudio.transforms.Spectrogram(
                n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                pad_mode="reflect", center=True, window_fn=window_fn_callable, power=None,
            )
            self.spectrogram_transforms.append(transform)
        
        self.lambda0 = lambda0
        self.delta_lambda = delta_lambda
        self.lmin = lmin
        self.dbrms_eps = dbrms_eps
        self.l1snr_eps = l1snr_eps
        self.use_regularization = use_regularization

        assert 0.0 <= l1_weight <= 1.0, "l1_weight must be between 0.0 and 1.0"
        self.l1_weight = l1_weight
        self.l1_scale = l1_scale
        
        self.pure_l1_mode = (self.l1_weight == 1.0)
        self.l1_loss = nn.L1Loss() if self.pure_l1_mode else None

    def _compute_complex_spec_l1snr_loss(self, est_spec: torch.Tensor, act_spec: torch.Tensor) -> torch.Tensor:
        """
        Computes L1-SNR loss or L1 loss on complex spectrograms.
        """
        batch_size = est_spec.shape[0]
        # Shape: [batch, freq, time, 2]
        est_spec_real = torch.view_as_real(est_spec)
        act_spec_real = torch.view_as_real(act_spec)
        
        if self.pure_l1_mode:
            return self.l1_loss(est_spec_real, act_spec_real) * self.l1_scale
        
        # Shape: [batch, num_elements]
        est_spec_flat = est_spec_real.reshape(batch_size, -1)
        act_spec_flat = act_spec_real.reshape(batch_size, -1)
        
        # --- L1SNR Component ---
        # Shape: [batch]
        l1_error = torch.mean(torch.abs(est_spec_flat - act_spec_flat), dim=1)
        # Shape: [batch]
        l1_true = torch.mean(torch.abs(act_spec_flat), dim=1)
        # Shape: [batch]
        snr = 20.0 * torch.log10((l1_true + self.l1snr_eps) / (l1_error + self.l1snr_eps))
        snr_loss = -torch.mean(snr)
        
        if self.l1_weight <= 0.0:
            return snr_loss
        
        # --- L1 Component ---
        l1_loss_component = torch.mean(l1_error) * self.l1_scale
        return ((1.0 - self.l1_weight) * snr_loss) + (self.l1_weight * l1_loss_component)
    
    def _compute_spec_level_matching(self, est_spec: torch.Tensor, act_spec: torch.Tensor) -> torch.Tensor:
        """
        Computes the level matching regularization term on spectrogram magnitudes.
        """
        # Shape: [batch, freq, time]
        est_mag = torch.abs(est_spec)
        act_mag = torch.abs(act_spec)
        
        # Shape: [batch]
        L_true = dbrms(act_mag, self.dbrms_eps)
        # Shape: [batch]
        L_pred = dbrms(est_mag, self.dbrms_eps)
        # Shape: [batch]
        R = torch.abs(L_pred - L_true)
        
        # Shape: [batch]
        lambda_weight = L1SNRDBLoss.compute_adaptive_weight(
            L_pred, L_true, self.lmin, self.lambda0, self.delta_lambda, R
        )
        return torch.mean(lambda_weight * R)

    def _validate_and_trim_specs(self, est_spec: torch.Tensor, act_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if est_spec.shape != act_spec.shape:
            min_freq = min(est_spec.shape[1], act_spec.shape[1])
            min_time = min(est_spec.shape[2], act_spec.shape[2])
            est_spec = est_spec[:, :min_freq, :min_time]
            act_spec = act_spec[:, :min_freq, :min_time]
        return est_spec, act_spec

    def forward(self, estimates: torch.Tensor, actuals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimates (torch.Tensor): Estimated signals.
                                      Shape: `(batch_size, num_samples)`
            actuals (torch.Tensor): Ground truth signals.
                                    Shape: `(batch_size, num_samples)`
        Returns:
            torch.Tensor: The calculated multi-resolution spectral loss (scalar).
        """
        if estimates.shape[-1] < self.min_audio_length:
            return torch.tensor(0.0, device=estimates.device)
        
        total_spec_loss = 0.0
        total_spec_reg_loss = 0.0
        valid_transforms = 0
        
        self.spectrogram_transforms = self.spectrogram_transforms.to(estimates.device)
        
        for transform in self.spectrogram_transforms:
            try:
                # Shape: [batch, freq, time] (complex)
                est_spec = transform(estimates)
                act_spec = transform(actuals)
                
                est_spec, act_spec = self._validate_and_trim_specs(est_spec, act_spec)
                
                spec_loss = self._compute_complex_spec_l1snr_loss(est_spec, act_spec)
                if torch.isnan(spec_loss) or torch.isinf(spec_loss): continue
                
                if not self.pure_l1_mode and self.use_regularization:
                    spec_reg_loss = self._compute_spec_level_matching(est_spec, act_spec)
                    if not (torch.isnan(spec_reg_loss) or torch.isinf(spec_reg_loss)):
                        total_spec_reg_loss += spec_reg_loss
                
                total_spec_loss += spec_loss
                valid_transforms += 1
            except RuntimeError:
                continue
        
        if valid_transforms == 0:
            return torch.tensor(0.0, device=estimates.device)
        
        avg_spec_loss = total_spec_loss / valid_transforms
        
        if not self.pure_l1_mode and self.use_regularization:
            avg_spec_reg_loss = total_spec_reg_loss / valid_transforms
            l1snr_weight = 1.0 - self.l1_weight
            final_loss = avg_spec_loss + l1snr_weight * avg_spec_reg_loss
        else:
            final_loss = avg_spec_loss
        
        return final_loss


class MultiL1SNRDBLoss(nn.Module):
    """
    A modular loss that combines time-domain and spectrogram-domain L1SNRDB losses.
    This loss is designed for audio source separation and can be configured to process
    either all stems jointly or a single specified stem.

    The final loss is a weighted sum of the time-domain and spectrogram-domain losses:
    Loss = weight * [(1 - spec_weight) * time_loss + spec_weight * spec_loss]

    Each domain's loss is a combination of L1SNR, adaptive level-matching regularization,
    and an optional L1 component, controlled by the `l1_weight` parameter. Setting
    `l1_weight=1.0` efficiently computes only L1 loss in both domains.

    Args:
        name (str): The name identifier for the loss instance.
        stem_dimension (Union[int, None]): The specific stem dimension to apply the loss on.
            If None, the loss is applied to all stems jointly after reshaping.
            The input tensor is expected to be `(batch, stems, channels, samples)`.
            This will be flattened to `(batch, stems * channels * samples)` for loss calculation.
        weight (float): The overall weight multiplier for the final combined loss.
        spec_weight (float): The weight for the spectrogram domain loss relative to the
            time domain loss. `(0.0 <= spec_weight <= 1.0)`.
        l1_weight (float): Weight for the L1 loss component vs. L1SNR+regularization.
            `0.0` (default) disables the L1 component. `1.0` computes *only* L1 loss.
        l1_scale_time (float): Scaling factor for the time-domain L1 loss.
        l1_scale_spec (float): Scaling factor for the spectrogram-domain L1 loss.
        use_time_regularization (bool): Enable/disable time-domain regularization.
        use_spec_regularization (bool): Enable/disable spectrogram-domain regularization.
        time_loss_params (dict): Optional dictionary to override default parameters
                                 for the time-domain `L1SNRDBLoss` component.
        spec_loss_params (dict): Optional dictionary to override default parameters
                                 for the spectrogram-domain `STFTL1SNRDBLoss` component.
        **kwargs: Catches and forwards any other shared parameters to both time and spec
                  loss components (e.g., `lambda0`, `delta_lambda`, `lmin`, etc.).
    """
    def __init__(
        self, 
        name: str,
        stem_dimension: Optional[int] = None, 
        weight: float = 1.0,
        spec_weight: float = 0.5,
        l1_weight: float = 0.0,
        l1_scale_time: float = 100.0,
        l1_scale_spec: float = 10.0,
        use_time_regularization: bool = True,
        use_spec_regularization: bool = True,
        time_loss_params: Optional[Dict] = None,
        spec_loss_params: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.stem_dimension = stem_dimension
        self.weight = weight
        self.spec_weight = spec_weight
        
        assert 0.0 <= l1_weight <= 1.0, "l1_weight must be between 0.0 and 1.0"

        # --- Configure Time-Domain Loss ---
        time_params = kwargs.copy()
        time_params.update({
            "l1_weight": l1_weight,
            "l1_scale": l1_scale_time,
            "use_regularization": use_time_regularization,
        })
        if time_loss_params:
            time_params.update(time_loss_params)
        self.time_loss = L1SNRDBLoss(**time_params)
        
        # --- Configure Spectrogram-Domain Loss ---
        spec_params = kwargs.copy()
        spec_params.update({
            "l1_weight": l1_weight,
            "l1_scale": l1_scale_spec,
            "use_regularization": use_spec_regularization,
        })
        if spec_loss_params:
            spec_params.update(spec_loss_params)
        self.spec_loss = STFTL1SNRDBLoss(**spec_params)

    def forward(self, estimates: torch.Tensor, actuals: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass to compute the combined multi-domain loss.

        Args:
            estimates (torch.Tensor): Model output predictions.
                                      Shape: `(batch, stems, channels, samples)`
            actuals (torch.Tensor): Ground truth targets.
                                    Shape: `(batch, stems, channels, samples)`
        
        Returns:
            torch.Tensor: The final combined weighted loss (scalar).
        """
        batch_size = estimates.shape[0]

        # Select a specific stem or flatten all stems for processing
        if self.stem_dimension is not None:
            # Shape: [batch, channels, samples] 
            est_source_unflat = estimates[:, self.stem_dimension, ...]
            act_source_unflat = actuals[:, self.stem_dimension, ...]
        else:
            # Shape: [batch, stems, channels, samples] 
            est_source_unflat = estimates
            act_source_unflat = actuals

        # Reshape to (batch, -1) for loss components
        est_source = est_source_unflat.reshape(batch_size, -1)
        act_source = act_source_unflat.reshape(batch_size, -1)

        # Compute time-domain loss
        time_loss = self.time_loss(est_source, act_source)
        
        # Compute spectrogram-domain loss
        spec_loss = self.spec_loss(est_source, act_source)
        
        # Combine with weighting
        combined_loss = (1 - self.spec_weight) * time_loss + self.spec_weight * spec_loss
        
        return combined_loss * self.weight 