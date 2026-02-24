# Changelog

## 0.1.0 (2026-01-29)

Initial release

## 0.1.1 (2026-01-30)

Improved README documentation with better context and clarity.

## 0.1.2 (2026-02-04)

**Bugfix:** Fixed gradient equivalence in L1/L1SNR auto-balancing.

The `l1_weight` parameter (for mixing L1 and L1SNR losses) was improperly applied due to scaling by error magnitude, which made L1 gradients identical to L1SNR gradients. Now scales by reference signal magnitude, preserving distinct gradient behaviors:
- L1SNR: inverse-error-scaled gradients (larger updates for smaller errors)
- L1: uniform gradients regardless of error magnitude

Credit: [Karn N. Watcharasupat](https://github.com/kwatcharasupat) for identifying the issue.

## 0.1.3 (2026-02-24)

**Fix:** MPS (Apple Silicon) compatibility for STFT-based losses.

PyTorch's MPS backend has a bug in `torch.abs()` backward on complex tensors that produces numerically incorrect (inflated) gradients. This caused `STFTL1SNRDBLoss` and `MultiL1SNRDBLoss` to produce gradient norms ~500,000x too large on MPS, corrupting model weights within a few training steps.

**Fix:** Added `mps_cpu_fallback` parameter (default `True`) to `STFTL1SNRDBLoss` and `MultiL1SNRDBLoss`. When on MPS, STFT loss computation is automatically routed through CPU via differentiable `.cpu()` calls, producing correct gradients. The scalar loss is moved back to MPS for the optimizer step.

- **MPS users:** Losses now work correctly out of the box. No code changes needed.
- **CUDA/CPU users:** Completely unaffected. The fallback is a no-op when not on MPS.
- **Time-domain losses** (`L1SNRLoss`, `L1SNRDBLoss`): Unaffected — they never had this issue.
- Set `mps_cpu_fallback=False` to disable the workaround if a future PyTorch release fixes the MPS bug.
