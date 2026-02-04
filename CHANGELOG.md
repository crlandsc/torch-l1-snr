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
