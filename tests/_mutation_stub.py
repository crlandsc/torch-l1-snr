"""pytest plugin that replaces every loss forward() with a graph-connected constant.

Used by mutation_gate.py. Loaded with `pytest -p _mutation_stub`, never during a normal run.

The stub keeps the autograd graph intact (estimates.sum() * 0.0 + 1.0) so .backward() still works and tests
fail on the loss carrying no information, rather than on a crash. A stub that raised would be detected by
almost any test and would tell us nothing about discriminating power.
"""
import torch_l1_snr.l1snr as _l1snr

CLASSES = ["L1SNRLoss", "L1SNRDBLoss", "STFTL1SNRDBLoss", "MultiL1SNRDBLoss"]


def _stub_forward(self, estimates, actuals, *args, **kwargs):
    return estimates.sum() * 0.0 + 1.0


def pytest_configure(config):
    for name in CLASSES:
        getattr(_l1snr, name).forward = _stub_forward
    config._mutation_stub_applied = sorted(CLASSES)


def pytest_report_header(config):
    return f"MUTATION STUB ACTIVE: forward() replaced by a constant on {', '.join(CLASSES)}"
