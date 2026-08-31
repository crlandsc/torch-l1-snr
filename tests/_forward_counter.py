"""pytest plugin that records, per test, whether any loss forward() was called.

Used by mutation_gate.py --audit-markers to verify that every test claiming `no_forward` genuinely exercises
no loss forward, and that no test lacking the marker is silently exempt for the wrong reason.

An adversarial reviewer pointed out that nothing checked the marker's honesty: a future test could carry it
and be quietly excluded from the gate. The marker is visible in a diff, but "a reviewer might notice" is not
a gate. This makes it one.
"""
import json
import os

import pytest

import torch_l1_snr.l1snr as _l1snr

CLASSES = ["L1SNRLoss", "L2SNRLoss", "L1SNRDBLoss", "STFTL1SNRDBLoss", "MultiL1SNRDBLoss"]
_calls = {"n": 0}
_records = {}


def pytest_configure(config):
    for name in CLASSES:
        cls = getattr(_l1snr, name)
        original = cls.forward

        def counted(self, *a, _orig=original, **kw):
            _calls["n"] += 1
            return _orig(self, *a, **kw)

        cls.forward = counted


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    _calls["n"] = 0
    yield
    _records[item.nodeid] = {
        "forward_calls": _calls["n"],
        "marked_no_forward": item.get_closest_marker("no_forward") is not None,
    }


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("FORWARD_COUNTER_OUT")
    if path:
        with open(path, "w") as fh:
            json.dump(_records, fh, indent=1)
