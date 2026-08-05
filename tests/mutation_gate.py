"""Mutation gate: fail if the numerical test suite cannot tell the real library from a constant.

This exists because of audit finding M2. Before the Phase 2 rebuild, replacing all four forward() methods
with a graph-connected constant left 122 of 127 tests passing: 96% of the suite could not detect that the
loss carried no information. 112 of those tests compared StemWrappedLoss against base_loss with
stem_dimension=None, where the wrapper forwards straight through, making the assertion structurally
f(x) == f(x).

Runs pytest as a subprocess because a test cannot meaningfully assert on the outcome of the run it is part of.

Scope is every numerical test file: tests/test_losses.py and tests/test_edge_cases.py. tests/test_docs.py is
excluded because it checks documentation strings, which a stubbed forward has no bearing on.

test_edge_cases.py was originally outside this gate, having been added after it. An adversarial reviewer
applied the same stub to it and found 28% detection against test_losses.py's 100%, with roughly a dozen tests
of exactly the M2 shape -- asserting only `ndim == 0` or `not isnan`, both of which a constant satisfies. A
gate that covers one of two test files is a gate over half the suite.

Usage:
    python tests/mutation_gate.py              # uses the default threshold
    python tests/mutation_gate.py --min 0.90   # require 90% of tests to fail under the stub
    python tests/mutation_gate.py --report     # print the ratio and exit 0 (for recording a baseline)
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGETS = ["tests/test_losses.py", "tests/test_edge_cases.py"]

# Set to 1.0 because that is what the rebuilt suite achieves, on CPU and on MPS hardware alike.
#
# It is exact rather than slack because the no_forward marker makes compliance cheap: a new test either
# detects a stubbed forward() or it declares, in the source, that it does not exercise one. Slack here would
# just be room for a tautology to reappear unnoticed, which is the entire defect this gate exists to prevent.
# Lowering it requires saying why in the commit message.
DEFAULT_MIN_FAIL_RATIO = 1.0


def run_stubbed():
    # Exclude tests marked no_forward: they legitimately never call a loss forward() (the reference
    # self-validation, the dbrms tests), so counting them would cap the achievable ratio below 100% and
    # invite a fudged threshold. The marker is explicit in the source, so a new test cannot dodge the gate
    # without a reviewer seeing it.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *TARGETS, "-p", "_mutation_stub", "-q", "--no-header",
         "--tb=no", "-m", "not no_forward"],
        cwd=REPO, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO / "tests")},
    )
    out = proc.stdout + proc.stderr
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", out)) else 0
    return failed, passed, errors, out


def audit_markers():
    """Verify every no_forward claim by counting actual forward() calls.

    The marker excludes a test from the ratio, so an untrue marker is a hole in the gate. Nothing checked it
    until an adversarial reviewer pointed out that "a reviewer would see it in the diff" is not a gate.
    """
    import json
    import tempfile
    out = tempfile.mktemp(suffix=".json")
    subprocess.run(
        [sys.executable, "-m", "pytest", *TARGETS, "-p", "_forward_counter", "-q", "--no-header",
         "--tb=no"],
        cwd=REPO, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO / "tests"),
             "FORWARD_COUNTER_OUT": out},
    )
    try:
        records = json.load(open(out))
    except Exception:
        print("MARKER AUDIT: could not collect forward-call records")
        return 1
    liars = [k for k, v in records.items() if v["marked_no_forward"] and v["forward_calls"] > 0]
    marked = sum(1 for v in records.values() if v["marked_no_forward"])
    print(f"marker audit: {marked} tests claim no_forward, {len(records)} tests inspected")
    if liars:
        print("\nMARKER AUDIT FAILED: these tests claim no_forward but do call a loss forward():")
        for k in liars:
            print(f"  {k}  ({records[k]['forward_calls']} calls)")
        return 1
    print("marker audit passed: every no_forward claim is true")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, default=DEFAULT_MIN_FAIL_RATIO)
    ap.add_argument("--report", action="store_true", help="print the ratio and exit 0")
    ap.add_argument("--audit-markers", action="store_true",
                    help="verify every no_forward marker by counting forward() calls")
    args = ap.parse_args()

    if args.audit_markers:
        return audit_markers()

    failed, passed, errors, out = run_stubbed()
    total = failed + passed
    if total == 0:
        print("MUTATION GATE: could not parse a pytest result. Raw output follows.\n")
        print(out)
        return 1

    ratio = failed / total
    print(f"stubbed forward() to a constant on all four loss classes")
    print(f"  {failed} failed, {passed} passed" + (f", {errors} errors" if errors else ""))
    print(f"  detection ratio: {ratio:.1%} of {total} tests noticed")
    print(f"  required:        {args.min:.1%}")

    if args.report:
        print("\n(--report: recording only, not enforcing)")
        return 0

    if ratio < args.min:
        print(f"\nMUTATION GATE FAILED: only {ratio:.1%} of the suite can distinguish the real library "
              f"from a constant.\nTests that pass against a stubbed forward() are not testing the loss.")
        return 1

    print("\nMUTATION GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
