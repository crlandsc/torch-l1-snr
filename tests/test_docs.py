"""Documentation gates.

Prose in a published library is a contract. Every claim asserted here shipped wrong at some point, so each
test exists to keep a specific claim honest rather than to check style.

A string check proves a claim is present or absent; it cannot prove the prose is correct. These tests are
therefore the regression half of each documentation fix, not the whole of it.
"""
import inspect
import re
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch_l1_snr
from torch_l1_snr import (
    L1SNRLoss,
    L1SNRDBLoss,
    STFTL1SNRDBLoss,
    MultiL1SNRDBLoss,
)

README = (REPO / "README.md").read_text()
CHANGELOG = (REPO / "CHANGELOG.md").read_text()
SOURCE = (REPO / "torch_l1_snr" / "l1snr.py").read_text()
LICENSE = (REPO / "LICENSE").read_text()
SETUP_CFG = (REPO / "setup.cfg").read_text()

ALL_CLASSES = [L1SNRLoss, L1SNRDBLoss, STFTL1SNRDBLoss, MultiL1SNRDBLoss]

# Every version published to PyPI, established from the PyPI JSON API during the audit.
PUBLISHED_VERSIONS = ["0.0.1", "0.0.2", "0.0.3", "0.0.4", "0.0.5",
                      "0.1.0", "0.1.1", "0.1.2", "0.1.3"]


def readme_python_blocks():
    """Fenced ```python blocks that are self-contained programs.

    A block without an import is a fragment illustrating one call (README's mps_cpu_fallback snippet), not a
    runnable example. The audit excluded it on the same grounds.
    """
    blocks = re.findall(r"```python\n(.*?)```", README, re.DOTALL)
    return [b for b in blocks if "import" in b]


# --------------------------------------------------------------------------------------
# B1 / T1-6 -- the examples must actually run
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("idx", range(len(readme_python_blocks())))
def test_readme_example_runs(idx):
    """M1: four of five runnable examples crashed on loss.backward() -- no requires_grad."""
    block = readme_python_blocks()[idx]
    exec(compile(block, f"<README block {idx}>", "exec"), {"__name__": "__readme__"})


def test_every_loss_example_calls_backward():
    """M1: one example survived only because it omitted .backward(). Coverage must be uniform.

    Scoped to blocks that construct a loss. A utility example (dbrms returns a level, not an objective) has
    no backward pass to exercise, and demanding one would be nonsense rather than rigour.
    """
    missing = [
        i for i, b in enumerate(readme_python_blocks())
        if re.search(r"\w*Loss\(", b) and ".backward()" not in b
    ]
    assert not missing, f"README loss example(s) {missing} never exercise the backward pass"


# --------------------------------------------------------------------------------------
# B2 -- the MPS root cause
# --------------------------------------------------------------------------------------

def _mps_passages():
    changelog_013 = CHANGELOG.split("## 0.1.3")[1].split("## ")[0]
    readme_mps = README.split("**MPS note:**")[1].split("##")[0]
    source_mps = SOURCE.split("Note: PyTorch's MPS backend")[1][:900]
    passages = {"CHANGELOG 0.1.3": changelog_013, "README": readme_mps, "l1snr.py": source_mps}
    # Drop blockquoted editorial notes. A retraction has to name the wrong cause to be intelligible to
    # someone who read the original, so quoting "torch.abs" inside one is correct rather than a relapse.
    # What must not survive anywhere is the *claim* that torch.abs is the cause.
    return {k: "\n".join(l for l in v.splitlines() if not l.lstrip().startswith(">"))
            for k, v in passages.items()}


@pytest.mark.parametrize("where", ["CHANGELOG 0.1.3", "README", "l1snr.py"])
def test_mps_cause_names_stft_not_abs(where):
    """M12: all three blamed torch.abs() on complex, a path that does not run by default.

    Measured with a dtype-recording spy: default config calls torch.abs on COMPLEX zero times. The real
    cause is torch.stft backward above ~65k samples (PYTORCH_BUG_REPORT.md).
    """
    text = _mps_passages()[where]
    assert "torch.stft" in text, f"{where} does not name torch.stft as the cause"
    assert "torch.abs" not in text, f"{where} still attributes the bug to torch.abs"


# --------------------------------------------------------------------------------------
# B3, B4, B5 -- attribution and citations
# --------------------------------------------------------------------------------------

def test_attribution_names_upstream_repos():
    """Q7: neither official repository was named or linked, and no upstream copyright was carried."""
    for where, text in [("source header", SOURCE[:3000]), ("README", README)]:
        assert "kwatcharasupat/bandit" in text, f"{where} does not name the bandit repo"
        assert "kwatcharasupat/query-bandit" in text, f"{where} does not name the query-bandit repo"
        assert "Apache-2.0" in text or "Apache 2.0" in text, f"{where} omits bandit's license"
        assert "MIT" in text, f"{where} omits query-bandit's license"


def test_paper1_citation_is_2024_with_doi():
    """Q20: cited as 'IEEE Open Journal of Signal Processing, 2023'. The record says vol. 5, pp. 73-81, 2024."""
    for where, text in [("source header", SOURCE[:3000]), ("README", README)]:
        assert "10.1109/OJSP.2023.3339428" in text, f"{where} omits the DOI"
        assert re.search(r"Signal Processing.{0,40}2024", text), f"{where} does not date paper [1] to 2024"
        assert not re.search(r"Signal Processing, 2023", text), f"{where} still says 2023"


def test_reference_numbering_agrees_between_source_and_readme():
    """Q19: [1] and [2] are swapped. Source header has [1]=2501.16171; README has [1]=2309.02539.

    Splits on the [n] markers rather than matching across the gap: the README writes the id as
    "[arXiv:2309.02539]", so a gap pattern excluding brackets yields an empty map and the test then fails
    for the wrong reason.
    """
    def numbering(text):
        out = {}
        parts = re.split(r"\[(\d)\]", text)
        for i in range(1, len(parts) - 1, 2):
            n, body = parts[i], parts[i + 1]
            m = re.search(r"(\d{4}\.\d{5})", body)
            if m and n not in out:
                out[n] = m.group(1)
        return out
    src = numbering(SOURCE[:3000])
    rdm = numbering(README.split("## References")[1])
    assert src and rdm, f"numbering parse failed -- src={src} readme={rdm} (fix the test, not the docs)"
    assert src == rdm, f"reference numbering disagrees: source {src} vs README {rdm}"


# --------------------------------------------------------------------------------------
# B6 -- release history
# --------------------------------------------------------------------------------------

def test_changelog_covers_every_published_version():
    """Q5: five published releases omitted and the first mislabelled. 0.0.1 shipped 2026-01-17."""
    missing = [v for v in PUBLISHED_VERSIONS if f"## {v}" not in CHANGELOG]
    assert not missing, f"CHANGELOG has no entry for published version(s): {missing}"


def test_changelog_newest_entry_matches_version():
    """Q8: nothing tied the declared version to the release notes."""
    newest = re.search(r"## (\d+\.\d+\.\d+)", CHANGELOG).group(1)
    assert newest == torch_l1_snr.__version__, (
        f"newest CHANGELOG entry {newest} != __version__ {torch_l1_snr.__version__}"
    )


# --------------------------------------------------------------------------------------
# B7, B8, B9 -- docstring accuracy
# --------------------------------------------------------------------------------------

def test_no_returns_zero_loss_claim():
    """M16: docstring said short audio 'returns zero loss'; the code returns a time-domain fallback."""
    assert "returns zero loss" not in SOURCE.lower(), (
        "docstring still claims short audio returns zero loss"
    )


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
def test_every_constructor_param_is_documented(cls):
    """Q1: 10 constructor params are absent from their class docstrings, spec_reg_coef among them.

    Parametrized so the four classes cannot drift apart again.

    Checks the Attributes section against the `name (type):` convention rather than searching the whole
    docstring: spec_reg_coef appears in a formula in STFTL1SNRDBLoss's body ("... * spec_reg_coef * mean(...)")
    while being absent from its Attributes list, so a plain substring check passes vacuously.
    """
    doc = cls.__doc__ or ""
    attrs = doc.split("Attributes:")[1] if "Attributes:" in doc else ""
    params = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
    missing = [p for p in params
               if not re.search(rf"^\s*{re.escape(p)}\s*\(", attrs, re.MULTILINE)]
    assert not missing, f"{cls.__name__} Attributes section omits: {missing}"


def test_spec_weight_default_stays_half():
    """M17 retraction: 0.5 is paper-faithful -- it uniquely reproduces paper [3] eq. (3)'s 1:1:1 weighting.

    A guard against a future 'fix' moving it. The earlier 0.3686 recommendation was wrong.
    """
    assert inspect.signature(MultiL1SNRDBLoss.__init__).parameters["spec_weight"].default == 0.5


def test_regularizer_constants_marked_as_local_choices():
    """Q21: lambda0/delta_lambda/lmin sit under 'per arXiv:2501.16171', but the paper specifies no values.

    No official implementation of that paper exists either (S22), so these have no upstream source at all.
    """
    doc = (L1SNRDBLoss.__doc__ or "") + (L1SNRDBLoss.compute_adaptive_weight.__doc__ or "")
    assert "local choice" in doc.lower(), (
        "the regularizer constants are not marked as local choices rather than paper-derived values"
    )


def test_d1_notation_is_mean_not_sum():
    """Q4: docstrings render D1 with an L1 norm where the code takes a mean. A literal sum differs ~1.5 dB.

    The mean is correct -- it is bit-exact to the authors' official implementation (S1). Only notation is wrong.
    """
    for bad in ["||ŷ - y||_1", "||e||_1", "||y||_1", "‖"]:
        assert bad not in SOURCE, f"D1 notation still renders a sum: {bad!r}"


# --------------------------------------------------------------------------------------
# B10, B12, B14, B16, B17 -- README claims
# --------------------------------------------------------------------------------------

def test_no_proportional_l1_weight_claims():
    """M22/B18: the README stated l1_weight as a behaviour fraction ("l1_weight=0.1 blends 90% L1SNR with
    10% L1"). Measured, 0.1 delivers 3.6%-8.2% depending on the batch's level spread."""
    for bad in ["10% L1 loss", "90% L1SNR", "blends 90%"]:
        assert bad not in README, f"README still states l1_weight as a proportion: {bad!r}"
    assert "interpolation coefficient, not a behaviour fraction" in README, (
        "README does not state what l1_weight actually is")


def test_readme_documents_ref_level_and_its_derivation():
    """A14 introduced two parameters whose defaults rest on measurements; both must be explained."""
    assert "ref_level" in README
    assert "spec_ref_level" in README
    assert "0.19" in README, "the measured STFT-to-time reference ratio is not stated"


def test_readme_states_the_per_domain_difference():
    """M23: at l1_weight=0.5 the time domain moves ~25% toward L1 and the spectrogram ~45%, so the single
    knob means different things in the two halves of MultiL1SNRDBLoss."""
    assert "different things in its two halves" in README or "two halves" in README, (
        "README does not state that l1_weight's effect differs by domain")


def test_no_unscoped_gradient_matching_claim():
    """Q3: holds only near 0 dB SNR. Measured ratio 1.00 at 0 dB, 9.00 at 20 dB, 74.81 at 60 dB."""
    assert "approximately match gradient magnitudes" not in README, (
        "README still claims the scaling approximately matches gradient magnitudes"
    )


def test_nan_robustness_claim_is_scoped():
    """Q12: only STFTL1SNRDBLoss (and MultiL1SNRDBLoss through it) handle NaN/Inf. The time losses do not."""
    bullet = re.search(r"\*\*Numerical Stability\*\*.*", README).group(0)
    assert "STFTL1SNRDBLoss" in bullet, (
        "the NaN/Inf robustness claim is not scoped to the losses that actually provide it"
    )


def test_novelty_claim_excludes_upstream_feature():
    """Q2: time/spectrogram balancing is listed as novel, but bandit/core/loss/_timefreq.py:11-40 has
    time_weight/freq_weight upstream. The single-knob spec_weight is a convenience reskin.

    Scoped to the intro paragraph, which is where the "novel algorithmic extensions (such as ...)" list
    lives. Do not extract that sentence with [^.]* -- it terminates on the period inside "arxiv.org" and
    the assertion then passes vacuously.
    """
    intro = README.split("## Quick Start")[0]
    assert "spectrogram loss balancing" not in intro.lower(), (
        "time vs. spectrogram balancing is still listed among the novel extensions"
    )
    # ...but the feature is real and must stay documented; do not "fix" this by deleting it.
    assert "spec_weight" in README, "spec_weight is no longer documented at all"


def test_limitations_documents_level_dependent_floor():
    """Section 5: D1's floor is 10*log10(eps/(mean|y|+eps)), so a target near -58 dBFS has <3 dB of range."""
    limitations = README.split("## Limitations")[1].split("##")[0]
    assert "floor" in limitations.lower(), (
        "Limitations does not document the level-dependent dynamic-range collapse"
    )


def test_public_api_is_documented_in_readme():
    """Q24: dbrms is exported in __all__ but documented nowhere."""
    missing = [n for n in torch_l1_snr.__all__ if n not in README]
    assert not missing, f"exported but undocumented in README: {missing}"


# --------------------------------------------------------------------------------------
# C1, C6, C7 -- packaging hygiene
# --------------------------------------------------------------------------------------

def test_numpy_is_not_a_declared_dependency():
    """Q6: numpy>=1.21.0 is declared and documented but never imported -- forced on every consumer."""
    install_requires = SETUP_CFG.split("install_requires")[1].split("[")[0]
    assert "numpy" not in install_requires.lower(), "numpy is still in install_requires"
    pkg_sources = "".join(p.read_text() for p in (REPO / "torch_l1_snr").glob("*.py"))
    assert "import numpy" not in pkg_sources, "numpy is imported after all -- do not drop the dependency"


def test_logo_carries_no_authoring_identifiers():
    """Q18: the committed PNG leaks pdf:Author and Canva doc=/user=/brand= identifiers. It is public."""
    data = (REPO / "images" / "logo.png").read_bytes()
    text = []
    pos = 8
    while pos < len(data) - 8:
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        if ctype in (b"tEXt", b"iTXt", b"zTXt", b"eXIf"):
            text.append(data[pos + 8:pos + 8 + length].decode("latin-1"))
        pos += 12 + length
    blob = " ".join(text)
    for leaked in ["pdf:Author", "Canva", "xmp:CreatorTool", "user=", "brand="]:
        assert leaked not in blob, f"logo.png still carries {leaked!r}"


def test_py_typed_is_present_and_packaged():
    """Q10: without py.typed, PEP 561 says no type hint in this package reaches a downstream checker.

    Presence alone is not enough -- the marker has to be declared as package_data or it is absent from the
    built wheel, which is where it actually matters.
    """
    assert (REPO / "torch_l1_snr" / "py.typed").exists(), "py.typed marker file is missing"
    assert "py.typed" in SETUP_CFG, "py.typed is not declared as package_data, so it will not ship"


def test_copyright_identifier_is_consistent():
    """Q17: LICENSE says 'Christopher Landschoot', the source header says 'crlandsc'."""
    holder = "Christopher Landschoot"
    assert holder in LICENSE, "LICENSE does not name the expected copyright holder"
    assert holder in SOURCE[:600], f"source header does not use the same identifier as LICENSE ({holder!r})"
    assert holder in SETUP_CFG, "setup.cfg does not use the same identifier as LICENSE"
