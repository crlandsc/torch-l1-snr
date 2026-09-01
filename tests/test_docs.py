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
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch_l1_snr
from torch_l1_snr import (
    L1SNRLoss,
    L2SNRLoss,
    L1SNRDBLoss,
    STFTL1SNRDBLoss,
    MultiL1SNRDBLoss,
)

README = (REPO / "README.md").read_text()
CHANGELOG = (REPO / "CHANGELOG.md").read_text()
SOURCE = (REPO / "torch_l1_snr" / "l1snr.py").read_text()
DESIGN_NOTES = (REPO / "docs" / "design_notes.md").read_text()
LICENSE = (REPO / "LICENSE").read_text()
SETUP_CFG = (REPO / "setup.cfg").read_text()

ALL_CLASSES = [L1SNRLoss, L2SNRLoss, L1SNRDBLoss, STFTL1SNRDBLoss, MultiL1SNRDBLoss]

# Every version published to PyPI, established from the PyPI JSON API during the audit.
PUBLISHED_VERSIONS = ["0.0.1", "0.0.2", "0.0.3", "0.0.4", "0.0.5",
                      "0.1.0", "0.1.1", "0.1.2", "0.1.3"]


# The README is known to contain this many runnable python examples. Pinned so the gates below cannot
# silently evaporate: they are parametrized over the discovered blocks, so if the fence syntax changed and
# the regex stopped matching, pytest would collect zero cases and report success rather than failure.
EXPECTED_README_EXAMPLES = 6


def readme_python_blocks():
    """Fenced ```python blocks that are self-contained programs.

    A block without an import is a fragment illustrating one call (README's mps_cpu_fallback snippet), not a
    runnable example. The audit excluded it on the same grounds.
    """
    blocks = re.findall(r"```python\n(.*?)```", README, re.DOTALL)
    return [b for b in blocks if "import" in b]


def test_readme_example_discovery_still_works():
    """Guards the gates that depend on discovery, which fail open rather than closed.

    Found by an adversarial reviewer: renaming every ```python fence to ```py made six example gates
    disappear while CI stayed green.
    """
    found = len(readme_python_blocks())
    assert found == EXPECTED_README_EXAMPLES, (
        f"README example discovery found {found} runnable blocks, expected {EXPECTED_README_EXAMPLES}. "
        "Either an example was added or removed (update the constant, deliberately), or the fence syntax "
        "changed and the example gates are no longer running at all.")


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
    # The WHOLE source, not a window around the first mention. A reviewer found three surviving
    # torch.abs attributions at source offsets 25170, 44197 and 44601 -- including the RuntimeWarning
    # text a user actually sees -- while this gate inspected only chars 21611-22511 and passed.
    passages = {"CHANGELOG 0.1.3": changelog_013, "README": readme_mps, "l1snr.py": SOURCE}
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
    # `torch.abs()` with empty parens is how prose refers to the function; `torch.abs(expr)` is a
    # legitimate call. Scanning the whole source for the bare name matched the implementation's own
    # arithmetic, which is why this looks for the prose form.
    assert "torch.abs()" not in text, f"{where} still attributes the bug to torch.abs"
    assert "bug in torch.abs" not in text, f"{where} still attributes the bug to torch.abs"


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


def test_design_notes_document_ref_level_and_its_derivation():
    """A14 introduced two parameters whose defaults rest on measurements; both must be explained.

    Moved out of the README (kept concise there) into docs/design-notes.md, so the measured derivation is
    gated where it now lives."""
    assert "ref_level" in DESIGN_NOTES
    assert "spec_ref_level" in DESIGN_NOTES
    assert "0.19" in DESIGN_NOTES, "the measured STFT-to-time reference ratio is not stated"
    # the README still has to name ref_level, since a user sets it
    assert "ref_level" in README, "the README no longer mentions ref_level at all"


def test_design_notes_state_the_per_domain_difference():
    """M23: at l1_weight=0.5 the time domain moves ~25% toward L1 and the spectrogram ~45%, so the single
    knob means different things in the two halves of MultiL1SNRDBLoss. Now in docs/design-notes.md."""
    assert "different things in its two halves" in DESIGN_NOTES or "two halves" in DESIGN_NOTES, (
        "design-notes does not state that l1_weight's effect differs by domain")


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


# L2SNRLoss ships deliberately quiet: importable, with a full docstring, but referenced in no prose doc so
# it is not advertised as a feature while it is still experimental. This set locks that intent -- both that
# it stays out of the prose, and that it can only leave this set by a deliberate edit here.
QUIET_UNDOCUMENTED = {"L2SNRLoss"}


def test_public_api_is_documented_in_readme_except_the_quiet_ones():
    """Q24: dbrms was exported in __all__ but documented nowhere. Every public name must be in the README
    except the deliberately-quiet experimental ones, which must be ABSENT from every prose doc."""
    missing = [n for n in torch_l1_snr.__all__ if n not in README and n not in QUIET_UNDOCUMENTED]
    assert not missing, f"exported but undocumented in README: {missing}"
    for n in QUIET_UNDOCUMENTED:
        assert n not in README, f"{n} is meant to ship quiet but appears in the README"
        assert n not in DESIGN_NOTES, f"{n} is meant to ship quiet but appears in docs/design-notes.md"


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
    # The declaration is kept as an explicit statement of intent, but note it is not what makes the marker
    # ship: with a pyproject-declared build backend, setuptools defaults include_package_data to True, so
    # py.typed lands in the wheel with or without this stanza. Verified by removing it on a clean tree. What
    # actually guarantees delivery is the wheel-content check in CI, so do not read this line as causal.
    assert "py.typed" in SETUP_CFG, "the py.typed package_data declaration was removed"


def test_copyright_identifier_is_consistent():
    """Q17: LICENSE says 'Christopher Landschoot', the source header says 'crlandsc'."""
    holder = "Christopher Landschoot"
    assert holder in LICENSE, "LICENSE does not name the expected copyright holder"
    assert holder in SOURCE[:600], f"source header does not use the same identifier as LICENSE ({holder!r})"
    assert holder in SETUP_CFG, "setup.cfg does not use the same identifier as LICENSE"


def test_changelog_states_compatibility_per_loss_not_globally():
    """This claim has now been wrong twice, in opposite directions, so the gate is about *shape*.

    First it said "bit-identical values and gradients" at the default l1_weight, and gradients differ. The
    correction then said loss values are bit-identical -- and a second reviewer showed that is false too for
    L1SNRDBLoss with regularization and for both spectrogram classes, which move at float32 rounding. Only
    L1SNRLoss and L1SNRDBLoss(use_regularization=False) are bit-identical on both.

    So the requirement is that compatibility is stated *per loss class*, never as one global sentence, and
    that the text steers a validating user to compare loss values rather than gradients.
    """
    breaking = CHANGELOG.split("### BREAKING CHANGES")[1].split("### ")[0]
    for banned in ["bit-identical values and gradients",
                   "All four losses return **bit-identical loss values**",
                   "are unaffected numerically"]:
        assert banned not in breaking, (
            f"the CHANGELOG makes a global compatibility claim again ({banned!r}); compatibility differs "
            "by loss class and must be stated that way")
    # a per-class table: every class named, and the two that are bit-identical distinguished
    for cls in ["L1SNRLoss", "L1SNRDBLoss", "STFTL1SNRDBLoss", "MultiL1SNRDBLoss"]:
        assert cls in breaking, f"the compatibility table does not mention {cls}"
    assert "bit-identical" in breaking and "differs" in breaking, (
        "the table must distinguish the losses that are bit-identical from those that are not")
    # Checked as a requirement, not a phrase. Pinning the exact sentence is what made an earlier version of
    # this gate defend a false claim, and it broke again the moment the wording was improved.
    # Scoped to the paragraph that gives validation advice. An earlier version searched the whole BREAKING
    # section and compared positions, which was meaningless: "gradients" appears in the compatibility table
    # header hundreds of characters before any guidance.
    paras = [p for p in breaking.split("\n\n") if "validate the upgrade" in p or "diffing" in p]
    assert paras, "the CHANGELOG does not tell a user how to validate the upgrade"
    guidance = " ".join(paras).lower()
    assert "tolerance" in guidance or "rtol" in guidance, (
        "the validation guidance must name a tolerance; an exact comparison fails on every path")
    assert "loss values" in guidance and "gradient" in guidance, (
        "the guidance must mention both, so a user knows which to compare")
    assert guidance.index("loss values") < guidance.index("gradient"), (
        "the guidance should reach loss values before gradients, since gradients differ by ~5e-04 relative "
        "at realistic shapes even at 10% error and a strict comparison tells a user nothing")


def test_readme_dbrms_formula_matches_the_code():
    """The README documented `20 * log10(sqrt(mean(x**2) + eps) + eps)` after 0.2.0 removed the outer
    epsilon. A user computing levels by hand from the documented formula would be off by 8.7e-04 dB and
    would get -79.99913 for silence where the code gives exactly -80."""
    import ast as _ast
    src = _ast.get_source_segment(SOURCE, next(
        n for n in _ast.walk(_ast.parse(SOURCE))
        if isinstance(n, _ast.FunctionDef) and n.name == "dbrms")) or ""
    outer_eps_in_code = "log10(rms + eps)" in src.replace(" ", "").replace("torch.", "") or \
                        "rms+eps" in src.replace(" ", "")
    outer_eps_in_readme = "+ eps) + eps)" in README
    assert outer_eps_in_code == outer_eps_in_readme, (
        f"the README's dbrms formula and the implementation disagree about the outer epsilon "
        f"(code has it: {outer_eps_in_code}, README shows it: {outer_eps_in_readme})")


def test_check_finite_is_on_both_stft_classes_and_documented():
    """The CHANGELOG said STFTL1SNRDBLoss *and* MultiL1SNRDBLoss gain check_finite. Only the former did,
    so the documented performance advice was unusable on the class the Quick Start recommends."""
    import inspect as _inspect
    for cls in (STFTL1SNRDBLoss, MultiL1SNRDBLoss):
        assert "check_finite" in _inspect.signature(cls.__init__).parameters, (
            f"{cls.__name__} does not accept check_finite, but the CHANGELOG says it does")
    assert "check_finite" in README, (
        "check_finite is absent from the README, while the Numerical Stability bullet promises the "
        "behaviour it switches off")


def test_the_changelog_does_not_advertise_the_reverted_device_guard():
    """The device-move guard was reverted, but the Performance section still sold it as a shipped win.

    The revert and the CHANGELOG were one commit apart and nothing connected them, so the release notes
    promised an optimization the wheel does not contain -- and specifically the one that silently zeroed the
    spectral loss. This gate ties the prose to the code: the guard may be described only as something that
    did not ship. It deliberately does not ban the "0.02 ms" figure, which is honest in that context.
    """
    assert "guarded rather than unconditional" not in CHANGELOG, (
        "the CHANGELOG presents the per-call device guard as shipped; it was reverted in b534e73 because a "
        "stale cache made the spectral loss return exactly 0.0 after nn.Module.to()")
    if "device move" in CHANGELOG:
        did_not_ship = CHANGELOG.index("did not ship")
        assert CHANGELOG.index("device move") > did_not_ship, (
            "the device move is discussed before the did-not-ship list, which reads as a shipped change")


def test_the_changelog_top_entry_matches_the_version_and_only_it_may_be_unreleased():
    """The top CHANGELOG entry is the version being prepared, and only it may be `(unreleased)`.

    Catches version/changelog drift -- the CI tag gate checks __version__ against the tag. The top entry's
    version must equal __version__; it may still say `(unreleased)` in development (strip to a date before
    tagging, as 0.2.0 was). At most one `(unreleased)` marker exists and it is the top entry, so every entry
    below stays dated. A leftover `## 0.1.4` section would describe a version nobody can install, and 0.1.4's
    summary claim (four signatures identical to 0.1.3) must not survive into a later entry.
    """
    version = torch_l1_snr.__version__
    top = re.search(r"^## (\S+) \(([^)]+)\)", CHANGELOG, re.MULTILINE)
    assert top and top.group(1) == version, (
        f"the top changelog entry is {top and top.group(1)!r}, not __version__={version!r}")
    unreleased = re.findall(r"^## (\S+) \(unreleased\)", CHANGELOG, re.MULTILINE)
    assert unreleased in ([], [version]), (
        f"only the top entry may be unreleased; found {unreleased}")
    if top.group(2) != "unreleased":
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", top.group(2)), (
            f"the top entry is dated {top.group(2)!r}; expected an ISO date or '(unreleased)'")
    assert not re.search(r"^## 0\.1\.4", CHANGELOG, re.MULTILINE), (
        "there is a 0.1.4 section; that version was never published and 0.2.0 contains all of its content")
    assert "signature change in this release. All four constructor signatures are identical to 0.1.3" \
        not in CHANGELOG, (
        "0.1.4's summary claim survived into a later entry; it appends parameters and is not "
        "signature-identical to 0.1.3")


def test_the_changelog_states_positional_compatibility_as_a_guarantee():
    """Preserving 0.1.3's parameter order is a compatibility guarantee, not a breaking change.

    An earlier version listed it as BREAKING item 4 and described a mid-signature insertion that never
    happened -- the reordering was introduced by the "fix" itself. A user reading that would go looking for a
    migration they do not need, in the one section they are told to read before upgrading.
    """
    # Bounded by the NEXT "### " heading, not by "### Performance". Hard-coding the following heading
    # meant that inserting any section between the two silently widened the slice: the 0.2.0 "### Added"
    # section landed there and carried its own "keep their meaning", so deleting the phrase from the real
    # BREAKING section would no longer have failed this gate.
    breaking = CHANGELOG.split("### BREAKING CHANGES")[1].split("### ")[0]
    assert "keep their meaning" in breaking, (
        "the BREAKING section must state that positional calls written against 0.1.3 still mean the same "
        "thing; it is the first thing a cautious upgrader checks")
    assert "briefly inserted" not in CHANGELOG, (
        "the CHANGELOG describes a mid-signature insertion in 0.1.4 that never happened; 0.1.4's signatures "
        "are identical to 0.1.3's")


def test_the_design_notes_document_the_three_inherent_properties():
    """Three behaviours a user can hit that no amount of reading the previous text would predict.

    Each is gated behaviourally in test_edge_cases as well; this checks the user-facing half exists, because a
    property that is true, tested, and undocumented still surprises the person it happens to. Requirements
    rather than sentences. Moved from the README's Limitations to docs/design-notes.md, which the README's
    Limitations section links to; the numbers are gated where they now live.
    """
    limits = DESIGN_NOTES
    for topic, needles in [
        # The mechanism, not just the fact. The first version of this entry said the eps floor *caused* the
        # inversion; eps in fact caps it, and there is a second cause (bin dilution) the entry omitted.
        ("the non-monotone spectrogram loss", ["monotone", "-23.6", "DC offset", "limits", "dilutes"]),
        ("the float32 dbrms overflow", ["overflow", "2.5e16", "float64"]),
        ("no regularizer gradient at silence", ["digital silence", "exactly zero"]),
    ]:
        missing = [n for n in needles if n not in limits]
        assert not missing, f"the Limitations section does not document {topic}; missing {missing}"
    """`nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0)` zeroes NaN and clamps infinities to full scale.

    Two docstrings, the RuntimeWarning a user reads, and the CHANGELOG all described this as "replaced with
    zeros". Full-scale audio is the opposite of silence, so someone debugging a level-matching regularizer
    after an Inf appeared would be looking for the wrong artifact entirely.
    """
    for where, doc in [("README", README), ("CHANGELOG", CHANGELOG), ("l1snr.py", SOURCE)]:
        for claim in ("replaced with zeros", "replacing it with zeros", "them with zeros"):
            assert claim not in doc, (
                f"{where} says non-finite input is {claim!r}; only NaN becomes zero, while +/-Inf becomes "
                "+/-1.0, which is full scale")
    assert "posinf=1.0" in SOURCE, "this gate assumes the source clamps infinities to full scale"
    for where, doc in [("CHANGELOG", CHANGELOG), ("l1snr.py", SOURCE)]:
        assert "1.0" in doc and ("full scale" in doc or "full-scale" in doc), (
            f"{where} must say what Inf actually becomes, not only what NaN becomes")


@pytest.mark.no_forward  # inspects module-level lists only
def test_every_per_class_list_in_the_suite_covers_the_public_classes():
    """The suite keeps four separate lists of "all the loss classes", and adding a class narrows them.

    This has now happened three times, each time silently:

    1. `test_edge_cases.py` was added after `mutation_gate.TARGETS` was written, so its detection rate sat
       at 28% while the gate reported 100% of the file it did cover.
    2. `_forward_counter.CLASSES` was never updated with `L2SNRLoss`, so a dishonest `no_forward` marker on
       the new class was invisible to the marker audit.
    3. `test_edge_cases.ALL_CLASSES` was never updated either, so three mutations to `L2SNRLoss` -- dropping
       `* self.weight` and dropping the validators from both `__init__` and `forward` -- survived the entire
       numerical suite.

    Fixing instances did not stop it, and fix 2 was itself ungated: restoring the old four-name `CLASSES`
    leaves everything green and reopens that hole exactly. So this gate covers the family. A public loss
    class must appear in every list, or be declared in that module's exclusion mapping with a reason, which
    turns a silent narrowing into a decision someone has to write down.
    """
    import _forward_counter
    import _mutation_stub
    import test_edge_cases

    public = {n for n in torch_l1_snr.__all__ if isinstance(getattr(torch_l1_snr, n), type)}
    assert len(public) >= 5, f"expected at least 5 public loss classes, found {sorted(public)}"

    registries = {
        "_forward_counter.CLASSES": (set(_forward_counter.CLASSES), {}),
        "_mutation_stub.CLASSES": (set(_mutation_stub.CLASSES), {}),
        "test_docs.ALL_CLASSES": ({c.__name__ for c in ALL_CLASSES}, {}),
        "test_edge_cases.ALL_CLASSES": (
            {c.__name__ for c in test_edge_cases.ALL_CLASSES},
            test_edge_cases.EXCLUDED_FROM_CROSS_CLASS_GATES,
        ),
    }

    for where, (covered, excluded) in registries.items():
        missing = public - covered - set(excluded)
        assert not missing, (
            f"{where} does not cover {sorted(missing)}. Every gate driven by this list silently stops "
            "covering that class. Add it, or declare it in that module's exclusion mapping with a reason.")
        stale = set(excluded) - public
        assert not stale, (
            f"{where}'s exclusion mapping names {sorted(stale)}, which is no longer a public class; "
            "a stale exclusion hides a real gap behind a plausible-looking reason")
        for name, reason in excluded.items():
            assert isinstance(reason, str) and len(reason) > 40, (
                f"{where} excludes {name} without a substantive reason; the reason is the whole point")
        assert "dbrms" not in covered, f"{where} lists dbrms, which is a function, not a loss class"


def test_no_superseded_figures_survive_in_the_documentation():
    """Specific numbers that measurement retired. Each was published, then shown wrong.

    They are checked as strings because that is what a reader sees. The list is deliberately concrete:
    a general "are the numbers right" gate is not possible, but a regression to a *known-wrong* number is.
    """
    retired = {
        "4% of the forward": "the window fold measures 14%, not 4%; 4% came from an isolated micro-benchmark",
        "6.2e-10 to 5.2e-10": "two careful measurements disagreed by 10x; no absolute figure is published",
        "9.6e-10": "same",
        "0.3686": "the retracted spec_weight recommendation",
        "74.81": "the gradient-ratio figure that did not reproduce even against the old code",
    }
    for text, why in retired.items():
        for where, doc in [("README", README), ("CHANGELOG", CHANGELOG), ("l1snr.py", SOURCE)]:
            assert text not in doc, f"{where} still contains the retired figure {text!r}: {why}"


def test_changelog_compatibility_bounds_are_not_exceeded():
    """The bounds in the compatibility table are a promise a user validates against, so measure them.

    They were first published as 1.1e-07 and 2.2e-07 and are actually up to 4e-05, exceeded worst on a
    silent estimate against a quiet target -- the collapse case the regularizer exists to detect, and the
    one a tester is most likely to construct. Driver is the dbrms epsilon removal, amplified by the
    regularizer's |L_pred - L_true| term when a level sits near the -80 dB floor, so it shows up in float64
    too and is not a rounding effect.
    """
    import os as _os
    import re as _re
    import subprocess as _sub
    import tempfile as _tmp
    import importlib.util as _il
    import torch as _t

    stated = {}
    for line in CHANGELOG.split("### BREAKING CHANGES")[1].split("### ")[0].splitlines():
        if line.startswith("|") and "differs, up to" in line:
            m = _re.search(r"up to \*{0,2}([0-9.e-]+)\*{0,2} relative", line)
            if m:
                for cls in ["L1SNRLoss", "L1SNRDBLoss", "STFTL1SNRDBLoss", "MultiL1SNRDBLoss"]:
                    if cls in line:
                        stated.setdefault(cls, float(m.group(1)))
    assert stated, "no compatibility bounds found in the CHANGELOG table"

    # check=False and skip: a shallow clone or a tarball has no 938b4f1, and erroring there reports a
    # missing baseline as a failed compatibility bound. CI now fetches full history so this really runs;
    # the skip is for anywhere that cannot. The sibling gate in test_edge_cases.py does the same.
    _show = _sub.run(["git", "-C", str(REPO), "show", "938b4f1:torch_l1_snr/l1snr.py"],
                     capture_output=True, text=True)
    if _show.returncode != 0:
        pytest.skip("cannot reach the 938b4f1 baseline from git; compatibility bounds unverifiable here")
    src = _show.stdout
    # TemporaryDirectory, not mkdtemp: tempfile.gettempdir() falls back to os.getcwd() when every
    # standard candidate fails its writability probe, which happens under a sandboxed or locked-down
    # /tmp. mkdtemp then leaves a directory in the repo root on every run. Where it lands is the
    # environment's business; cleaning it up is ours.
    with _tmp.TemporaryDirectory() as _d:
        path = _os.path.join(_d, "base.py")
        open(path, "w").write(src)
        spec = _il.spec_from_file_location("base_bounds", path)
        base = _il.module_from_spec(spec)
        spec.loader.exec_module(base)

    import torch_l1_snr.l1snr as cur
    worst = {}
    for cls in stated:
        for dt in (_t.float32, _t.float64):
            for lvl in (1.0, 0.05, 0.001):
                for kind in ("noisy", "silent_est"):
                    g = _t.Generator().manual_seed(0)
                    a = (_t.randn(2, 2, 4096, generator=g) * lvl).to(dt)
                    e = (torch.zeros_like(a) if kind == "silent_est"
                         else a + (_t.randn(2, 2, 4096, generator=g) * lvl * 0.1).to(dt))
                    vo = getattr(base, cls)(name="t")(e, a)
                    vn = getattr(cur, cls)(name="t")(e, a)
                    if vo.item() == 0:
                        continue
                    r = abs((vn.item() - vo.item()) / vo.item())
                    worst[cls] = max(worst.get(cls, 0.0), r)
    for cls, bound in stated.items():
        assert worst.get(cls, 0.0) <= bound * 1.5, (
            f"the CHANGELOG says {cls} differs by up to {bound:.1e} relative; measured "
            f"{worst[cls]:.2e}, which is {worst[cls] / bound:.0f}x the stated bound")


# --------------------------------------------------------------------------------------
# Working-tree hygiene -- the suite must not litter the repo
# --------------------------------------------------------------------------------------

@pytest.mark.no_forward  # inspects process side effects, not any loss forward()
def test_running_the_suite_leaves_no_untracked_files_behind():
    """Both leaks came from trusting tempfile.gettempdir() to be outside the repo.

    It is not, necessarily: gettempdir() probes TMPDIR, TEMP, TMP, /tmp, /var/tmp and /usr/tmp for
    writability and falls back to os.getcwd() when all of them fail, which is what a sandboxed or
    locked-down /tmp produces. tempfile.mktemp() and mkdtemp() then wrote a JSON report and a
    directory of copied source into the repo root on every run, neither of them gitignored, so
    `git status` came back dirty after running the tests. CI hit the mktemp one on every push.

    Asserting on git's own view rather than globbing for `tmp*`, so a differently-named leak still
    fails this.
    """
    import os as _os
    import subprocess as _sub

    def untracked():
        out = _sub.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all"],
                       capture_output=True, text=True, check=True).stdout
        return {ln[3:] for ln in out.splitlines() if ln.startswith("??")}

    before = untracked()

    # The two operations that leaked. Scoped to one test and one script run rather than the whole
    # suite, so this gate stays cheap enough to live in it.
    env = {**_os.environ, "PYTHONPATH": str(REPO / "tests")}
    _sub.run([sys.executable, "-m", "pytest", str(REPO / "tests" / "test_docs.py"),
              "-k", "changelog_compatibility_bounds_are_not_exceeded", "-q", "--no-header"],
             cwd=REPO, capture_output=True, text=True, env=env)
    _sub.run([sys.executable, str(REPO / "tests" / "mutation_gate.py"), "--audit-markers"],
             cwd=REPO, capture_output=True, text=True, env=env)

    leaked = untracked() - before
    assert not leaked, (
        "running the test suite left untracked files in the repo: "
        f"{sorted(leaked)}. Use tempfile.TemporaryDirectory (which cleans up wherever it lands) "
        "rather than mkdtemp/mktemp.")


# --------------------------------------------------------------------------------------
# 0.2.1 -- grad_scale is documented, and its magnitude claim is backed by measurement
# --------------------------------------------------------------------------------------

def _gh_slug(text):
    """GitHub's header-anchor slug: lowercase, drop punctuation (keep word chars and spaces),
    spaces to hyphens. `grad_scale` keeps its underscore; backticks are dropped."""
    s = re.sub(r"[^\w\s-]", "", text.strip().lower())
    return re.sub(r"\s+", "-", s)


def test_grad_scale_is_surfaced_in_the_readme_and_the_design_notes():
    """0.2.1 added grad_scale as an opt-in knob. The README must surface it with the clipping guidance
    that is its whole reason to exist, and must state it does not change the loss value (else a reader
    fears it perturbs training). The README's deep link into the design notes must resolve to a real
    header, and the design notes must carry the value-preserving and optimizer-specific claims.
    """
    assert "grad_scale" in README, "the README never mentions grad_scale"
    assert "clip" in README.lower(), "the README does not connect grad_scale to gradient clipping"
    assert re.search(r"without changing the loss value|does not change the loss", README), (
        "the README does not state grad_scale leaves the loss value unchanged")
    # the README deep-links to a design-notes header; that anchor must exist
    m = re.search(r"docs/design_notes\.md#([\w-]+)", README)
    assert m, "the README does not link into docs/design_notes.md for the gradient discussion"
    anchor = m.group(1)
    headers = {_gh_slug(h) for h in re.findall(r"^#+\s+(.*)$", DESIGN_NOTES, re.MULTILINE)}
    assert anchor in headers, (
        f"README links to #{anchor} but no design-notes header slugifies to it; headers are {sorted(headers)}")
    assert "bit-exact" in DESIGN_NOTES and "grad_scale" in DESIGN_NOTES, (
        "the design notes do not state the loss value is bit-exact regardless of grad_scale")
    assert "learning-rate rescale" in DESIGN_NOTES, (
        "the design notes do not carry the SGD learning-rate-rescale caveat")


def test_the_hundreds_times_magnitude_claim_is_backed_by_measurement():
    """The docs claim these gradients are 'a few hundred times larger than a plain L1 loss' and that the
    ratio 'grows as the estimate converges'. Both are measurable, so measure them rather than trust the
    prose. On the documented [4, 2, 44100] @ 0.05 shape: L1SNR's input gradient norm over plain L1's must
    be in the hundreds at 10% relative error, and must rise monotonically as the error shrinks. A change
    that quietly flattened the gradient would make the doc false and trip this.
    """
    import torch.nn.functional as F

    def ratio(rel_err):
        torch.manual_seed(1)
        act = 0.05 * torch.randn(4, 2, 44100)
        noise = 0.05 * torch.randn_like(act)
        est_s = (act + rel_err * noise).detach().clone().requires_grad_(True)
        L1SNRLoss("m")(est_s, act).backward()
        gsnr = est_s.grad.norm().item()
        est_l = (act + rel_err * noise).detach().clone().requires_grad_(True)
        F.l1_loss(est_l, act).backward()
        return gsnr / est_l.grad.norm().item()

    r50, r10, r1 = ratio(0.5), ratio(0.1), ratio(0.01)
    assert 200 < r10, f"L1SNR/L1 gradient ratio at 10% error is {r10:.0f}x, not the documented hundreds"
    assert r50 < r10 < r1, (
        f"ratio does not grow as the estimate converges: 50%={r50:.0f}x 10%={r10:.0f}x 1%={r1:.0f}x")
