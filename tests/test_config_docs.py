from pathlib import Path

from generate_config_skeleton import (
    RULES_BEGIN_MARKER,
    RULES_END_MARKER,
    SKELETON_BEGIN_MARKER,
    SKELETON_END_MARKER,
    render_reference_defaults,
    render_skeleton,
    render_validation_rules,
)

_CONFIG_MD = Path(__file__).resolve().parent.parent / "docs" / "config.md"
_CONFIG_REFERENCE_MD = (
    Path(__file__).resolve().parent.parent / "docs" / "config_reference.md"
)


def _between_markers(text: str, begin: str, end: str) -> str:
    start = text.index(begin) + len(begin)
    stop = text.index(end)
    return text[start:stop].strip()


def _committed_skeleton() -> str:
    block = _between_markers(
        _CONFIG_MD.read_text(encoding="utf-8"),
        SKELETON_BEGIN_MARKER,
        SKELETON_END_MARKER,
    )
    assert block.startswith("```yaml") and block.endswith("```"), (
        "expected a fenced yaml block between the generated-skeleton markers"
    )
    return block.removeprefix("```yaml").removesuffix("```").strip()


def _committed_validation_rules() -> str:
    return _between_markers(
        _CONFIG_MD.read_text(encoding="utf-8"), RULES_BEGIN_MARKER, RULES_END_MARKER
    )


def test_config_md_skeleton_matches_generated_output():
    """docs/config.md's Config Structure block must match what
    scripts/generate_config_skeleton.py produces from config_schema.py --
    if this fails, a field was added/changed in code without regenerating
    the doc. Run `python scripts/generate_config_skeleton.py` to fix.
    """
    assert _committed_skeleton() == render_skeleton(), (
        "docs/config.md's Config Structure block does not match what "
        "scripts/generate_config_skeleton.py produces from config_schema.py. "
        "Run `python scripts/generate_config_skeleton.py` to update the doc."
    )


def test_config_md_validation_rules_matches_generated_output():
    """docs/config.md's Validation Rules block must match what
    scripts/generate_config_skeleton.py produces from config_schema.py --
    if this fails, a Field constraint was added/changed without regenerating
    the doc. Run `python scripts/generate_config_skeleton.py` to fix.
    """
    assert _committed_validation_rules() == render_validation_rules(), (
        "docs/config.md's Validation Rules block does not match what "
        "scripts/generate_config_skeleton.py produces from config_schema.py. "
        "Run `python scripts/generate_config_skeleton.py` to update the doc."
    )


def _committed_reference_defaults() -> dict[str, str]:
    # Not _between_markers: "<!-- /DEFAULT -->" repeats once per field, so
    # searching for it from the start of the document (as _between_markers
    # does) can find an earlier field's closing marker instead of this
    # one's -- search for `end` starting from `start` instead.
    text = _CONFIG_REFERENCE_MD.read_text(encoding="utf-8")
    committed: dict[str, str] = {}
    for name in render_reference_defaults():
        begin = f"<!-- DEFAULT {name} -->"
        end = "<!-- /DEFAULT -->"
        start = text.index(begin) + len(begin)
        stop = text.index(end, start)
        committed[name] = text[start:stop].strip()
    return committed


def test_config_reference_defaults_match_generated_output():
    """docs/config_reference.md's inline <!-- DEFAULT ... --> spans must
    match what scripts/generate_config_skeleton.py produces from
    config_schema.py -- if this fails, a default changed without
    regenerating the doc. Run `python scripts/generate_config_skeleton.py`
    to fix.
    """
    expected = {
        name: value.strip() for name, value in render_reference_defaults().items()
    }
    assert _committed_reference_defaults() == expected, (
        "docs/config_reference.md's default-value markers do not match what "
        "scripts/generate_config_skeleton.py produces from config_schema.py. "
        "Run `python scripts/generate_config_skeleton.py` to update the doc."
    )
