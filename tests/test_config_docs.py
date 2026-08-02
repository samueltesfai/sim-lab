from pathlib import Path

from generate_config_skeleton import (
    RULES_BEGIN_MARKER,
    RULES_END_MARKER,
    SKELETON_BEGIN_MARKER,
    SKELETON_END_MARKER,
    render_skeleton,
    render_validation_rules,
)

_CONFIG_MD = Path(__file__).resolve().parent.parent / "docs" / "config.md"


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
    the doc. Run `uv run python scripts/generate_config_skeleton.py` to fix.
    """
    assert _committed_skeleton() == render_skeleton(), (
        "docs/config.md's Config Structure block does not match what "
        "scripts/generate_config_skeleton.py produces from config_schema.py. "
        "Run `uv run python scripts/generate_config_skeleton.py` to update the doc."
    )


def test_config_md_validation_rules_matches_generated_output():
    """docs/config.md's Validation Rules block must match what
    scripts/generate_config_skeleton.py produces from config_schema.py --
    if this fails, a Field constraint was added/changed without regenerating
    the doc. Run `uv run python scripts/generate_config_skeleton.py` to fix.
    """
    assert _committed_validation_rules() == render_validation_rules(), (
        "docs/config.md's Validation Rules block does not match what "
        "scripts/generate_config_skeleton.py produces from config_schema.py. "
        "Run `uv run python scripts/generate_config_skeleton.py` to update the doc."
    )
