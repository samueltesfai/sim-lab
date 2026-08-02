from pathlib import Path

from generate_config_skeleton import BEGIN_MARKER, END_MARKER, render_skeleton

_CONFIG_MD = Path(__file__).resolve().parent.parent / "docs" / "config.md"


def _committed_skeleton() -> str:
    text = _CONFIG_MD.read_text(encoding="utf-8")
    start = text.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = text.index(END_MARKER)
    block = text[start:end].strip()
    assert block.startswith("```yaml") and block.endswith("```"), (
        "expected a fenced yaml block between the generated-skeleton markers"
    )
    return block.removeprefix("```yaml").removesuffix("```").strip()


def test_config_md_skeleton_matches_generated_output():
    """docs/config.md's Config Structure block must match what
    scripts/generate_config_skeleton.py produces from config_schema.py --
    if this fails, a field was added/changed in code without regenerating
    the doc. Run `uv run python scripts/generate_config_skeleton.py` to fix.
    """
    assert _committed_skeleton() == render_skeleton()
