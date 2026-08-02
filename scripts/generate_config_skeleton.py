"""Render docs/config.md's "Config Structure" YAML skeleton directly from
simlab.config_schema, so it can't silently drift from the actual schema.

Run directly to regenerate docs/config.md in place:

    uv run python scripts/generate_config_skeleton.py

``render_skeleton()`` is also imported by tests/test_config_docs.py, which
fails if the committed doc doesn't match what this script would produce.
"""

from __future__ import annotations

import sys
import typing
from pathlib import Path

from pydantic import BaseModel

from simlab.config_schema import AgentSettings, WorldSection

BEGIN_MARKER = "<!-- BEGIN GENERATED CONFIG SKELETON -->"
END_MARKER = "<!-- END GENERATED CONFIG SKELETON -->"

_TOKENS: dict[type, str] = {
    int: "<int>",
    float: "<float>",
    bool: "<bool>",
    str: "<str>",
}


def _base_type(annotation: object) -> object:
    """Unwrap ``Annotated[float, Strict(...)]`` -> ``float``; a plain type
    like ``int`` (no args) passes through unchanged.
    """
    args = typing.get_args(annotation)
    return args[0] if args and isinstance(args[0], type) else annotation


def _type_token(annotation: object) -> str:
    base = _base_type(annotation)
    return _TOKENS.get(base, "<?>")  # type: ignore[arg-type]


def _render_model(model_cls: type[BaseModel], indent: int) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    for name, field in model_cls.model_fields.items():
        annotation = field.annotation
        base = _base_type(annotation)

        if isinstance(base, type) and issubclass(base, BaseModel):
            lines.append(f"{pad}{name}:")
            lines.extend(_render_model(base, indent + 1))
            continue

        origin = typing.get_origin(annotation)
        if origin is dict:
            key_type, value_type = typing.get_args(annotation)
            value_token = _type_token(value_type)
            if typing.get_origin(key_type) is typing.Literal:
                keys = typing.get_args(key_type)
                inline = ", ".join(f"{k}: {value_token}" for k in keys)
            elif name == "truths":
                inline = f"<claim_id:int>: {value_token}, ..."
            else:
                inline = f"...: {value_token}"
            lines.append(f"{pad}{name}: {{ {inline} }}")
            continue

        lines.append(f"{pad}{name}: {_type_token(annotation)}")

    return lines


def render_skeleton() -> str:
    """Render the full ``world`` / ``agent`` YAML skeleton (no fence, no
    markers -- just the YAML body)."""
    lines: list[str] = ["world:"]
    lines.extend(_render_model(WorldSection, 1))
    lines.append("")
    lines.append("agent:")
    lines.append("  defaults:")
    lines.extend(_render_model(AgentSettings, 2))
    lines.append("  profiles:")
    lines.append("    - name: <str>")
    lines.append("      count: <int>")
    lines.append("      # any subset of the agent.defaults fields above, as overrides")
    return "\n".join(lines)


def _replace_between_markers(text: str, replacement: str) -> str:
    start = text.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = text.index(END_MARKER)
    return f"{text[:start]}\n```yaml\n{replacement}\n```\n{text[end:]}"


def main() -> None:
    doc_path = Path(__file__).resolve().parent.parent / "docs" / "config.md"
    text = doc_path.read_text(encoding="utf-8")
    doc_path.write_text(
        _replace_between_markers(text, render_skeleton()), encoding="utf-8"
    )
    print(f"Regenerated skeleton in {doc_path}")


if __name__ == "__main__":
    sys.exit(main())
