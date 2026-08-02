"""Render docs/config.md's "Config Structure" YAML skeleton and "Validation
Rules" table directly from simlab.config_schema, so they can't silently
drift from the actual schema.

Run directly to regenerate docs/config.md in place:

    uv run python scripts/generate_config_skeleton.py

``render_skeleton()``/``render_validation_rules()`` are also imported by
tests/test_config_docs.py, which fails if the committed doc doesn't match
what this script would produce.
"""

from __future__ import annotations

import sys
import typing
from pathlib import Path

import annotated_types
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from simlab.config_schema import (
    AgentProfile,
    AgentSection,
    AgentSettings,
    WorldObservation,
    WorldSection,
)

SKELETON_BEGIN_MARKER = "<!-- BEGIN GENERATED CONFIG SKELETON -->"
SKELETON_END_MARKER = "<!-- END GENERATED CONFIG SKELETON -->"
RULES_BEGIN_MARKER = "<!-- BEGIN GENERATED VALIDATION RULES -->"
RULES_END_MARKER = "<!-- END GENERATED VALIDATION RULES -->"

# Backwards-compatible aliases (previously the only markers this script had).
BEGIN_MARKER = SKELETON_BEGIN_MARKER
END_MARKER = SKELETON_END_MARKER

_TOKENS: dict[type, str] = {
    int: "<int>",
    float: "<float>",
    bool: "<bool>",
    str: "<str>",
}


def _base_type(annotation: object) -> object:
    """Unwrap ``Annotated[float, Strict(...)]`` -> ``float`` (recursively,
    in case of doubly-nested ``Annotated``); a plain type like ``int`` (no
    args) passes through unchanged.
    """
    while True:
        args = typing.get_args(annotation)
        if not args:
            return annotation
        if isinstance(args[0], type):
            return args[0]
        annotation = args[0]


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


def _fmt_num(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def _constraint_text(field: FieldInfo, base_type: object) -> str | None:
    """Derive a human-readable rule from a field's ``Field(ge=, le=, gt=,
    lt=)`` metadata and required-ness. Returns ``None`` when there's nothing
    worth stating beyond the field's type (already shown in the skeleton).
    """
    ge = le = gt = lt = None
    for constraint in field.metadata:
        if isinstance(constraint, annotated_types.Ge):
            ge = constraint.ge
        elif isinstance(constraint, annotated_types.Le):
            le = constraint.le
        elif isinstance(constraint, annotated_types.Gt):
            gt = constraint.gt
        elif isinstance(constraint, annotated_types.Lt):
            lt = constraint.lt

    range_text: str | None = None
    if ge is not None and le is not None:
        range_text = f"in [{_fmt_num(ge)}, {_fmt_num(le)}]"
    elif ge is not None:
        range_text = f">= {_fmt_num(ge)}"
    elif le is not None:
        range_text = f"<= {_fmt_num(le)}"
    elif gt is not None:
        range_text = f"> {_fmt_num(gt)}"
    elif lt is not None:
        range_text = f"< {_fmt_num(lt)}"

    if range_text is not None:
        return f"required, {range_text}" if field.is_required() else range_text
    if base_type is bool:
        return "must be boolean"
    if field.is_required():
        return "required"
    return None


def _render_rules(model_cls: type[BaseModel], prefix: str) -> list[str]:
    """Recursively derive Validation Rules table rows for every scalar leaf
    field of ``model_cls``. ``dict[...]``-typed fields (``action_preference``,
    ``action_cost``, ``noise``, ``truths``) are skipped -- their range checks
    are enforced by custom ``@field_validator`` methods, not ``Field``
    metadata, so they aren't introspectable this way; those rules are
    hand-written in docs/config.md instead.
    """
    rows: list[str] = []
    for name, field in model_cls.model_fields.items():
        annotation = field.annotation
        base = _base_type(annotation)

        if isinstance(base, type) and issubclass(base, BaseModel):
            rows.extend(_render_rules(base, f"{prefix}{name}."))
            continue

        if typing.get_origin(annotation) is dict:
            continue

        text = _constraint_text(field, base)
        if text is not None:
            rows.append(f"| `{prefix}{name}` | {text} |")

    return rows


def render_validation_rules() -> str:
    """Render the Validation Rules markdown table (header included, no
    surrounding markers -- just the table)."""
    rows: list[str] = []
    rows.extend(_render_rules(WorldObservation, "world.observation."))

    min_len = next(
        c.min_length
        for c in AgentSection.model_fields["profiles"].metadata
        if isinstance(c, annotated_types.MinLen)
    )
    plural = "" if min_len == 1 else "s"
    rows.append(f"| `agent.profiles` | required, at least {min_len} profile{plural} |")
    rows.append(
        f"| `agent.profiles[*].name` | "
        f"{_constraint_text(AgentProfile.model_fields['name'], str)} |"
    )
    rows.append(
        f"| `agent.profiles[*].count` | "
        f"{_constraint_text(AgentProfile.model_fields['count'], int)} |"
    )

    rows.extend(_render_rules(AgentSettings, "*."))

    return "\n".join(["| Field | Rule |", "| --- | --- |", *rows])


def _replace_between_markers(text: str, begin: str, end: str, replacement: str) -> str:
    start = text.index(begin) + len(begin)
    stop = text.index(end)
    return f"{text[:start]}\n{replacement}\n{text[stop:]}"


def main() -> None:
    doc_path = Path(__file__).resolve().parent.parent / "docs" / "config.md"
    text = doc_path.read_text(encoding="utf-8")
    text = _replace_between_markers(
        text,
        SKELETON_BEGIN_MARKER,
        SKELETON_END_MARKER,
        f"```yaml\n{render_skeleton()}\n```",
    )
    text = _replace_between_markers(
        text, RULES_BEGIN_MARKER, RULES_END_MARKER, render_validation_rules()
    )
    doc_path.write_text(text, encoding="utf-8")
    print(f"Regenerated skeleton and validation rules in {doc_path}")


if __name__ == "__main__":
    sys.exit(main())
