"""Runtime guard ensuring every field of a settings object a constructor
receives is actually read somewhere in ``__init__``.

Passing a validated settings object straight into ``Agent``/``World``
(instead of translating it into a fixed set of flat kwargs first) removes a
whole layer of code, but it also removes a safety net that layer had almost
by accident: a settings field nobody wired up used to surface as a loud
``TypeError`` (an unrecognized kwarg); now it just sits on the object,
unread, with no error anywhere. ``FieldTracker`` restores that guarantee
directly -- wrap the settings object, use it exactly as you would the real
thing, then call ``assert_fully_consumed()`` once construction is done.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from pydantic import BaseModel


@cache
def _leaf_field_paths(model_cls: type[BaseModel], prefix: str = "") -> frozenset[str]:
    """Every dotted leaf field path on ``model_cls``, recursing into nested
    settings sections. A ``dict[...]``-typed field (e.g. ``action_preference``)
    is a leaf in its own right -- it's consumed as a whole dict, not
    field-by-field.
    """
    paths: set[str] = set()
    for name, field in model_cls.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            paths |= _leaf_field_paths(annotation, path)
        else:
            paths.add(path)
    return frozenset(paths)


class FieldTracker:
    """Wraps a pydantic settings instance; every leaf attribute read through
    it (including through nested sections) is recorded in a shared
    ``_accessed`` set. Only call ``assert_fully_consumed()`` on the root
    tracker returned by the initial ``FieldTracker(settings, ExpectedCls)``
    call -- a nested tracker (e.g. one returned for a ``.observation``
    access) shares that set but doesn't know the full top-level schema to
    check it against.

    ``expected_cls`` is checked against separately from ``type(model)``
    because a caller may legitimately pass a *subclass* carrying extra
    fields the constructor was never meant to consume -- e.g. ``Agent``
    declares it accepts ``AgentSettings``, but ``world_from_config`` passes
    an ``AgentProfile`` (which adds ``name``/``count``, consumed elsewhere,
    not by ``Agent``). Checking completeness against the object's runtime
    type would flag those as false positives; checking against the
    constructor's own declared type doesn't.
    """

    def __init__(
        self,
        model: BaseModel,
        expected_cls: type[BaseModel] | None = None,
        _prefix: str = "",
        _accessed: set[str] | None = None,
    ):
        self._model = model
        self._expected_cls = expected_cls if expected_cls is not None else type(model)
        self._prefix = _prefix
        self._accessed = _accessed if _accessed is not None else set()

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._model, name)
        path = f"{self._prefix}.{name}" if self._prefix else name
        if isinstance(value, BaseModel):
            return FieldTracker(value, None, path, self._accessed)
        self._accessed.add(path)
        return value

    def assert_fully_consumed(self) -> None:
        missing = _leaf_field_paths(self._expected_cls) - self._accessed
        if missing:
            raise RuntimeError(
                f"{self._expected_cls.__name__} field(s) never read during "
                f"construction: {sorted(missing)}. A field exists on the "
                f"config schema with no behavior wired up for it."
            )
