"""Shared pytest fixtures for reaching into model internals.

Centralizes the (intentionally private) mechanics that tests occasionally
need to exercise directly, such as seeding an agent's memory, so a change to
internal APIs only needs to be reflected here, not in every test.
"""

from functools import cache
from typing import Any

import pytest
from pydantic import BaseModel

from simlab.agent import Agent
from simlab.kernel_types import MemoryType


def seed_memory(
    agent: Agent,
    *,
    memory_type: MemoryType,
    evidence: float,
    tick: int = 0,
    claim_id: int = 0,
    source: int | None = None,
) -> None:
    """Seed a single memory onto an agent for cognition tests.

    Wraps the private ``Agent._add_memory`` so tests that verify internal
    belief/trust mechanics stay readable without encouraging external callers
    to create memories directly.
    """
    agent._add_memory(
        tick=tick,
        memory_type=memory_type,
        claim_id=claim_id,
        evidence=evidence,
        source=source,
    )


@pytest.fixture
def memory_seeder():
    return seed_memory


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
    ``_accessed`` set. Call ``assert_fully_consumed()`` once construction is
    done to raise if any field was never read -- only on the root tracker,
    since a nested one (e.g. from a ``.observation`` access) doesn't know
    the full top-level schema to check against.

    Test-only: verifies a constructor reads every field of the settings
    object it declares it accepts, without adding that check to the
    constructor itself -- Agent/World read plain attributes in production,
    at no per-construction cost.

    ``expected_cls`` is checked separately from ``type(model)`` because a
    caller may pass a subclass with extra fields the constructor was never
    meant to consume (e.g. ``Agent`` declares ``AgentSettings``, but
    ``world_from_config`` passes an ``AgentProfile``, whose ``name``/``count``
    are consumed elsewhere) -- checking against the runtime type would flag
    those as false positives.
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


@pytest.fixture
def field_tracker():
    return FieldTracker
