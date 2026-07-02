"""Shared test helpers for reaching into model internals.

These helpers centralize the (intentionally private) mechanics that tests
occasionally need to exercise directly, such as seeding an agent's memory.
Keeping them in one place means a change to internal APIs only needs to be
reflected here, not in every test.
"""

from simlab.sim import Agent, MemoryType


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
