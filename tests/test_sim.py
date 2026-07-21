def test_sim_facade_exports_public_api():
    """simlab.sim is a compatibility facade; guard its public import surface."""
    from simlab.sim import (
        Agent,
        AgentUpdateTrace,
        World,
        Action,
        ActionTrace,
        ActionType,
        Memory,
        MemoryType,
        ObservationEvent,
        Snapshot,
        clamp,
    )

    assert Agent is not None
    assert AgentUpdateTrace is not None
    assert World is not None
    assert Action is not None
    assert ActionTrace is not None
    assert ActionType is not None
    assert Memory is not None
    assert MemoryType is not None
    assert ObservationEvent is not None
    assert Snapshot is not None
    assert clamp(1.2) == 1.0
