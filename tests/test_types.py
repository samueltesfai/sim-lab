import pytest

from simlab.types import Action, ActionType, clamp


def test_clamp():
    """Test the clamp utility function."""
    assert clamp(0.5) == 0.5
    assert clamp(-0.1) == 0.0
    assert clamp(1.1) == 1.0

    assert clamp(5, min_value=0, max_value=10) == 5
    assert clamp(-5, min_value=0, max_value=10) == 0
    assert clamp(15, min_value=0, max_value=10) == 10


def test_action_validation():
    """Test Action validation logic."""
    with pytest.raises(ValueError, match="'VERIFY' is not a valid ActionType"):
        Action("VERIFY", claim_id=0)

    with pytest.raises(ValueError, match="VERIFY action requires claim_id"):
        Action(ActionType.VERIFY)

    with pytest.raises(
        ValueError, match="COMMUNICATE action requires claim_id and target_agent_id"
    ):
        Action(ActionType.COMMUNICATE, claim_id=0)

    with pytest.raises(
        ValueError, match="COMMUNICATE action requires claim_id and target_agent_id"
    ):
        Action(ActionType.COMMUNICATE, target_agent_id=1)

    with pytest.raises(
        ValueError, match="BROADCAST action requires claim_id and no target_agent_id"
    ):
        Action(ActionType.BROADCAST)

    with pytest.raises(
        ValueError, match="BROADCAST action requires claim_id and no target_agent_id"
    ):
        Action(ActionType.BROADCAST, claim_id=0, target_agent_id=1)
