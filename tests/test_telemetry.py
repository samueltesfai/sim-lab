import csv
import json
import math

import pytest

from simlab.sim import Agent, World
from simlab.telemetry import Telemetry


def _build_world(n: int = 5) -> World:
    agents = [Agent(i, rng_seed=i) for i in range(n)]
    return World(agents=agents, truths={0: True}, rng_seed=1)


def test_record_populates_latest_and_history():
    world = _build_world(6)
    telemetry = Telemetry()

    snapshot = world.step()
    row = telemetry.record(snapshot, world)

    assert telemetry.latest == row
    assert telemetry.history == [row]
    assert row.tick == snapshot.tick

    assert row.num_observations == len(snapshot.observed_ids)
    assert row.num_verifications == len(snapshot.verified_ids)
    assert row.num_communicate_edges == len(snapshot.communicate_edges)
    assert row.num_broadcast_edges == len(snapshot.broadcast_edges)
    assert row.num_agent_updates == snapshot.n_agent_updates


def test_record_computes_global_belief_metrics():
    """Test that Telemetry computes global belief metrics across all beliefs."""
    world = _build_world(4)
    telemetry = Telemetry()
    snapshot = world.step()
    row = telemetry.record(snapshot, world)

    # Flatten all beliefs from snapshot
    vals = []
    for claim_beliefs in snapshot.agent_beliefs.values():
        vals.extend(claim_beliefs.values())

    n = len(vals)
    mean = sum(vals) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in vals) / n if n else 0.0
    std = math.sqrt(var)

    assert row.belief_mean == pytest.approx(mean)
    assert row.belief_std == pytest.approx(std)
    assert row.belief_min == pytest.approx(min(vals))
    assert row.belief_max == pytest.approx(max(vals))


def test_first_telemetry_row_has_zero_deltas():
    """Test that first telemetry row has zero deltas (no previous state)."""
    world = _build_world(3)
    telemetry = Telemetry()
    snapshot = world.step()
    row = telemetry.record(snapshot, world)

    assert row.mean_abs_delta == 0.0
    assert row.max_abs_delta == 0.0


def test_second_telemetry_row_computes_deltas():
    """Test that second telemetry row computes nonzero deltas from prior belief state."""
    world = _build_world(3)
    telemetry = Telemetry()

    # First step - deltas should be zero
    s0 = world.step()
    r0 = telemetry.record(s0, world)
    assert r0.mean_abs_delta == 0.0
    assert r0.max_abs_delta == 0.0

    # Second step - deltas should be computed
    s1 = world.step()
    r1 = telemetry.record(s1, world)

    # Compute expected deltas manually
    deltas = []
    for agent_id, claim_beliefs in s1.agent_beliefs.items():
        if agent_id in s0.agent_beliefs:
            prev_claims = s0.agent_beliefs[agent_id]
            for claim_id, belief_value in claim_beliefs.items():
                prev_value = prev_claims[claim_id]
                deltas.append(abs(belief_value - prev_value))

    if deltas:
        expected_mean = sum(deltas) / len(deltas)
        expected_max = max(deltas)
        assert r1.mean_abs_delta == pytest.approx(expected_mean)
        assert r1.max_abs_delta == pytest.approx(expected_max)
    else:
        assert r1.mean_abs_delta == 0.0
        assert r1.max_abs_delta == 0.0


def test_max_history_trims_oldest():
    world = _build_world(3)
    telemetry = Telemetry(max_history=2)

    s0 = world.step()
    r0 = telemetry.record(s0, world)
    s1 = world.step()
    r1 = telemetry.record(s1, world)
    s2 = world.step()
    r2 = telemetry.record(s2, world)

    assert telemetry.latest == r2
    assert telemetry.history == [r1, r2]
    assert r0 not in telemetry.history


def test_export_jsonl_and_csv(tmp_path):
    world = _build_world(3)
    telemetry = Telemetry()

    telemetry.record(world.step(), world, step_runtime_ms=1.23)
    telemetry.record(world.step(), world, step_runtime_ms=None)

    jsonl_path = tmp_path / "telemetry.jsonl"
    csv_path = tmp_path / "telemetry.csv"

    telemetry.export_jsonl(str(jsonl_path))
    telemetry.export_csv(str(csv_path))

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    objs = [json.loads(line) for line in lines]
    assert objs[0]["tick"] == telemetry.history[0].tick
    assert "step_runtime_ms" in objs[0]
    assert "mean_abs_error_to_truth" in objs[0]
    assert "max_abs_error_to_truth" in objs[0]
    assert "fraction_truth_aligned" in objs[0]
    # claim_id should not be in the exported data
    assert "claim_id" not in objs[0]

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["tick"] == str(telemetry.history[0].tick)
    assert "mean_abs_error_to_truth" in rows[0]
    assert "max_abs_error_to_truth" in rows[0]
    assert "fraction_truth_aligned" in rows[0]
    # claim_id should not be in the CSV header
    assert "claim_id" not in rows[0]


def test_telemetry_records_from_repeated_steps():
    """Test that telemetry can record rows from repeated World.step() calls."""
    world = _build_world(3)
    telemetry = Telemetry()

    # Record multiple steps
    for _ in range(5):
        snapshot = world.step()
        telemetry.record(snapshot, world)

    # Should have 5 rows in history
    assert len(telemetry.history) == 5
    assert telemetry.latest is not None

    # Ticks should be sequential
    ticks = [row.tick for row in telemetry.history]
    assert ticks == list(range(5))


def test_format_telemetry_row_includes_core_fields():
    """Test that formatted telemetry output includes core fields."""
    world = _build_world(3)
    telemetry = Telemetry()
    snapshot = world.step()
    row = telemetry.record(snapshot, world, step_runtime_ms=1.5)

    formatted = row.format_cli()

    # Check that core fields are present in the formatted string
    assert "Tick" in formatted
    assert f"mean={row.belief_mean:.3f}" in formatted
    assert f"std={row.belief_std:.3f}" in formatted
    assert f"min={row.belief_min:.3f}" in formatted
    assert f"max={row.belief_max:.3f}" in formatted
    assert f"Δabs_mean={row.mean_abs_delta:.4f}" in formatted
    assert f"Δmax={row.max_abs_delta:.4f}" in formatted
    assert f"com={row.num_communicate_edges}" in formatted
    assert f"bcast={row.num_broadcast_edges}" in formatted
    assert f"obs={row.num_observations}" in formatted
    assert f"ver={row.num_verifications}" in formatted
    assert f"updates={row.num_agent_updates}" in formatted
    assert f"runtime={row.step_runtime_ms:.3f}ms" in formatted


def test_format_telemetry_row_without_runtime():
    """Test that formatted telemetry output works without runtime."""
    world = _build_world(3)
    telemetry = Telemetry()
    snapshot = world.step()
    row = telemetry.record(snapshot, world, step_runtime_ms=None)

    formatted = row.format_cli()

    # Should not include runtime
    assert "runtime" not in formatted
    # But should include other fields
    assert "Tick" in formatted


def test_record_initial_creates_baseline_row():
    """Test that record_initial creates a baseline telemetry row with zero events."""
    world = _build_world(3)
    telemetry = Telemetry()

    row = telemetry.record_initial(world)

    # Default tick is -1
    assert row.tick == -1

    # All event counts should be zero
    assert row.num_observations == 0
    assert row.num_verifications == 0
    assert row.num_communicate_edges == 0
    assert row.num_broadcast_edges == 0
    assert row.num_agent_updates == 0

    # Deltas should be zero (no prior state)
    assert row.mean_abs_delta == 0.0
    assert row.max_abs_delta == 0.0

    # Truth alignment should be computed
    assert row.mean_abs_error_to_truth >= 0.0
    assert row.max_abs_error_to_truth >= 0.0
    assert 0.0 <= row.fraction_truth_aligned <= 1.0

    # Runtime should be None
    assert row.step_runtime_ms is None

    # Should be in history and latest
    assert telemetry.latest == row
    assert row in telemetry.history


def test_record_initial_with_custom_tick():
    """Test that record_initial accepts custom tick value."""
    world = _build_world(3)
    telemetry = Telemetry()

    row = telemetry.record_initial(world, tick=0)

    assert row.tick == 0


def test_record_initial_sets_previous_beliefs():
    """Test that record_initial sets _previous_beliefs for subsequent delta calculations."""
    world = _build_world(3)
    telemetry = Telemetry()

    # Record initial state
    telemetry.record_initial(world)

    # Verify _previous_beliefs was set
    assert telemetry._previous_beliefs is not None
    assert len(telemetry._previous_beliefs) == 3

    # Take a step and record
    snapshot = world.step()
    row = telemetry.record(snapshot, world)

    # Deltas should now be computed against the initial state
    # (may be zero if beliefs didn't change, but the comparison should happen)
    assert row.mean_abs_delta >= 0.0
    assert row.max_abs_delta >= 0.0


def test_record_initial_materializes_known_claims_for_first_step_delta():
    """Initial telemetry should store complete belief vectors for first-step delta computation."""
    world = _build_world(3)
    telemetry = Telemetry()

    # Simulate lazy defaultdict state before any claim entries have been touched.
    for agent in world.agents:
        agent.beliefs.clear()

    telemetry.record_initial(world)

    assert telemetry._previous_beliefs is not None

    for agent_id, claim_beliefs in telemetry._previous_beliefs.items():
        assert set(claim_beliefs.keys()) == set(world.truths.keys()), (
            f"Agent {agent_id} baseline beliefs did not include all known claims"
        )


def test_first_step_delta_uses_initial_belief_state():
    """First real step delta should compare against initial telemetry baseline."""
    world = _build_world(3)
    telemetry = Telemetry()

    for agent in world.agents:
        agent.beliefs.clear()

    telemetry.record_initial(world)
    initial_beliefs = {
        aid: dict(claims) for aid, claims in telemetry._previous_beliefs.items()
    }

    snapshot = world.step()
    row = telemetry.record(snapshot, world)

    deltas = []
    for agent_id, current_claims in snapshot.agent_beliefs.items():
        previous_claims = initial_beliefs[agent_id]
        for claim_id, current_value in current_claims.items():
            previous_value = previous_claims[claim_id]
            deltas.append(abs(current_value - previous_value))

    assert deltas, "Expected at least one belief delta to compare"

    expected_mean = sum(deltas) / len(deltas)
    expected_max = max(deltas)

    assert row.mean_abs_delta == pytest.approx(expected_mean)
    assert row.max_abs_delta == pytest.approx(expected_max)


# ---------------------------------------------------------------------------
# New social-dynamics telemetry metric tests
# ---------------------------------------------------------------------------


def test_mean_claim_belief_variance_present_and_nonneg():
    """mean_claim_belief_variance is non-negative and appears in exports."""
    world = _build_world(4)
    telemetry = Telemetry()
    row = telemetry.record(world.step(), world)

    assert row.mean_claim_belief_variance >= 0.0


def test_mean_claim_belief_variance_computed_correctly():
    """mean_claim_belief_variance matches manual per-claim variance."""
    world = _build_world(4)
    # Fix all agent beliefs to known values for claim 0.
    vals = [0.2, 0.4, 0.6, 0.8]
    for agent, v in zip(world.agents, vals):
        agent.beliefs[0] = v

    beliefs = world.get_agent_beliefs_snapshot()
    claim_ids = list(world.truths.keys())

    mean = sum(vals) / len(vals)
    expected_var = sum((x - mean) ** 2 for x in vals) / len(vals)

    actual = Telemetry._claim_belief_variance(beliefs, claim_ids)
    assert actual == pytest.approx(expected_var)


def test_mean_claim_belief_variance_zero_for_identical_beliefs():
    """Variance is 0.0 when all agents share the same belief."""
    world = _build_world(3)
    for agent in world.agents:
        agent.beliefs[0] = 0.5

    beliefs = world.get_agent_beliefs_snapshot()
    claim_ids = list(world.truths.keys())

    assert Telemetry._claim_belief_variance(beliefs, claim_ids) == pytest.approx(0.0)


def test_fraction_confident_wrong_zero_when_beliefs_correct():
    """No agent is confident-and-wrong when all beliefs are on the correct side."""
    world = _build_world(3)
    # claim 0 is True; beliefs near 1.0 are correct and confident.
    for agent in world.agents:
        agent.beliefs[0] = 0.9

    beliefs = world.get_agent_beliefs_snapshot()
    result = Telemetry._fraction_confident_wrong(beliefs, world.truths)
    assert result == pytest.approx(0.0)


def test_fraction_confident_wrong_one_when_all_confident_wrong():
    """1.0 when every agent is confidently wrong about a true claim."""
    agents = [Agent(i, rng_seed=i) for i in range(3)]
    world = World(agents=agents, truths={0: True}, rng_seed=1)
    for agent in world.agents:
        agent.beliefs[0] = 0.1  # wrong side for a true claim, confidence=0.8

    beliefs = world.get_agent_beliefs_snapshot()
    result = Telemetry._fraction_confident_wrong(beliefs, world.truths)
    assert result == pytest.approx(1.0)


def test_fraction_confident_wrong_is_between_zero_and_one():
    """fraction_confident_wrong is always in [0, 1]."""
    world = _build_world(5)
    telemetry = Telemetry()
    row = telemetry.record(world.step(), world)

    assert 0.0 <= row.fraction_confident_wrong <= 1.0


def test_mean_trust_and_trust_std_zero_with_no_trust_entries():
    """mean_trust and trust_std are 0.0 when no trust entries have been realized."""
    world = _build_world(3)
    telemetry = Telemetry()

    # Before any social interaction, trust defaultdicts are empty.
    row = telemetry.record_initial(world)

    assert row.mean_trust == pytest.approx(0.0)
    assert row.trust_std == pytest.approx(0.0)


def test_mean_trust_reflects_realized_entries():
    """mean_trust equals the average of trust values explicitly set on agents."""
    world = _build_world(3)
    # Manually trigger trust entries.
    world.get_agent(0).trust[1] = 0.6
    world.get_agent(0).trust[2] = 0.4

    mean, std = Telemetry._trust_stats(world.agents)

    assert mean == pytest.approx(0.5)
    assert std == pytest.approx(0.1)


def test_trust_std_zero_for_uniform_trust():
    """trust_std is 0.0 when all realized trust values are identical."""
    world = _build_world(2)
    world.get_agent(0).trust[1] = 0.7
    world.get_agent(1).trust[0] = 0.7

    _, std = Telemetry._trust_stats(world.agents)
    assert std == pytest.approx(0.0)


def test_new_metrics_in_to_dict_and_jsonl():
    """New metrics are present in to_dict output and JSONL export."""
    world = _build_world(3)
    telemetry = Telemetry()
    row = telemetry.record(world.step(), world)
    d = row.to_dict()

    assert "mean_claim_belief_variance" in d
    assert "fraction_confident_wrong" in d
    assert "mean_trust" in d
    assert "trust_std" in d


def test_new_metrics_in_format_cli():
    """New metrics appear in the CLI-formatted string."""
    world = _build_world(3)
    telemetry = Telemetry()
    row = telemetry.record(world.step(), world)
    formatted = row.format_cli()

    assert "bvar=" in formatted
    assert "conf_wrong=" in formatted
    assert "trust_mean=" in formatted
    assert "trust_std=" in formatted
