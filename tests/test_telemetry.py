import csv
import json
import math
import io
import sys

import pytest

from simlab.sim import Agent, World, Snapshot
from simlab.telemetry import Telemetry
from simlab.viz import run_viz
from simlab.viz.view_model import compute_viewmodel
from simlab.viz.scene import build_scene


def _build_world(n: int = 5) -> World:
    agents = [Agent(i, rng_seed=i) for i in range(n)]
    return World(agents=agents, truths={0: True}, rng_seed=1)


def test_snapshot_stores_full_beliefs():
    """Test that Snapshot stores full beliefs as agent_id -> claim_id -> belief_value."""
    world = _build_world(3)
    snapshot = world.step()

    # Check that beliefs is a dict with agent_id as keys
    assert isinstance(snapshot.agent_beliefs, dict)
    assert len(snapshot.agent_beliefs) == 3

    # Check that each agent's beliefs is a dict with claim_id as keys
    for agent_id, claim_beliefs in snapshot.agent_beliefs.items():
        assert isinstance(agent_id, int)
        assert isinstance(claim_beliefs, dict)
        # Each agent should have beliefs for at least claim 0
        assert 0 in claim_beliefs
        # Belief values should be floats in [0, 1]
        for claim_id, belief_value in claim_beliefs.items():
            assert isinstance(claim_id, int)
            assert isinstance(belief_value, float)
            assert 0.0 <= belief_value <= 1.0


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


def test_world_step_does_not_print_logs():
    """Test that World.step() does not print logs (no World.log_step())."""
    world = _build_world(3)

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        for _ in range(5):
            world.step()
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    # World.step() should not print anything
    assert output == ""


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


def test_run_viz_records_telemetry_history(mocker):
    """Test that run_viz correctly records telemetry history with mocked world.step."""
    # Given
    world = _build_world(3)
    telemetry = Telemetry()

    snapshots = [
        Snapshot(
            tick=tick,
            agent_beliefs={0: {0: 0.5, 1: 0.5}, 1: {0: 0.5, 1: 0.5}, 2: {0: 0.5, 1: 0.5}},
            observed_ids=[0],
            verified_ids=[],
            communicate_edges=[],
            broadcast_edges=[],
            n_agent_updates=1,
            agent_memory_sizes={0: 0, 1: 0, 2: 0},
        )
        for tick in range(5)
    ]

    # Mock matplotlib functions to avoid GUI
    mocker.patch.object(world, "step", side_effect=snapshots)
    mocker.patch("simlab.viz.network_viz.plt.ion")
    mocker.patch("simlab.viz.network_viz.plt.show")
    mocker.patch("simlab.viz.network_viz.plt.pause")
    mocker.patch("simlab.viz.network_viz.plt.ioff")

    # When
    run_viz(
        world,
        steps=5,
        telemetry=telemetry,
        log_every=10,  # Don't print during test
        draw_every=10,  # Don't draw during test
    )

    # Then
    # Verify that telemetry.history has the correct number of entries
    assert len(telemetry.history) == 6

    # Verify that tick values are correct (-1, 0, 1, 2, 3, 4)
    ticks = [row.tick for row in telemetry.history]
    assert ticks == [-1, 0, 1, 2, 3, 4]

    # Verify that latest is the last snapshot
    assert telemetry.latest.tick == 4


def test_run_viz_validates_log_every():
    """Test that run_viz raises ValueError when log_every is 0."""
    world = _build_world(3)
    telemetry = Telemetry()

    with pytest.raises(ValueError, match="log_every must be >= 1"):
        run_viz(world, steps=5, telemetry=telemetry, log_every=0)


def test_run_viz_validates_draw_every():
    """Test that run_viz raises ValueError when draw_every is 0."""
    world = _build_world(3)
    telemetry = Telemetry()

    with pytest.raises(ValueError, match="draw_every must be >= 1"):
        run_viz(world, steps=5, telemetry=telemetry, draw_every=0)


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


def test_compute_viewmodel_requires_snapshot():
    """Test that compute_viewmodel requires a non-None Snapshot."""
    world = _build_world(3)
    scene = build_scene(world)

    # Should raise TypeError or AttributeError when snapshot is None
    with pytest.raises((TypeError, AttributeError)):
        compute_viewmodel(scene, claim_id=0, step_snapshot=None)


def test_compute_viewmodel_with_telemetry_row():
    """Test that compute_viewmodel accepts telemetry_row parameter."""
    world = _build_world(3)
    scene = build_scene(world)
    snapshot = world.step()
    telemetry = Telemetry()
    telemetry_row = telemetry.record(snapshot, world)

    # Should work with telemetry_row (even if unused currently)
    vm = compute_viewmodel(scene, claim_id=0, step_snapshot=snapshot, telemetry_row=telemetry_row)

    assert vm.tick == snapshot.tick
    assert vm.claim_id == 0


def test_run_viz_draw_uses_snapshot_tick(mocker):
    """Test that run_viz draw check uses snapshot.tick instead of world.tick."""
    world = _build_world(3)
    telemetry = Telemetry()

    # Create snapshots with specific tick values
    snapshots = [
        Snapshot(
            tick=tick,
            agent_beliefs={0: {0: 0.5, 1: 0.5}, 1: {0: 0.5, 1: 0.5}, 2: {0: 0.5, 1: 0.5}},
            observed_ids=[],
            verified_ids=[],
            communicate_edges=[],
            broadcast_edges=[],
            n_agent_updates=0,
            agent_memory_sizes={0: 0, 1: 0, 2: 0},
        )
        for tick in range(5)
    ]

    # Mock world.step to return snapshots
    mocker.patch.object(world, "step", side_effect=snapshots)
    mocker.patch("simlab.viz.network_viz.plt.ion")
    mocker.patch("simlab.viz.network_viz.plt.show")
    mocker.patch("simlab.viz.network_viz.plt.pause")
    mocker.patch("simlab.viz.network_viz.plt.ioff")

    # Track draw calls
    draw_calls = []
    original_draw = mocker.patch("simlab.viz.network_viz.NetworkViz.draw", side_effect=lambda *args, **kwargs: draw_calls.append((args, kwargs)))

    # Run with draw_every=2 (should draw on ticks 0, 2, 4)
    run_viz(
        world,
        steps=5,
        telemetry=telemetry,
        log_every=10,
        draw_every=2,
    )

    # Verify draw was called for ticks 0, 2, 4
    assert len(draw_calls) == 3


def test_get_agent_beliefs_snapshot_materializes_known_claims_from_lazy_beliefs():
    world = _build_world(3)

    for agent in world.agents:
        agent.beliefs.clear()

    snapshot = world.get_agent_beliefs_snapshot()

    assert set(snapshot.keys()) == {agent.id for agent in world.agents}

    for agent in world.agents:
        claim_beliefs = snapshot[agent.id]
        assert set(claim_beliefs.keys()) == set(world.truths.keys())

        for claim_id in world.truths:
            assert isinstance(claim_beliefs[claim_id], float)


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
        aid: dict(claims)
        for aid, claims in telemetry._previous_beliefs.items()
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