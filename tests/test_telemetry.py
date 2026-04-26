import csv
import json
import math
import io
import sys

import pytest

from simlab.sim import Agent, World
from simlab.telemetry import Telemetry, format_telemetry_row


def _build_world(n: int = 5) -> World:
    agents = [Agent(i, rng_seed=i) for i in range(n)]
    return World(agents=agents, truths={0: True}, rng_seed=1)


def test_snapshot_stores_full_beliefs():
    """Test that Snapshot stores full beliefs as agent_id -> claim_id -> belief_value."""
    world = _build_world(3)
    snapshot = world.step()

    # Check that beliefs is a dict with agent_id as keys
    assert isinstance(snapshot.beliefs, dict)
    assert len(snapshot.beliefs) == 3

    # Check that each agent's beliefs is a dict with claim_id as keys
    for agent_id, claim_beliefs in snapshot.beliefs.items():
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
    assert row.num_agent_updates == snapshot.agent_updates


def test_record_computes_global_belief_metrics():
    """Test that Telemetry computes global belief metrics across all beliefs."""
    world = _build_world(4)
    telemetry = Telemetry()
    snapshot = world.step()
    row = telemetry.record(snapshot, world)

    # Flatten all beliefs from snapshot
    vals = []
    for claim_beliefs in snapshot.beliefs.values():
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
    for agent_id, claim_beliefs in s1.beliefs.items():
        if agent_id in s0.beliefs:
            prev_claims = s0.beliefs[agent_id]
            for claim_id, belief_value in claim_beliefs.items():
                prev_value = prev_claims.get(claim_id, belief_value)
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
    # claim_id should not be in the exported data
    assert "claim_id" not in objs[0]

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["tick"] == str(telemetry.history[0].tick)
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

    formatted = format_telemetry_row(row)

    # Check that core fields are present in the formatted string
    assert f"Tick {row.tick}" in formatted
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

    formatted = format_telemetry_row(row)

    # Should not include runtime
    assert "runtime" not in formatted
    # But should include other fields
    assert f"Tick {row.tick}" in formatted
