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

    snapshot = world.step(claim_id=0)
    row = telemetry.record(snapshot, world)

    assert telemetry.latest == row
    assert telemetry.history == [row]
    assert row.tick == snapshot.tick
    assert row.claim_id == 0

    assert row.num_observations == len(snapshot.observed_ids)
    assert row.num_verifications == len(snapshot.verified_ids)
    assert row.num_communicate_edges == len(snapshot.communicate_edges)
    assert row.num_broadcast_edges == len(snapshot.broadcast_edges)
    assert row.num_agent_updates == snapshot.agent_updates


def test_record_matches_population_std_and_delta_stats():
    world = _build_world(4)
    telemetry = Telemetry()
    snapshot = world.step(claim_id=0)
    row = telemetry.record(snapshot, world)

    vals = list(snapshot.belief_after.values())
    n = len(vals)
    mean = sum(vals) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in vals) / n if n else 0.0
    std = math.sqrt(var)

    assert row.belief_mean == pytest.approx(mean)
    assert row.belief_std == pytest.approx(std)
    assert row.belief_min == pytest.approx(min(vals))
    assert row.belief_max == pytest.approx(max(vals))

    deltas = [
        abs(
            snapshot.belief_after[aid]
            - snapshot.belief_before.get(aid, snapshot.belief_after[aid])
        )
        for aid in snapshot.belief_after.keys()
    ]
    assert row.mean_abs_delta == pytest.approx(sum(deltas) / len(deltas))
    assert row.max_abs_delta == pytest.approx(max(deltas))


def test_max_history_trims_oldest():
    world = _build_world(3)
    telemetry = Telemetry(max_history=2)

    s0 = world.step(claim_id=0)
    r0 = telemetry.record(s0, world)
    s1 = world.step(claim_id=0)
    r1 = telemetry.record(s1, world)
    s2 = world.step(claim_id=0)
    r2 = telemetry.record(s2, world)

    assert telemetry.latest == r2
    assert telemetry.history == [r1, r2]
    assert r0 not in telemetry.history


def test_export_jsonl_and_csv(tmp_path):
    world = _build_world(3)
    telemetry = Telemetry()

    telemetry.record(world.step(claim_id=0), world, step_runtime_ms=1.23)
    telemetry.record(world.step(claim_id=0), world, step_runtime_ms=None)

    jsonl_path = tmp_path / "telemetry.jsonl"
    csv_path = tmp_path / "telemetry.csv"

    telemetry.export_jsonl(str(jsonl_path))
    telemetry.export_csv(str(csv_path))

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    objs = [json.loads(line) for line in lines]
    assert objs[0]["tick"] == telemetry.history[0].tick
    assert "step_runtime_ms" in objs[0]

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["tick"] == str(telemetry.history[0].tick)
