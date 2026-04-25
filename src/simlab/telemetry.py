from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math

from simlab.sim import Snapshot, World


@dataclass(frozen=True, slots=True)
class TelemetryRow:
    tick: int
    claim_id: int

    belief_mean: float
    belief_std: float
    belief_min: float
    belief_max: float

    mean_abs_delta: float
    max_abs_delta: float

    num_observations: int
    num_verifications: int
    num_communicate_edges: int
    num_broadcast_edges: int
    num_agent_updates: int

    step_runtime_ms: float | None = None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "tick": self.tick,
            "claim_id": self.claim_id,
            "belief_mean": self.belief_mean,
            "belief_std": self.belief_std,
            "belief_min": self.belief_min,
            "belief_max": self.belief_max,
            "mean_abs_delta": self.mean_abs_delta,
            "max_abs_delta": self.max_abs_delta,
            "num_observations": self.num_observations,
            "num_verifications": self.num_verifications,
            "num_communicate_edges": self.num_communicate_edges,
            "num_broadcast_edges": self.num_broadcast_edges,
            "num_agent_updates": self.num_agent_updates,
            "step_runtime_ms": self.step_runtime_ms,
        }


class Telemetry:
    def __init__(self, *, keep_history: bool = True, max_history: int | None = None):
        if max_history is not None and max_history < 1:
            raise ValueError("max_history must be >= 1 or None")

        self.keep_history = keep_history
        self.max_history = max_history

        self.latest: TelemetryRow | None = None
        self.history: list[TelemetryRow] = []

    @staticmethod
    def _belief_stats(vals: list[float]) -> tuple[float, float, float, float]:
        if not vals:
            return 0.0, 0.0, 0.0, 0.0
        n = len(vals)
        mean = sum(vals) / n
        var = sum((x - mean) ** 2 for x in vals) / n  # population variance
        std = math.sqrt(var)
        return mean, std, min(vals), max(vals)

    @staticmethod
    def _delta_stats(
        before: dict[int, float], after: dict[int, float]
    ) -> tuple[float, float]:
        if not after:
            return 0.0, 0.0
        deltas = [abs(after[aid] - before.get(aid, after[aid])) for aid in after.keys()]
        if not deltas:
            return 0.0, 0.0
        n = len(deltas)
        return (sum(deltas) / n, max(deltas))

    def record(
        self,
        snapshot: Snapshot,
        world: World,
        *,
        step_runtime_ms: float | None = None,
    ) -> TelemetryRow:
        # world is accepted for future extensions; current metrics derive from snapshot.
        _ = world

        vals = list(snapshot.belief_after.values())
        belief_mean, belief_std, belief_min, belief_max = self._belief_stats(vals)
        mean_abs_delta, max_abs_delta = self._delta_stats(
            snapshot.belief_before, snapshot.belief_after
        )

        row = TelemetryRow(
            tick=snapshot.tick,
            claim_id=snapshot.claim_id,
            belief_mean=belief_mean,
            belief_std=belief_std,
            belief_min=belief_min,
            belief_max=belief_max,
            mean_abs_delta=mean_abs_delta,
            max_abs_delta=max_abs_delta,
            num_observations=len(snapshot.observed_ids),
            num_verifications=len(snapshot.verified_ids),
            num_communicate_edges=len(snapshot.communicate_edges),
            num_broadcast_edges=len(snapshot.broadcast_edges),
            num_agent_updates=int(snapshot.agent_updates),
            step_runtime_ms=step_runtime_ms,
        )

        self.latest = row
        if self.keep_history:
            self.history.append(row)
            if self.max_history is not None and len(self.history) > self.max_history:
                extra = len(self.history) - self.max_history
                del self.history[0:extra]

        return row

    def export_csv(self, path: str) -> None:
        rows = self.history
        fieldnames = list(TelemetryRow.__annotations__.keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r.to_dict())

    def export_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in self.history:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
