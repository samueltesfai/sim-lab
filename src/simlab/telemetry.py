from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math

from simlab.sim import Snapshot, World


@dataclass(frozen=True, slots=True)
class TelemetryRow:
    tick: int

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
    
    def format_cli(self) -> str:
        runtime_str = (
            f" | runtime={self.step_runtime_ms:.3f}ms"
            if self.step_runtime_ms is not None
            else ""
        )

        return (
            f"Tick {self.tick:4d} | "
            f"belief mean={self.belief_mean:.3f} std={self.belief_std:.3f} "
            f"min={self.belief_min:.3f} max={self.belief_max:.3f} | "
            f"Δabs_mean={self.mean_abs_delta:.4f} Δmax={self.max_abs_delta:.4f} | "
            f"events: com={self.num_communicate_edges} | "
            f"bcast={self.num_broadcast_edges} | "
            f"obs={self.num_observations} | "
            f"ver={self.num_verifications} | "
            f"updates={self.num_agent_updates}"
            f"{runtime_str}"
        )


class Telemetry:
    def __init__(self, *, keep_history: bool = True, max_history: int | None = None):
        if max_history is not None and max_history < 1:
            raise ValueError("max_history must be >= 1 or None")

        self.keep_history = keep_history
        self.max_history = max_history

        self.latest: TelemetryRow | None = None
        self.history: list[TelemetryRow] = []
        self._previous_beliefs: dict[int, dict[int, float]] | None = None

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
        before: dict[int, dict[int, float]] | None,
        after: dict[int, dict[int, float]],
    ) -> tuple[float, float]:
        """
        Compute delta statistics by comparing current beliefs against previous beliefs.
        Flattens all agent_id -> claim_id -> belief_value pairs into a single list.
        """
        if before is None:
            # First call - no previous state to compare against
            return 0.0, 0.0

        deltas = []
        for agent_id, claim_beliefs in after.items():
            if agent_id in before:
                prev_claims = before[agent_id]
                for claim_id, belief_value in claim_beliefs.items():
                    prev_value = prev_claims.get(claim_id, belief_value)
                    deltas.append(abs(belief_value - prev_value))
            else:
                # New agent - all beliefs are new
                for belief_value in claim_beliefs.values():
                    deltas.append(belief_value)

        if not deltas:
            return 0.0, 0.0
        n = len(deltas)
        return (sum(deltas) / n, max(deltas))

    def record(
        self,
        snapshot: Snapshot,
        *,
        step_runtime_ms: float | None = None,
    ) -> TelemetryRow:
        # Flatten snapshot.beliefs values into one list of floats
        vals = []
        for claim_beliefs in snapshot.agent_beliefs.values():
            vals.extend(claim_beliefs.values())

        belief_mean, belief_std, belief_min, belief_max = self._belief_stats(vals)
        mean_abs_delta, max_abs_delta = self._delta_stats(
            self._previous_beliefs, snapshot.agent_beliefs
        )

        row = TelemetryRow(
            tick=snapshot.tick,
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
            num_agent_updates=int(snapshot.n_agent_updates),
            step_runtime_ms=step_runtime_ms,
        )

        self.latest = row
        if self.keep_history:
            self.history.append(row)
            if self.max_history is not None and len(self.history) > self.max_history:
                extra = len(self.history) - self.max_history
                del self.history[0:extra]

        # Update previous beliefs for next delta computation
        self._previous_beliefs = snapshot.agent_beliefs

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

