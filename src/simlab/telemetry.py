from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math

from simlab.sim import Snapshot, World


@dataclass(frozen=True, slots=True)
class TelemetryRow:
    tick: int

    # ---------------------------------------------------------------------
    # Global belief distribution
    # Flattened across all agents and all claims.
    # Useful as a broad "what is the belief space doing?" signal.
    # ---------------------------------------------------------------------
    belief_mean: float
    belief_std: float
    belief_min: float
    belief_max: float

    # ---------------------------------------------------------------------
    # Belief movement / volatility
    # Compares current beliefs against the previous recorded snapshot.
    # ---------------------------------------------------------------------
    mean_abs_delta: float
    max_abs_delta: float

    # ---------------------------------------------------------------------
    # Truth alignment
    # Compares each belief to the actual world truth for that claim.
    # For True claims, target = 1.0.
    # For False claims, target = 0.0.
    # ---------------------------------------------------------------------
    mean_abs_error_to_truth: float
    max_abs_error_to_truth: float
    fraction_truth_aligned: float

    # ---------------------------------------------------------------------
    # Event / action counts
    # ---------------------------------------------------------------------
    num_observations: int
    num_verifications: int
    num_communicate_edges: int
    num_broadcast_edges: int
    num_agent_updates: int

    # ---------------------------------------------------------------------
    # Runtime
    # ---------------------------------------------------------------------
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
            "mean_abs_error_to_truth": self.mean_abs_error_to_truth,
            "max_abs_error_to_truth": self.max_abs_error_to_truth,
            "fraction_truth_aligned": self.fraction_truth_aligned,
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
            f"truth_err_mean={self.mean_abs_error_to_truth:.4f} "
            f"truth_err_max={self.max_abs_error_to_truth:.4f} "
            f"aligned={self.fraction_truth_aligned:.2%} | "
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
    def _truth_alignment_stats(
        beliefs: dict[int, dict[int, float]],
        truths: dict[int, bool],
        *,
        alignment_threshold: float = 0.2,
    ) -> tuple[float, float, float]:
        """
        Compute truth-alignment metrics across all agent/claim beliefs.

        For each belief:
        - if the claim is true, target belief is 1.0
        - if the claim is false, target belief is 0.0

        Metrics:
        - mean_abs_error_to_truth:
            Average absolute distance from the correct truth target.
        - max_abs_error_to_truth:
            Worst absolute distance from the correct truth target.
        - fraction_truth_aligned:
            Fraction of beliefs within `alignment_threshold` of the truth target.

        Claims not present in `truths` are skipped.
        """
        errors: list[float] = []

        for _agent_id, claim_beliefs in beliefs.items():
            for claim_id, belief_value in claim_beliefs.items():
                if claim_id not in truths:
                    continue

                target = 1.0 if truths[claim_id] else 0.0
                errors.append(abs(belief_value - target))

        if not errors:
            return 0.0, 0.0, 0.0

        n = len(errors)
        mean_abs_error = sum(errors) / n
        max_abs_error = max(errors)
        fraction_aligned = sum(err <= alignment_threshold for err in errors) / n

        return mean_abs_error, max_abs_error, fraction_aligned

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
    
    def record_initial(self, world: World, *, tick: int = -1) -> TelemetryRow:
        """
        Record the initial world state before any simulation steps are processed.

        This creates a baseline telemetry row so plots and analysis can show
        where the simulation started before tick 0 was applied.

        Event/action counts are zero because no step has occurred yet.
        Delta metrics are zero because there is no prior recorded state.
        """
        # -----------------------------------------------------------------
        # 1. Capture current full belief state from the world
        # agent_id -> claim_id -> belief_value
        # -----------------------------------------------------------------
        agent_beliefs: dict[int, dict[int, float]] = {
            agent.id: dict(agent.beliefs)
            for agent in world.agents
        }

        # -----------------------------------------------------------------
        # 2. Global belief distribution
        # -----------------------------------------------------------------
        vals: list[float] = []
        for claim_beliefs in agent_beliefs.values():
            vals.extend(claim_beliefs.values())

        belief_mean, belief_std, belief_min, belief_max = self._belief_stats(vals)

        # -----------------------------------------------------------------
        # 3. Initial movement / volatility
        # No prior state exists, so deltas are zero.
        # -----------------------------------------------------------------
        mean_abs_delta = 0.0
        max_abs_delta = 0.0

        # -----------------------------------------------------------------
        # 4. Truth alignment against static world truths
        # -----------------------------------------------------------------
        (
            mean_abs_error_to_truth,
            max_abs_error_to_truth,
            fraction_truth_aligned,
        ) = self._truth_alignment_stats(agent_beliefs, world.truths)

        # -----------------------------------------------------------------
        # 5. Initial baseline row
        # No events/actions have happened yet.
        # -----------------------------------------------------------------
        row = TelemetryRow(
            tick=tick,
            belief_mean=belief_mean,
            belief_std=belief_std,
            belief_min=belief_min,
            belief_max=belief_max,
            mean_abs_delta=mean_abs_delta,
            max_abs_delta=max_abs_delta,
            mean_abs_error_to_truth=mean_abs_error_to_truth,
            max_abs_error_to_truth=max_abs_error_to_truth,
            fraction_truth_aligned=fraction_truth_aligned,
            num_observations=0,
            num_verifications=0,
            num_communicate_edges=0,
            num_broadcast_edges=0,
            num_agent_updates=0,
            step_runtime_ms=None,
        )

        self.latest = row

        if self.keep_history:
            self.history.append(row)
            if self.max_history is not None and len(self.history) > self.max_history:
                extra = len(self.history) - self.max_history
                del self.history[0:extra]

        # Important: set previous beliefs so tick 0 deltas compare against
        # the actual initial world state.
        self._previous_beliefs = agent_beliefs

        return row

    def record(
        self,
        snapshot: Snapshot,
        world: World,
        *,
        step_runtime_ms: float | None = None,
    ) -> TelemetryRow:
        # -----------------------------------------------------------------
        # 1. Global belief distribution
        # Flatten all agent_id -> claim_id -> belief_value pairs into one
        # list. This is intentionally broad and claim-agnostic.
        # -----------------------------------------------------------------
        vals: list[float] = []
        for claim_beliefs in snapshot.agent_beliefs.values():
            vals.extend(claim_beliefs.values())

        belief_mean, belief_std, belief_min, belief_max = self._belief_stats(vals)

        # -----------------------------------------------------------------
        # 2. Belief movement / volatility
        # Compare current full belief state against the previous recorded
        # full belief state. The compact history does not store full beliefs.
        # -----------------------------------------------------------------
        mean_abs_delta, max_abs_delta = self._delta_stats(
            self._previous_beliefs,
            snapshot.agent_beliefs,
        )

        # -----------------------------------------------------------------
        # 3. Truth alignment
        # Compare beliefs against static world truths.
        # This is more semantically meaningful than raw belief mean.
        # -----------------------------------------------------------------
        (
            mean_abs_error_to_truth,
            max_abs_error_to_truth,
            fraction_truth_aligned,
        ) = self._truth_alignment_stats(
            snapshot.agent_beliefs,
            world.truths,
        )

        # -----------------------------------------------------------------
        # 4. Event/action counts + runtime
        # -----------------------------------------------------------------
        row = TelemetryRow(
            tick=snapshot.tick,
            belief_mean=belief_mean,
            belief_std=belief_std,
            belief_min=belief_min,
            belief_max=belief_max,
            mean_abs_delta=mean_abs_delta,
            max_abs_delta=max_abs_delta,
            mean_abs_error_to_truth=mean_abs_error_to_truth,
            max_abs_error_to_truth=max_abs_error_to_truth,
            fraction_truth_aligned=fraction_truth_aligned,
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

        # Store full belief state privately for next-step delta computation.
        # Telemetry.history remains compact.
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

