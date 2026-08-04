from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from simlab.config import (
    expand_agent_specs,
    world_from_config,
    load_config,
)
from simlab.run_analysis import (
    RunSummary,
    compute_run_summary,
    extract_scenario_features,
)
from simlab.telemetry import Telemetry, TelemetryRow

SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class RunRequest:
    config_path: str
    steps: int
    run_id: str | None = None

    def __post_init__(self) -> None:
        # bool is an int subclass, so steps=True would otherwise pass.
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise TypeError(f"steps must be an int, got {type(self.steps).__name__}")
        if self.steps < 0:
            raise ValueError(f"steps must be >= 0, got {self.steps}")


@dataclass(frozen=True, slots=True)
class RunMetadata:
    schema_version: str
    run_id: str
    config_path: str
    scenario_fingerprint: str
    run_spec_fingerprint: str
    resolved_config: dict[str, Any]
    world_seed: int
    requested_steps: int
    completed_steps: int
    num_agents: int
    num_claims: int
    profile_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class RunResult:
    metadata: RunMetadata
    scenario: dict[str, float | int]
    summary: RunSummary
    telemetry: list[TelemetryRow]


def _normalize_for_fingerprint(data: Any) -> Any:
    """Recursively normalize numbers so behaviorally-equivalent
    representations hash identically: ``-0.0`` becomes ``0.0``, and an
    integral float (e.g. YAML's ``1.0``) is unified with the plain int it
    equals (``1``).

    ``json.dumps(-0.0) != json.dumps(0.0)`` even though ``-0.0 == 0.0`` in
    every arithmetic sense the simulation cares about, and likewise
    ``json.dumps(1) != json.dumps(1.0)`` even though config validation
    accepts either for a float-valued field -- both would otherwise give
    behaviorally identical scenarios different fingerprints.

    This only ever converts float -> int, never int -> float: a blanket
    ``float(x)`` on every int would silently collapse distinct large seeds
    that exceed float64's 2**53 exact-integer range (e.g.
    ``9007199254740992`` and ``...993`` both become the same float), which
    would corrupt ``run_spec_fingerprint`` for exactly the field it exists
    to distinguish. Genuine ints are therefore left untouched. Bools are
    left alone despite being an ``int`` subclass, since ``True``/``False``
    are a distinct JSON type from numbers.

    :param data: A JSON-safe value (dict, list, or scalar)
    :type data: Any
    :return: The same structure with every number in canonical form
    :rtype: Any
    """
    if isinstance(data, dict):
        return {k: _normalize_for_fingerprint(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_normalize_for_fingerprint(v) for v in data]
    if isinstance(data, bool):
        return data
    if isinstance(data, float):
        normalized = data + 0.0
        return int(normalized) if normalized.is_integer() else normalized
    return data


def _fingerprint(data: dict[str, Any]) -> str:
    """Hash a JSON-safe dict so identical inputs share an identifier
    regardless of key order, numeric sign-of-zero, or the source file's
    path/formatting.

    :param data: A JSON-serializable dict to fingerprint
    :type data: dict
    :return: A SHA-256 hex digest of the dict's canonical JSON encoding
    :rtype: str
    """
    canonical = json.dumps(
        _normalize_for_fingerprint(data), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def execute_run(request: RunRequest) -> RunResult:
    """
    Run a simulation headlessly, with no visualization or console output.

    Records an initial telemetry row followed by one row per completed step,
    timing each step in isolation so visualization/pause overhead never
    contaminates the runtime measurement. Also derives run metadata --
    a ``scenario_fingerprint`` identifying the behaviorally meaningful
    configuration (excluding the seed, since different seeds are stochastic
    replicates of the same scenario, not different scenarios; and excluding
    profile names, which are reporting-only labels), a
    ``run_spec_fingerprint`` identifying this exact requested replicate
    (scenario + seed + steps), and a ``run_id`` for this specific execution
    -- scenario features describing the conditions the run started under,
    and a run summary aggregating the full trajectory into initial/final
    state, activity totals, and a conservative set of outcome labels.

    :param request: The config path, step count, and optional run id
    :type request: RunRequest
    :return: Metadata, scenario features, run summary, and full telemetry
    :rtype: RunResult
    """
    cfg = load_config(request.config_path)
    world = world_from_config(cfg)
    resolved_config: dict[str, Any] = cfg.model_dump()
    # world_from_config only consumes the flattened per-agent list, never
    # profile boundaries -- fingerprint that, not the raw profile list
    per_agent_settings = [
        profile.model_dump(exclude={"name", "count"})
        for profile in expand_agent_specs(cfg)
    ]
    scenario_fingerprint = _fingerprint(
        {
            "world": cfg.world.model_dump(exclude={"rng_seed"}),
            "agents": per_agent_settings,
        }
    )
    world_seed = cfg.world.rng_seed
    run_spec_fingerprint = _fingerprint(
        {
            "scenario_fingerprint": scenario_fingerprint,
            "world_seed": world_seed,
            "requested_steps": request.steps,
        }
    )

    telemetry = Telemetry()
    initial_row = telemetry.record_initial(world)
    scenario = extract_scenario_features(world, initial_row, cfg)

    run_start = perf_counter()

    for _ in range(request.steps):
        step_start = perf_counter()
        snapshot = world.step()
        step_runtime_ms = (perf_counter() - step_start) * 1000
        telemetry.record(snapshot, world, step_runtime_ms=step_runtime_ms)

    total_runtime_ms = (perf_counter() - run_start) * 1000

    metadata = RunMetadata(
        schema_version=SCHEMA_VERSION,
        run_id=request.run_id or uuid.uuid4().hex,
        config_path=request.config_path,
        scenario_fingerprint=scenario_fingerprint,
        run_spec_fingerprint=run_spec_fingerprint,
        resolved_config=resolved_config,
        world_seed=world_seed,
        requested_steps=request.steps,
        completed_steps=request.steps,
        num_agents=len(world.agents),
        num_claims=len(world.claims),
        profile_counts=world.profile_counts,
    )

    summary = compute_run_summary(telemetry.history, total_runtime_ms=total_runtime_ms)

    return RunResult(
        metadata=metadata,
        scenario=scenario,
        summary=summary,
        telemetry=telemetry.history,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a simulation headlessly and write run artifacts "
        "(manifest.json, summary.json, trajectory.csv).",
    )
    parser.add_argument(
        "-f",
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "-t",
        "--steps",
        type=int,
        required=True,
        help="Number of simulation steps to run",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="runs",
        help="Directory to write the run's artifact folder under (default: runs)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run id to use (default: generated). Must be unique "
        "across any concurrently running writers to the same output-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing run directory with the same run id",
    )
    return parser


def main() -> None:
    # Imported lazily so that importing simlab.runner for its library API
    # (execute_run/RunRequest/RunResult) never pulls in file-writing code.
    from simlab.experiment_io import write_run_artifacts

    args = _build_arg_parser().parse_args()

    result = execute_run(
        RunRequest(config_path=args.config, steps=args.steps, run_id=args.run_id)
    )
    run_dir = write_run_artifacts(result, args.output_dir, overwrite=args.overwrite)

    print(
        f"Run {result.metadata.run_id} complete: "
        f"{result.metadata.completed_steps} steps"
    )
    print(f"Artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
