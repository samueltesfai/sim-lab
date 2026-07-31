from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from omegaconf import OmegaConf

from simlab.config import build_world, load_config
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


@dataclass(frozen=True, slots=True)
class RunMetadata:
    schema_version: str
    run_id: str
    config_path: str
    config_fingerprint: str
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


def _fingerprint_resolved_config(resolved_config: dict[str, Any]) -> str:
    """Hash the fully-resolved config so identical scenarios share an
    identifier regardless of the source file's path or formatting."""
    canonical = json.dumps(resolved_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def execute_run(request: RunRequest) -> RunResult:
    """
    Run a simulation headlessly, with no visualization or console output.

    Records an initial telemetry row followed by one row per completed step,
    timing each step in isolation so visualization/pause overhead never
    contaminates the runtime measurement. Also derives run metadata (a
    reproducibility fingerprint for the resolved config, world seed, run id),
    scenario features describing the conditions the run started under, and a
    run summary aggregating the full trajectory into initial/final state,
    activity totals, and a conservative set of outcome labels.
    """
    cfg = load_config(request.config_path)
    world = build_world(cfg)
    resolved_config: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)

    telemetry = Telemetry()
    initial_row = telemetry.record_initial(world)
    scenario = extract_scenario_features(world, initial_row)

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
        config_fingerprint=_fingerprint_resolved_config(resolved_config),
        resolved_config=resolved_config,
        world_seed=int(cfg.world.rng_seed),
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
        required=True,
        help="Directory to write the run's artifact folder under",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run id to use (default: generated)",
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
