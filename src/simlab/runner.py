from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from simlab.config import build_world, load_config
from simlab.telemetry import Telemetry, TelemetryRow


@dataclass(frozen=True, slots=True)
class RunRequest:
    config_path: str
    steps: int


@dataclass(frozen=True, slots=True)
class RunResult:
    telemetry: list[TelemetryRow]
    completed_steps: int
    total_runtime_ms: float


def execute_run(request: RunRequest) -> RunResult:
    """
    Run a simulation headlessly, with no visualization or console output.

    Records an initial telemetry row followed by one row per completed step,
    timing each step in isolation so visualization/pause overhead never
    contaminates the runtime measurement.
    """
    cfg = load_config(request.config_path)
    world = build_world(cfg)

    telemetry = Telemetry()
    telemetry.record_initial(world)

    run_start = perf_counter()

    for _ in range(request.steps):
        step_start = perf_counter()
        snapshot = world.step()
        step_runtime_ms = (perf_counter() - step_start) * 1000
        telemetry.record(snapshot, world, step_runtime_ms=step_runtime_ms)

    total_runtime_ms = (perf_counter() - run_start) * 1000

    return RunResult(
        telemetry=telemetry.history,
        completed_steps=request.steps,
        total_runtime_ms=total_runtime_ms,
    )
