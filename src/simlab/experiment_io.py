from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from simlab.telemetry import TelemetryRow

if TYPE_CHECKING:
    from simlab.runner import RunResult

_LABEL_FIELDS = (
    "converged",
    "final_consensus",
    "final_truth_aligned",
    "final_false_consensus",
)


def _validate_run_id(run_id: str) -> None:
    """Reject anything that isn't a single, literal path component.

    ``run_id`` becomes a directory name under ``output_dir``; without this
    check a value like ``"../../etc"`` would place the run (and, with
    ``overwrite=True``, an rmtree) outside ``output_dir``.
    """
    if not run_id or run_id in (".", "..") or os.path.basename(run_id) != run_id:
        raise ValueError(
            f"invalid run_id: {run_id!r} (must be a single path component)"
        )


def _write_trajectory_csv(rows: Sequence[TelemetryRow], path: str) -> None:
    fieldnames = list(TelemetryRow.__annotations__.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def _build_manifest(result: RunResult) -> dict[str, Any]:
    metadata = result.metadata
    return {
        "schema_version": metadata.schema_version,
        "run_id": metadata.run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": metadata.config_path,
        "config_fingerprint": metadata.config_fingerprint,
        "resolved_config": metadata.resolved_config,
        "world_seed": metadata.world_seed,
        "requested_steps": metadata.requested_steps,
        "completed_steps": metadata.completed_steps,
        "num_agents": metadata.num_agents,
        "num_claims": metadata.num_claims,
        "profile_counts": metadata.profile_counts,
    }


def _build_summary_doc(result: RunResult) -> dict[str, Any]:
    summary_dict = asdict(result.summary)
    outcomes = {k: v for k, v in summary_dict.items() if k not in _LABEL_FIELDS}
    labels = {k: summary_dict[k] for k in _LABEL_FIELDS}
    return {
        "run_id": result.metadata.run_id,
        "scenario": result.scenario,
        "outcomes": outcomes,
        "labels": labels,
    }


def write_run_artifacts(
    result: RunResult, output_dir: str, *, overwrite: bool = False
) -> str:
    """
    Write manifest.json, summary.json, and trajectory.csv for a completed
    run to `<output_dir>/<run_id>/`.

    All three files are written to a temporary sibling directory first and
    moved into place only once complete, so a crash or exception mid-write
    never leaves a partially-written run directory at the final path.

    This function does not coordinate across processes: concurrent callers
    must use distinct `run_id`s. Two writers racing on the *same* run_id
    is only made to fail safely (never silently clobber) when `overwrite`
    is False; with `overwrite=True` a concurrent writer to the same
    run_id can still raise, since "overwrite" only promises to replace
    whatever was there when this call started, not to out-wait a rival.

    :raises FileExistsError: if the run directory already exists and
        `overwrite` is False.
    :return: the final run directory path.
    """
    run_id = result.metadata.run_id
    _validate_run_id(run_id)

    os.makedirs(output_dir, exist_ok=True)
    final_dir = os.path.join(output_dir, run_id)

    if not overwrite and os.path.exists(final_dir):
        raise FileExistsError(
            f"Run directory already exists: {final_dir} "
            "(pass overwrite=True to replace it)"
        )

    tmp_dir = tempfile.mkdtemp(prefix=f".{run_id}-", dir=output_dir)
    try:
        with open(os.path.join(tmp_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(_build_manifest(result), f, indent=2, sort_keys=True)
        with open(os.path.join(tmp_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(_build_summary_doc(result), f, indent=2, sort_keys=True)
        _write_trajectory_csv(result.telemetry, os.path.join(tmp_dir, "trajectory.csv"))

        if overwrite and os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        try:
            os.rename(tmp_dir, final_dir)
        except OSError as exc:
            # os.rename() refuses to land on a non-empty directory, so if a
            # concurrent writer populated final_dir between our check above
            # and this rename, that's what we land here for -- confirm that
            # is really what happened (rather than assuming any OSError
            # means a lost race) before reporting it as a collision.
            if not overwrite and os.path.exists(final_dir):
                raise FileExistsError(
                    f"Run directory already exists: {final_dir} "
                    "(pass overwrite=True to replace it)"
                ) from exc
            raise
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return final_dir
