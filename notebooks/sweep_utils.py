"""Reusable helpers for investigation notebooks under ``notebooks/``.

Each investigation notebook answers its own question and is free to structure
itself however that requires -- this module holds only the mechanical parts
that don't change between investigations: the reproducibility header, running
a scenario x seed sweep through :func:`simlab.runner.execute_run`, the
scenario-vs-seed variance decomposition, and a grouped-CV predictability
probe (plus matching plot helpers). See ``notebooks/README.md`` for how a
notebook is expected to use these.

Lives here rather than in ``src/simlab`` because nothing in it is used by the
CLI or by a single ``execute_run()`` call -- it exists only to orchestrate
and compare *many* runs for notebook analysis, not a kernel concern. That
also means scikit-learn is a normal import below rather than a lazy one:
anywhere this module gets loaded already has the project's "notebook" uv
dependency group installed.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

from simlab._merge import deep_merge
from simlab.runner import SCHEMA_VERSION, RunRequest, execute_run


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _is_ancestor_of_main(commit: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "main"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _is_clean(pathspec: str) -> bool:
    return _git("status", "--porcelain", "--", pathspec) == ""


def reproducibility_header(experiment_name: str) -> dict[str, Any]:
    """Print and return the standard investigation-notebook reproducibility
    checkpoint: which kernel version produced the notebook's results.

    ``kernel_commit`` is the most recent commit touching ``src/simlab``, not
    ``git_commit`` (``HEAD``, which also includes this notebook's own
    not-yet-merged commit -- checking that would be circular, since it can't
    be on ``main`` before its own PR merges). ``kernel_commit_on_main`` is
    only true once that kernel commit is both reachable from ``main`` and
    ``src/simlab`` has no uncommitted changes -- see ``notebooks/README.md``
    for the rule this backs: only commit a dated notebook once this prints
    ``True``.

    :param experiment_name: This notebook's own identifier, conventionally
        matching its filename (``notebooks/<date>-<slug>.ipynb``)
    :type experiment_name: str
    :return: The printed fields, keyed the same as what's printed
    :rtype: dict[str, Any]
    """
    git_branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    git_commit = _git("rev-parse", "HEAD")
    git_dirty = bool(_git("status", "--porcelain"))
    # ":/" anchors the pathspec to the repo root regardless of the caller's
    # cwd (typically notebooks/), which a plain "src/simlab" would silently
    # miss.
    kernel_commit = _git("log", "-1", "--format=%H", "--", ":/src/simlab")
    kernel_path_clean = _is_clean(":/src/simlab")
    kernel_commit_on_main = (
        bool(kernel_commit)
        and kernel_path_clean
        and _is_ancestor_of_main(kernel_commit)
    )

    header = {
        "experiment_name": experiment_name,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": git_branch,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "kernel_commit": kernel_commit,
        "kernel_path_clean": kernel_path_clean,
        "kernel_commit_on_main": kernel_commit_on_main,
        "schema_version": SCHEMA_VERSION,
    }

    width = max(len(key) for key in header)
    print("Notebook reproducibility header")
    for key, value in header.items():
        print(f"  {key:<{width}}: {value}")

    if not kernel_path_clean:
        print()
        print("WARNING: src/simlab has uncommitted changes. The code actually")
        print("exercised by this run may not match kernel_commit at all -- commit")
        print("or discard those changes and rerun this cell before trusting")
        print("kernel_commit_on_main.")
    elif not kernel_commit_on_main:
        print()
        print("WARNING: kernel_commit is not reachable from main. Do not commit this")
        print("notebook to notebooks/ until the kernel code it exercised")
        print("has actually merged to main -- rerun this cell after that merge to")
        print("confirm a stable, permanent commit reference.")

    return header


def expand_ofat_scenarios(
    baseline_cfg: dict,
    axes: dict[str, list[tuple[str, dict]]],
    *,
    baseline_id: str = "baseline",
) -> list[dict]:
    """Expand a baseline config plus one-factor-at-a-time axes into a
    scenario list, in the shape :func:`run_sweep` expects.

    :param baseline_cfg: The shared starting scenario, in raw YAML-shaped
        dict form (as ``simlab.config.load_config`` would read)
    :type baseline_cfg: dict
    :param axes: Axis name -> list of ``(label, override_dict)`` pairs, each
        deep-merged onto ``baseline_cfg`` to produce one scenario
    :type axes: dict[str, list[tuple[str, dict]]]
    :param baseline_id: Scenario id for the unmodified baseline
    :type baseline_id: str
    :return: One dict per scenario: ``{"id", "group", "swept_axis", "label", "cfg"}``
    :rtype: list[dict]
    """
    scenarios = [
        {
            "id": baseline_id,
            "group": "baseline",
            "swept_axis": None,
            "label": baseline_id,
            "cfg": baseline_cfg,
        }
    ]
    for axis, variants in axes.items():
        for label, overrides in variants:
            cfg = deep_merge(baseline_cfg, overrides)
            scenarios.append(
                {
                    "id": f"{axis}:{label}",
                    "group": "ofat",
                    "swept_axis": axis,
                    "label": label,
                    "cfg": cfg,
                }
            )
    return scenarios


def run_sweep(
    scenarios: list[dict],
    seeds: list[int],
    steps: int,
    *,
    keep_trajectories_for_seed: int | None = 0,
) -> tuple[pd.DataFrame, dict[str, list]]:
    """Execute every scenario x seed combination through
    :func:`simlab.runner.execute_run`, unmodified.

    Each scenario's config is written to a temporary YAML file per replicate
    (``execute_run`` reads by path); nothing is persisted to ``runs/``, so an
    exploratory sweep doesn't scatter run directories through the repo --
    ``result.scenario``/``result.summary`` already carry what analysis needs.

    :param scenarios: Scenario dicts, each with ``{"id", "group",
        "swept_axis", "label", "cfg"}`` (see :func:`expand_ofat_scenarios`)
    :type scenarios: list[dict]
    :param seeds: RNG seeds to run each scenario under
    :type seeds: list[int]
    :param steps: Simulation steps per run
    :type steps: int
    :param keep_trajectories_for_seed: Keep full per-tick telemetry only for
        this seed of each scenario (``None`` to keep none), so trajectory
        plots don't require holding every replicate's full history
    :type keep_trajectories_for_seed: int | None
    :return: A run-level DataFrame (one row per scenario x seed) and a
        ``scenario_id -> telemetry`` dict for the kept trajectories
    :rtype: tuple[pd.DataFrame, dict[str, list]]
    """
    records = []
    trajectories: dict[str, list] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for scenario in scenarios:
            for seed in seeds:
                cfg = deep_merge(scenario["cfg"], {"world": {"rng_seed": seed}})
                safe_name = (
                    scenario["id"]
                    .replace("/", "_")
                    .replace(":", "_")
                    .replace(",", "_")
                    .replace("=", "")
                )
                path = os.path.join(tmpdir, f"{safe_name}_{seed}.yaml")
                with open(path, "w") as f:
                    yaml.safe_dump(cfg, f)

                result = execute_run(RunRequest(config_path=path, steps=steps))

                records.append(
                    {
                        "scenario_id": scenario["id"],
                        "scenario_group": scenario["group"],
                        "swept_axis": scenario["swept_axis"],
                        "label": scenario["label"],
                        "seed": seed,
                        "run_id": result.metadata.run_id,
                        "scenario_fingerprint": result.metadata.scenario_fingerprint,
                        "run_spec_fingerprint": result.metadata.run_spec_fingerprint,
                        **result.scenario,
                        **asdict(result.summary),
                    }
                )

                if (
                    keep_trajectories_for_seed is not None
                    and seed == keep_trajectories_for_seed
                ):
                    trajectories[scenario["id"]] = result.telemetry

    return pd.DataFrame(records), trajectories


def between_scenario_variance_share(
    df: pd.DataFrame, metric: str, *, group_col: str = "scenario_id"
) -> float:
    """Fraction of ``metric``'s total variance explained by ``group_col``
    (a one-way ANOVA style decomposition: ``SS_between / SS_total``).

    High means a given group's outcome is reproducible across the other rows
    varying within it (e.g. RNG seeds within a scenario); low means the
    metric is dominated by that within-group variation instead.

    :param df: A run-level table containing ``group_col`` and ``metric``
    :type df: pd.DataFrame
    :param metric: The outcome column to decompose
    :type metric: str
    :param group_col: The grouping column (e.g. ``scenario_id``)
    :type group_col: str
    :return: ``SS_between / SS_total``, or ``nan`` if ``metric`` has zero
        total variance in ``df``
    :rtype: float
    """
    sub = df[[group_col, metric]].dropna()
    grand_mean = sub[metric].mean()
    group_means = sub.groupby(group_col)[metric].mean()
    group_sizes = sub.groupby(group_col)[metric].size()
    ss_between = float((group_sizes * (group_means - grand_mean) ** 2).sum())
    ss_total = float(((sub[metric] - grand_mean) ** 2).sum())
    return ss_between / ss_total if ss_total else float("nan")


def predictability_probe(
    df: pd.DataFrame,
    feature_cols: list[str],
    regression_targets: list[str],
    binary_target: str | None = None,
    *,
    n_splits: int = 5,
    group_col: str = "scenario_fingerprint",
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Grouped-CV linear vs. random-forest predictability probe: for each
    regression target, and optionally one binary target, how well do the
    swept parameters predict the outcome on *held-out scenarios*.

    CV folds are grouped by ``group_col`` (default ``scenario_fingerprint``),
    not shuffled row-level: every seed replicate of a scenario shares an
    identical feature vector (``feature_cols`` comes purely from config, not
    seed-dependent state), so a row-level split would let a model score well
    just by recognizing a scenario it already saw in training rather than
    generalizing to a held-out one. Each target is dropna'd independently, so
    a target with its own missing values (e.g. a convergence tick that's
    undefined for runs that never converge) doesn't shrink the sample used
    for every other target too.

    :param df: A run-level table containing ``feature_cols``, the targets,
        and ``group_col``
    :type df: pd.DataFrame
    :param feature_cols: Scenario-feature columns to predict from (see
        ``simlab.run_analysis.extract_scenario_features`` for what's
        available)
    :type feature_cols: list[str]
    :param regression_targets: Continuous outcome columns to probe
    :type regression_targets: list[str]
    :param binary_target: An optional 0/1-valued outcome column to probe;
        skipped if it's constant in the (dropna'd) data
    :type binary_target: str | None
    :param n_splits: Number of grouped CV folds
    :type n_splits: int
    :param group_col: The grouping column CV folds are held out by
    :type group_col: str
    :param random_state: Random state for the random forest estimators
    :type random_state: int
    :return: A regression results table indexed by target (columns ``n``,
        ``Linear``, ``RandomForest`` -- mean R^2 across folds), and either a
        classification result dict (``target``, ``n``, ``majority_baseline``,
        ``Logistic``, ``RandomForest``) or ``None`` if ``binary_target``
        wasn't given or was constant
    :rtype: tuple[pd.DataFrame, dict[str, Any] | None]
    """
    gkf = GroupKFold(n_splits=n_splits)

    regression_results = []
    for target in regression_targets:
        target_df = df.dropna(subset=feature_cols + [target])
        X = target_df[feature_cols].values
        y = target_df[target].values
        groups = target_df[group_col].values
        lin_r2 = cross_val_score(
            LinearRegression(), X, y, cv=gkf, groups=groups, scoring="r2"
        )
        rf_r2 = cross_val_score(
            RandomForestRegressor(n_estimators=200, random_state=random_state),
            X,
            y,
            cv=gkf,
            groups=groups,
            scoring="r2",
        )
        regression_results.append(
            {
                "target": target,
                "n": len(target_df),
                "Linear": lin_r2.mean(),
                "RandomForest": rf_r2.mean(),
            }
        )
    regression_df = pd.DataFrame(regression_results).set_index("target")

    classification_result: dict[str, Any] | None = None
    if binary_target is not None:
        class_df = df.dropna(subset=feature_cols + [binary_target])
        y_bin = class_df[binary_target].astype(int).values
        if len(set(y_bin)) > 1:
            X_bin = class_df[feature_cols].values
            groups_bin = class_df[group_col].values
            logit_acc = cross_val_score(
                LogisticRegression(max_iter=1000),
                X_bin,
                y_bin,
                cv=gkf,
                groups=groups_bin,
                scoring="accuracy",
            )
            rf_acc = cross_val_score(
                RandomForestClassifier(n_estimators=200, random_state=random_state),
                X_bin,
                y_bin,
                cv=gkf,
                groups=groups_bin,
                scoring="accuracy",
            )
            classification_result = {
                "target": binary_target,
                "n": len(class_df),
                "majority_baseline": max(y_bin.mean(), 1 - y_bin.mean()),
                "Logistic": logit_acc.mean(),
                "RandomForest": rf_acc.mean(),
            }

    return regression_df, classification_result


def plot_variance_shares(shares: dict[str, float]):
    """Bar chart of :func:`between_scenario_variance_share` results.

    :param shares: Metric name -> ``SS_between / SS_total``
    :type shares: dict[str, float]
    :return: The created figure
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(shares.keys()), list(shares.values()), color="tab:blue")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("SS_between / SS_total")
    ax.set_title("Scenario vs seed: share of variance explained by scenario")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return fig


def plot_parameter_response(
    df: pd.DataFrame,
    metric: str,
    *,
    baseline_id: str = "baseline",
    title: str | None = None,
):
    """Small multiples of ``metric`` (mean +- std across seeds) for each
    swept axis's baseline + variants -- one subplot per axis, dynamically
    gridded to fit however many axes ``df`` contains.

    :param df: A run-level table restricted to baseline + OFAT rows,
        containing ``scenario_id``, ``swept_axis``, ``label``, and ``metric``
    :type df: pd.DataFrame
    :param metric: The outcome column to plot
    :type metric: str
    :param baseline_id: Scenario id for the shared baseline row
    :type baseline_id: str
    :param title: Figure title; defaults to a generic description of ``metric``
    :type title: str | None
    :return: The created figure
    """
    axes_list = list(df.swept_axis.dropna().unique())
    n_axes = len(axes_list)
    ncols = min(4, n_axes) or 1
    nrows = -(-n_axes // ncols)  # ceil division
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False
    )
    for ax, axis in zip(axs.flat, axes_list):
        sub = pd.concat([df[df.scenario_id == baseline_id], df[df.swept_axis == axis]])
        grouped = sub.groupby("label")[metric].agg(["mean", "std"])
        grouped = grouped.reindex(
            sorted(grouped.index, key=lambda label: (label != baseline_id, label))
        )
        ax.bar(
            grouped.index,
            grouped["mean"],
            yerr=grouped["std"],
            capsize=3,
            color="tab:orange",
        )
        ax.set_title(axis, fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.set_ylabel(metric)
    for ax in axs.flat[n_axes:]:
        ax.axis("off")
    fig.suptitle(title or f"Parameter response: {metric} by axis")
    fig.tight_layout()
    return fig


def plot_predictability(
    regression_df: pd.DataFrame, classification_result: dict[str, Any] | None = None
):
    """Plot :func:`predictability_probe`'s results: a symlog-scaled R^2 bar
    chart per regression target, plus an accuracy bar chart (with a
    majority-baseline reference line) if a classification result is given.

    R^2 is unbounded below (a model can do worse than predicting the mean),
    while accuracy is bounded to [0, 1] -- separate axes rather than forcing
    both onto one [0, 1] scale, which would hide negative R^2 entirely. Some
    targets' R^2 can be extremely negative while others sit near [-1, 1] --
    symlog keeps the near-zero targets readable instead of flattened by scale.

    :param regression_df: The first return value of :func:`predictability_probe`
    :type regression_df: pd.DataFrame
    :param classification_result: The second return value of
        :func:`predictability_probe`, or ``None`` to omit that panel
    :type classification_result: dict[str, Any] | None
    :return: The created figure
    """
    if classification_result is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    else:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax2 = None

    x = list(range(len(regression_df)))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], regression_df["Linear"], width, label="Linear")
    ax1.bar(
        [i + width / 2 for i in x],
        regression_df["RandomForest"],
        width,
        label="RandomForest",
    )
    ax1.axhline(0, color="gray", linewidth=1)
    ax1.set_yscale("symlog", linthresh=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regression_df.index, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("R^2 (held-out scenarios, symlog)")
    ax1.set_title("Regression targets")
    ax1.legend(fontsize=8)

    if ax2 is not None and classification_result is not None:
        bars_x = [0, 1]
        ax2.bar(
            bars_x,
            [classification_result["Logistic"], classification_result["RandomForest"]],
            color="tab:orange",
        )
        ax2.axhline(
            classification_result["majority_baseline"],
            color="gray",
            linestyle="--",
            linewidth=1,
            label="majority baseline",
        )
        ax2.set_xticks(bars_x)
        ax2.set_xticklabels(["Logistic", "RandomForest"])
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("accuracy (held-out scenarios)")
        ax2.set_title(classification_result["target"])
        ax2.legend(fontsize=8)

    fig.suptitle("Simple vs nonlinear models as landscape probes")
    fig.tight_layout()
    return fig
