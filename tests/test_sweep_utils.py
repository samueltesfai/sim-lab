from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

# sweep_utils imports scikit-learn at module level (it's meant to be loaded
# only from a notebook context, where the "notebook" uv dependency group is
# already installed) -- skip this whole file gracefully rather than fail
# collection when running the base test suite without that group.
pytest.importorskip("sklearn")

from simlab.runner import SCHEMA_VERSION  # noqa: E402
from sweep_utils import (  # noqa: E402
    between_scenario_variance_share,
    dedupe_scenarios,
    expand_ofat_scenarios,
    plot_parameter_response,
    plot_predictability,
    plot_variance_shares,
    predictability_probe,
    reproducibility_header,
    run_sweep,
)

BASELINE_CFG = {
    "world": {
        "rng_seed": 0,
        "truths": {0: True},
        "noise": {"OBSERVE": 0.0, "HEAR": 0.0, "VERIFY": 0.0},
        "observation": {"private_event_rate": 0.1, "global_event_rate": 0.0},
    },
    "agent": {
        "defaults": {
            "action_preference": {
                "IDLE": 0.0,
                "VERIFY": 0.9,
                "COMMUNICATE": 0.7,
                "BROADCAST": 0.5,
            },
            "action_cost": {
                "IDLE": 0.0,
                "VERIFY": 0.35,
                "COMMUNICATE": 0.15,
                "BROADCAST": 0.30,
            },
        },
        "profiles": [{"name": "default", "count": 5}],
    },
}

ATTENTION_AXIS = {
    "attention": [
        ("low", {"agent": {"defaults": {"observation": {"attention": 0.2}}}}),
        ("high", {"agent": {"defaults": {"observation": {"attention": 1.0}}}}),
    ],
}


# ---------------------------------------------------------------------------
# expand_ofat_scenarios
# ---------------------------------------------------------------------------


def test_expand_ofat_scenarios_includes_baseline_and_variants():
    scenarios = expand_ofat_scenarios(BASELINE_CFG, ATTENTION_AXIS)

    assert [s["id"] for s in scenarios] == [
        "baseline",
        "attention:low",
        "attention:high",
    ]
    assert scenarios[0]["group"] == "baseline"
    assert scenarios[0]["swept_axis"] is None
    assert scenarios[1]["group"] == "ofat"
    assert scenarios[1]["swept_axis"] == "attention"
    assert scenarios[1]["cfg"]["agent"]["defaults"]["observation"]["attention"] == 0.2


def test_expand_ofat_scenarios_empty_axes_returns_only_baseline():
    scenarios = expand_ofat_scenarios(BASELINE_CFG, {})
    assert len(scenarios) == 1
    assert scenarios[0]["id"] == "baseline"


def test_expand_ofat_scenarios_does_not_mutate_baseline_cfg():
    assert "observation" not in BASELINE_CFG["agent"]["defaults"]

    expand_ofat_scenarios(BASELINE_CFG, ATTENTION_AXIS)

    assert "observation" not in BASELINE_CFG["agent"]["defaults"]


# ---------------------------------------------------------------------------
# dedupe_scenarios
# ---------------------------------------------------------------------------


def test_dedupe_scenarios_drops_duplicate_resolved_config():
    # "attention:high" sets attention=1.0 explicitly, which is also the
    # schema default baseline gets by leaving it unset -- same resolved
    # config, so it's a genuine duplicate of "baseline".
    scenarios = expand_ofat_scenarios(BASELINE_CFG, ATTENTION_AXIS)

    kept, duplicate_of = dedupe_scenarios(scenarios)

    assert [s["id"] for s in kept] == ["baseline", "attention:low"]
    assert duplicate_of == {"attention:high": "baseline"}


def test_dedupe_scenarios_no_duplicates_returns_all_and_empty_map():
    scenarios = [
        {
            "id": "baseline",
            "group": "baseline",
            "swept_axis": None,
            "label": "baseline",
            "cfg": BASELINE_CFG,
        }
    ]

    kept, duplicate_of = dedupe_scenarios(scenarios)

    assert kept == scenarios
    assert duplicate_of == {}


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------


def test_run_sweep_returns_one_row_per_scenario_seed_pair():
    scenarios = expand_ofat_scenarios(BASELINE_CFG, ATTENTION_AXIS)

    runs, trajectories = run_sweep(scenarios, seeds=[0, 1], steps=3)

    assert len(runs) == 6  # 3 scenarios x 2 seeds
    assert set(runs["scenario_id"]) == {"baseline", "attention:low", "attention:high"}
    assert set(runs["seed"]) == {0, 1}
    assert "final_mean_truth_error" in runs.columns
    assert "scenario_fingerprint" in runs.columns
    # Only seed 0's trajectory is kept by default.
    assert set(trajectories.keys()) == {"baseline", "attention:low", "attention:high"}
    assert len(trajectories["baseline"]) == 4  # initial row + 3 steps


def test_run_sweep_keep_trajectories_for_seed_none_keeps_nothing():
    scenarios = expand_ofat_scenarios(BASELINE_CFG, {})

    _, trajectories = run_sweep(
        scenarios, seeds=[0], steps=2, keep_trajectories_for_seed=None
    )

    assert trajectories == {}


# ---------------------------------------------------------------------------
# between_scenario_variance_share
# ---------------------------------------------------------------------------


def test_between_scenario_variance_share_all_between():
    # No within-group variance: the metric is fully determined by the group.
    df = pd.DataFrame(
        {"scenario_id": ["a", "a", "b", "b"], "metric": [1.0, 1.0, 3.0, 3.0]}
    )
    assert between_scenario_variance_share(df, "metric") == pytest.approx(1.0)


def test_between_scenario_variance_share_all_within():
    # Same mean per group: the metric doesn't depend on the group at all.
    df = pd.DataFrame(
        {"scenario_id": ["a", "a", "b", "b"], "metric": [1.0, 3.0, 1.0, 3.0]}
    )
    assert between_scenario_variance_share(df, "metric") == pytest.approx(0.0)


def test_between_scenario_variance_share_nan_when_no_variance():
    df = pd.DataFrame({"scenario_id": ["a", "a"], "metric": [2.0, 2.0]})
    assert math.isnan(between_scenario_variance_share(df, "metric"))


def test_between_scenario_variance_share_custom_group_col():
    df = pd.DataFrame({"config": ["a", "a", "b", "b"], "metric": [1.0, 1.0, 3.0, 3.0]})
    assert between_scenario_variance_share(
        df, "metric", group_col="config"
    ) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# reproducibility_header
# ---------------------------------------------------------------------------


def test_reproducibility_header_returns_expected_keys(capsys):
    header = reproducibility_header("test-experiment")

    assert header["experiment_name"] == "test-experiment"
    assert isinstance(header["git_commit"], str) and len(header["git_commit"]) == 40
    assert isinstance(header["git_dirty"], bool)
    assert isinstance(header["kernel_path_clean"], bool)
    assert isinstance(header["kernel_commit_on_main"], bool)
    assert header["schema_version"] == SCHEMA_VERSION

    captured = capsys.readouterr()
    assert "Notebook reproducibility header" in captured.out
    assert "test-experiment" in captured.out


# ---------------------------------------------------------------------------
# predictability_probe (requires the optional "notebook" dependency group)
# ---------------------------------------------------------------------------


def _synthetic_scenario_rows(
    n_scenarios: int = 10, seeds_per_scenario: int = 3
) -> pd.DataFrame:
    rows = []
    for i in range(n_scenarios):
        for seed in range(seeds_per_scenario):
            rows.append(
                {
                    "scenario_fingerprint": f"scenario-{i}",
                    "x": float(i),
                    "y": 2 * float(i) + 0.01 * seed,
                    "flag": 1 if i % 2 == 0 else 0,
                }
            )
    return pd.DataFrame(rows)


def test_predictability_probe_regression_and_classification_shapes():
    df = _synthetic_scenario_rows()

    regression_df, classification = predictability_probe(
        df,
        feature_cols=["x"],
        regression_targets=["y"],
        binary_target="flag",
        n_splits=5,
    )

    assert list(regression_df.index) == ["y"]
    assert regression_df.loc["y", "n"] == 30
    assert {"Linear", "RandomForest"}.issubset(regression_df.columns)
    assert classification is not None
    assert classification["target"] == "flag"
    assert classification["n"] == 30
    assert 0.0 <= classification["majority_baseline"] <= 1.0


def test_predictability_probe_skips_constant_binary_target():
    df = _synthetic_scenario_rows()
    df["flag"] = 1  # constant -- nothing for a classifier to learn

    _, classification = predictability_probe(
        df, feature_cols=["x"], regression_targets=["y"], binary_target="flag"
    )

    assert classification is None


def test_predictability_probe_no_binary_target_skips_classification():
    df = _synthetic_scenario_rows()

    _, classification = predictability_probe(
        df, feature_cols=["x"], regression_targets=["y"]
    )

    assert classification is None


def test_predictability_probe_drops_na_per_target_independently():
    df = _synthetic_scenario_rows()
    df.loc[df["x"] == 0.0, "z"] = None
    df.loc[df["x"] != 0.0, "z"] = df.loc[df["x"] != 0.0, "y"]

    regression_df, _ = predictability_probe(
        df, feature_cols=["x"], regression_targets=["y", "z"]
    )

    assert regression_df.loc["y", "n"] == 30
    assert regression_df.loc["z", "n"] == 27  # one scenario's 3 seed-rows dropped


# ---------------------------------------------------------------------------
# plotting helpers -- smoke tests: they must run without error and return a
# usable figure, not reproduce matplotlib's own correctness.
# ---------------------------------------------------------------------------


def test_plot_variance_shares_returns_figure():
    fig = plot_variance_shares({"metric_a": 0.9, "metric_b": 0.2})
    assert fig is not None
    plt.close(fig)


def test_plot_parameter_response_returns_figure():
    df = pd.DataFrame(
        {
            "scenario_id": ["baseline", "baseline", "attention:low", "attention:low"],
            "swept_axis": [None, None, "attention", "attention"],
            "label": ["baseline", "baseline", "low", "low"],
            "metric": [0.1, 0.12, 0.3, 0.28],
        }
    )
    fig = plot_parameter_response(df, "metric")
    assert fig is not None
    plt.close(fig)


def test_plot_predictability_with_and_without_classification():
    regression_df = pd.DataFrame(
        {"n": [10], "Linear": [0.5], "RandomForest": [0.6]},
        index=pd.Index(["y"], name="target"),
    )

    fig1 = plot_predictability(regression_df)
    assert fig1 is not None
    plt.close(fig1)

    classification = {
        "target": "flag",
        "n": 10,
        "majority_baseline": 0.6,
        "Logistic": 0.5,
        "RandomForest": 0.55,
    }
    fig2 = plot_predictability(regression_df, classification)
    assert fig2 is not None
    plt.close(fig2)
