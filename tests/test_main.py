import pytest
import tempfile
import os
from unittest.mock import patch
from omegaconf import OmegaConf

from simlab.main import main


def create_test_config_file(config_dict: dict) -> str:
    """Create a temporary YAML config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        OmegaConf.save(config_dict, f.name)
        return f.name


def test_main_with_real_config_loading():
    """Test main with real config loading but mocked visualization."""
    config_dict = {
        "world": {
            "num_agents": 2,
            "rng_seed": 42,
            "observation": {"individual_event_rate": 0.0},  # No random observations
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.0, "VERIFY": 0.0},
        },
        "agent": {
            "action_preference": {
                "IDLE": 1.0,
                "VERIFY": 0.0,
                "COMMUNICATE": 0.0,
                "BROADCAST": 0.0,
            },
            "action_cost": {
                "IDLE": 0.0,
                "VERIFY": 0.1,
                "COMMUNICATE": 0.1,
                "BROADCAST": 0.1,
            },
        },
    }

    config_path = create_test_config_file(config_dict)

    try:
        with patch("simlab.main.run_viz") as mock_run_viz:
            with patch(
                "sys.argv",
                ["python -m simlab", "--config", config_path, "--steps", "2"],
            ):
                main()

            # Verify that run_viz was called with a real World object
            mock_run_viz.assert_called_once()
            call_args = mock_run_viz.call_args[0]  # positional args
            world = call_args[0]

            # World should be properly configured
            assert len(world.agents) == 2
            assert world.truths == {0: True}
            # Check keyword args
            call_kwargs = mock_run_viz.call_args[1]
            assert call_kwargs["steps"] == 2

    finally:
        os.unlink(config_path)


def test_main_telemetry_export_integration():
    """Test telemetry export with real file operations."""
    config_dict = {
        "world": {
            "num_agents": 1,
            "rng_seed": 123,
            "observation": {"individual_event_rate": 0.0},
            "truths": {0: True},
            "noise": {"OBSERVE": 0.0, "HEAR": 0.0, "VERIFY": 0.0},
        },
        "agent": {
            "action_preference": {
                "IDLE": 1.0,
                "VERIFY": 0.0,
                "COMMUNICATE": 0.0,
                "BROADCAST": 0.0,
            },
            "action_cost": {
                "IDLE": 0.0,
                "VERIFY": 0.1,
                "COMMUNICATE": 0.1,
                "BROADCAST": 0.1,
            },
        },
    }

    config_path = create_test_config_file(config_dict)

    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, "test.csv")
        jsonl_path = os.path.join(temp_dir, "test.jsonl")

        try:
            with patch("simlab.main.run_viz") as mock_run_viz:
                # Simulate run_viz that actually records some telemetry
                def simulate_run_viz(world, steps, telemetry, **kwargs):
                    # Record initial state
                    telemetry.record_initial(world)
                    # Simulate a few steps
                    for _ in range(steps):
                        snapshot = world.step()
                        telemetry.record(snapshot, world)

                mock_run_viz.side_effect = simulate_run_viz

                with patch(
                    "sys.argv",
                    [
                        "python -m simlab",
                        "--config",
                        config_path,
                        "--steps",
                        "3",
                        "--export-telemetry-csv",
                        csv_path,
                        "--export-telemetry-jsonl",
                        jsonl_path,
                    ],
                ):
                    main()

                # Files should be created
                assert os.path.exists(csv_path)
                assert os.path.exists(jsonl_path)

                # Check file contents
                with open(csv_path, "r") as f:
                    csv_content = f.read()
                    assert "tick" in csv_content
                    assert "belief_mean" in csv_content

                with open(jsonl_path, "r") as f:
                    jsonl_lines = f.readlines()
                    assert len(jsonl_lines) >= 3  # Should have initial + 3 steps
                    for line in jsonl_lines:
                        assert '"tick"' in line

        finally:
            os.unlink(config_path)


@patch("simlab.main.run_viz")
@patch("simlab.main.load_config")
def test_main_handles_run_viz_exceptions(mock_load_config, mock_run_viz):
    """Test that main properly handles exceptions from run_viz."""
    mock_cfg = OmegaConf.create(
        {
            "world": {
                "num_agents": 1,
                "rng_seed": 42,
                "observation": {"individual_event_rate": 0.1},
                "truths": {0: True},
                "noise": {"OBSERVE": 0.0, "HEAR": 0.0, "VERIFY": 0.0},
            },
            "agent": {
                "action_preference": {"IDLE": 0.0},
                "action_cost": {"IDLE": 0.0},
            },
        }
    )
    mock_load_config.return_value = mock_cfg

    # Test that exceptions from run_viz are propagated
    mock_run_viz.side_effect = RuntimeError("Visualization error")

    with patch("sys.argv", ["python -m simlab", "--steps", "1"]):
        with pytest.raises(RuntimeError, match="Visualization error"):
            main()
