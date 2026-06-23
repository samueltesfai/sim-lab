import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch

from simlab.sim import World, Agent, Snapshot
from simlab.telemetry import Telemetry
from simlab.viz.scene import Scene, build_scene
from simlab.viz.view_model import compute_viewmodel
from simlab.viz.network_viz import NetworkViz
from simlab.viz import run_viz


def _build_test_world(n_agents: int = 3, n_claims: int = 2) -> World:
    """Helper to build a test world with specified number of agents and claims."""
    agents = [Agent(i, rng_seed=i) for i in range(n_agents)]
    truths = {i: (i % 2 == 0) for i in range(n_claims)}  # Alternate true/false
    return World(agents=agents, truths=truths, rng_seed=42)


def _build_test_snapshot(world: World, tick: int = 0) -> Snapshot:
    """Helper to build a test snapshot."""
    return Snapshot(
        tick=tick,
        observation_event_count=2,
        observed_ids=[0, 1],
        verified_ids=[2],
        communicate_edges=[(0, 1), (1, 2)],
        broadcast_edges=[(0, 1), (0, 2)],
        n_agent_updates=3,
        agent_beliefs=world.get_agent_beliefs_snapshot(),
        agent_memory_sizes={0: 5, 1: 3, 2: 4},
    )


def test_build_scene_basic():
    """Test basic scene building."""
    world = _build_test_world(3, 2)

    scene = build_scene(world, layout_seed=42)

    # Check graph structure
    assert isinstance(scene.G, nx.DiGraph)
    assert set(scene.G.nodes()) == {0, 1, 2}
    assert len(scene.G.edges()) <= 6  # Max possible directed edges

    # Check positions
    assert isinstance(scene.pos, dict)
    assert set(scene.pos.keys()) == {0, 1, 2}
    for node_id, pos in scene.pos.items():
        assert hasattr(pos, "__len__") and len(pos) == 2  # numpy array or tuple
        assert all(isinstance(coord, (int, float)) for coord in pos)

    # Check nodes
    assert scene.nodes == [0, 1, 2]

    # Check degrees
    assert isinstance(scene.degrees, dict)
    assert set(scene.degrees.keys()) == {0, 1, 2}
    for degree in scene.degrees.values():
        assert isinstance(degree, int)
        assert degree >= 0

    # Check sizes_base
    assert isinstance(scene.sizes_base, np.ndarray)
    assert len(scene.sizes_base) == 3
    assert all(size >= 200 for size in scene.sizes_base)  # Base size + degree bonus

    # Check truths
    assert scene.truths == world.truths


def test_build_scene_deterministic_layout():
    """Test that scene building is deterministic with same seed."""
    world = _build_test_world(3, 2)

    scene1 = build_scene(world, layout_seed=123)
    scene2 = build_scene(world, layout_seed=123)

    # Should be identical (compare arrays properly)
    for node_id in scene1.pos:
        assert np.array_equal(scene1.pos[node_id], scene2.pos[node_id])
    assert np.array_equal(scene1.sizes_base, scene2.sizes_base)


def test_build_scene_different_layouts():
    """Test that different seeds produce different layouts."""
    world = _build_test_world(3, 2)

    scene1 = build_scene(world, layout_seed=123)
    scene2 = build_scene(world, layout_seed=456)

    # Should have different positions (very unlikely to be the same)
    positions_equal = all(
        np.array_equal(scene1.pos[node_id], scene2.pos[node_id])
        for node_id in scene1.pos
    )
    assert not positions_equal


def test_build_scene_size_calculation():
    """Test that node sizes are calculated correctly based on degree."""
    world = _build_test_world(3, 2)

    # Force specific network structure for testing
    world.network = {
        0: [1, 2],  # degree 2
        1: [0],  # degree 1
        2: [],  # degree 0
    }

    scene = build_scene(world, layout_seed=42)

    # Check sizes based on out-degree
    expected_size_0 = 200 + 120 * 2  # degree 2
    expected_size_1 = 200 + 120 * 1  # degree 1
    expected_size_2 = 200 + 120 * 0  # degree 0

    # Map sizes to nodes
    size_map = {node: size for node, size in zip(scene.nodes, scene.sizes_base)}

    assert size_map[0] == expected_size_0
    assert size_map[1] == expected_size_1
    assert size_map[2] == expected_size_2


def test_compute_viewmodel_basic():
    """Test basic viewmodel computation."""
    world = _build_test_world(3, 2)
    scene = build_scene(world, layout_seed=42)
    snapshot = _build_test_snapshot(world, tick=3)

    vm = compute_viewmodel(scene, claim_id=0, step_snapshot=snapshot)

    # Check basic fields
    assert vm.tick == 3
    assert vm.claim_id == 0
    assert vm.truth_bool == world.truths[0]

    # Check beliefs are filtered by claim_id
    assert set(vm.beliefs.keys()) == {0, 1, 2}
    for agent_id, belief in vm.beliefs.items():
        assert belief == snapshot.agent_beliefs[agent_id][0]

    # Check agent memory sizes
    assert vm.agent_memory_sizes == snapshot.agent_memory_sizes

    # Check event tracking
    assert vm.observed_ids == snapshot.observed_ids
    assert vm.verified_ids == snapshot.verified_ids
    assert vm.communicate_edges == snapshot.communicate_edges
    assert vm.broadcast_edges == snapshot.broadcast_edges

    # Check edge processing
    expected_active_edges = list(
        dict.fromkeys(snapshot.communicate_edges + snapshot.broadcast_edges)
    )
    assert vm.active_edges == expected_active_edges

    # Check receiver identification
    expected_receivers = list({r for (_s, r) in expected_active_edges})
    assert vm.heard_receivers == expected_receivers

    expected_comm_receivers = list({r for (_s, r) in snapshot.communicate_edges})
    assert vm.communicate_receivers == expected_comm_receivers

    expected_bcast_receivers = list({r for (_s, r) in snapshot.broadcast_edges})
    assert vm.broadcast_receivers == expected_bcast_receivers

    # Check position inheritance
    assert vm.pos == scene.pos


def test_compute_viewmodel_stats():
    """Test belief statistics computation."""
    world = _build_test_world(3, 2)
    scene = build_scene(world, layout_seed=42)

    # Create snapshot with specific beliefs for claim 0
    agent_beliefs = {0: {0: 0.2, 1: 0.8}, 1: {0: 0.6, 1: 0.4}, 2: {0: 0.9, 1: 0.1}}

    snapshot = Snapshot(
        tick=0,
        observation_event_count=0,
        observed_ids=[],
        verified_ids=[],
        communicate_edges=[],
        broadcast_edges=[],
        n_agent_updates=0,
        agent_beliefs=agent_beliefs,
        agent_memory_sizes={0: 0, 1: 0, 2: 0},
    )

    vm = compute_viewmodel(scene, claim_id=0, step_snapshot=snapshot)

    # Check statistics
    beliefs_for_claim = [0.2, 0.6, 0.9]
    expected_mean = sum(beliefs_for_claim) / len(beliefs_for_claim)
    expected_min = min(beliefs_for_claim)
    expected_max = max(beliefs_for_claim)

    assert vm.stats["mean"] == expected_mean
    assert vm.stats["min"] == expected_min
    assert vm.stats["max"] == expected_max


def test_compute_viewmodel_empty_beliefs():
    """Test viewmodel computation with empty beliefs."""
    world = _build_test_world(0, 1)  # No agents
    scene = build_scene(world, layout_seed=42)

    snapshot = Snapshot(
        tick=0,
        observation_event_count=0,
        observed_ids=[],
        verified_ids=[],
        communicate_edges=[],
        broadcast_edges=[],
        n_agent_updates=0,
        agent_beliefs={},
        agent_memory_sizes={},
    )

    vm = compute_viewmodel(scene, claim_id=0, step_snapshot=snapshot)

    # Should handle empty case gracefully
    assert vm.beliefs == {}
    assert vm.stats["mean"] == 0.0
    assert vm.stats["min"] == 0.0
    assert vm.stats["max"] == 0.0


def test_compute_viewmodel_missing_claim():
    """Test viewmodel computation when claim doesn't exist in agent beliefs."""
    world = _build_test_world(2, 1)
    scene = build_scene(world, layout_seed=42)

    # Agent beliefs only for claim 0, but we request claim 1
    agent_beliefs = {0: {0: 0.5}, 1: {0: 0.7}}

    snapshot = Snapshot(
        tick=0,
        observation_event_count=0,
        observed_ids=[],
        verified_ids=[],
        communicate_edges=[],
        broadcast_edges=[],
        n_agent_updates=0,
        agent_beliefs=agent_beliefs,
        agent_memory_sizes={0: 0, 1: 0},
    )

    vm = compute_viewmodel(scene, claim_id=1, step_snapshot=snapshot)

    # Should default to 0.0 for missing claims
    assert vm.beliefs == {0: 0.0, 1: 0.0}
    assert vm.stats["mean"] == 0.0
    assert vm.stats["min"] == 0.0
    assert vm.stats["max"] == 0.0


def test_compute_viewmodel_duplicate_edge_removal():
    """Test that duplicate edges are removed from active_edges."""
    world = _build_test_world(3, 1)
    scene = build_scene(world, layout_seed=42)

    # Create snapshot with duplicate edges
    snapshot = Snapshot(
        tick=0,
        observation_event_count=0,
        observed_ids=[],
        verified_ids=[],
        communicate_edges=[(0, 1), (0, 1), (1, 2)],  # Duplicate (0,1)
        broadcast_edges=[(0, 2), (1, 2), (0, 2)],  # Duplicate (0,2)
        n_agent_updates=0,
        agent_beliefs=world.get_agent_beliefs_snapshot(),
        agent_memory_sizes={0: 0, 1: 0, 2: 0},
    )

    vm = compute_viewmodel(scene, claim_id=0, step_snapshot=snapshot)

    # Should remove duplicates while preserving order
    expected_active_edges = [(0, 1), (1, 2), (0, 2)]
    assert vm.active_edges == expected_active_edges


@patch("simlab.viz.network_viz.plt")
def test_network_viz_initialization(mock_plt):
    """Test NetworkViz initialization with mocked matplotlib."""
    world = _build_test_world(3, 2)

    # Mock plt.subplots
    mock_fig = Mock()
    mock_ax = Mock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    viz = NetworkViz(world, claim_id=1, layout_seed=123)

    assert viz.claim_id == 1
    assert viz.scene is not None
    assert isinstance(viz.scene, Scene)
    assert viz.fig is mock_fig
    assert viz.ax is mock_ax
    assert viz._initialized is False
    assert len(viz.components) > 0


@patch("simlab.viz.network_viz.plt")
def test_run_viz_basic(mock_plt):
    """Test run_viz function with mocked matplotlib."""
    world = _build_test_world(2, 1)
    telemetry = Telemetry()

    # Mock matplotlib functions
    mock_plt.ion = Mock()
    mock_plt.show = Mock()
    mock_plt.pause = Mock()
    mock_plt.ioff = Mock()
    mock_plt.subplots.return_value = (Mock(), Mock())

    # Run for a few steps
    run_viz(
        world,
        steps=3,
        claim_id=0,
        telemetry=telemetry,
        log_every=10,  # Don't log during test
        draw_every=10,  # Don't draw during test
    )

    # Should have recorded telemetry
    assert len(telemetry.history) >= 3

    # Check matplotlib was used (but don't assert specific call counts)
    mock_plt.ion.assert_called_once()
    mock_plt.show.assert_called()
    mock_plt.ioff.assert_called_once()


def test_run_viz_validation():
    """Test run_viz parameter validation."""
    world = _build_test_world(2, 1)
    telemetry = Telemetry()

    # Test invalid log_every
    with pytest.raises(ValueError, match="log_every must be >= 1"):
        run_viz(world, steps=1, telemetry=telemetry, log_every=0)

    # Test invalid draw_every
    with pytest.raises(ValueError, match="draw_every must be >= 1"):
        run_viz(world, steps=1, telemetry=telemetry, draw_every=0)


@patch("simlab.viz.network_viz.plt")
def test_component_structure(mock_plt):
    """Test component structure and basic functionality."""
    try:
        world = _build_test_world(3, 2)

        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        viz = NetworkViz(world, claim_id=0, layout_seed=42)

        # Check that components were created
        assert len(viz.components) > 0

        # Check component types (basic validation)
        component_types = [type(comp).__name__ for comp in viz.components]
        expected_components = ["BaseEdges", "Nodes", "ActiveEdges", "ActiveEdges"]

        # Should have at least the expected components
        for expected in expected_components:
            assert expected in component_types

    except ImportError as e:
        pytest.skip(f"Component testing failed: {e}")


# Test existing viz tests from test_telemetry.py (moved here for organization)
def test_compute_viewmodel_requires_snapshot():
    """Test that compute_viewmodel requires a non-None Snapshot."""
    world = _build_test_world(3)
    scene = build_scene(world)

    # Should raise TypeError or AttributeError when snapshot is None
    with pytest.raises((TypeError, AttributeError)):
        compute_viewmodel(scene, claim_id=0, step_snapshot=None)


def test_compute_viewmodel_with_telemetry_row():
    """Test that compute_viewmodel accepts telemetry_row parameter."""
    world = _build_test_world(3)
    scene = build_scene(world)
    snapshot = _build_test_snapshot(world)
    telemetry = Telemetry()
    telemetry_row = telemetry.record(snapshot, world)

    # Should work with telemetry_row (even if unused currently)
    vm = compute_viewmodel(
        scene, claim_id=0, step_snapshot=snapshot, telemetry_row=telemetry_row
    )

    assert vm.tick == snapshot.tick
    assert vm.claim_id == 0
