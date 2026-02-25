# Simulation Lab: Social Network Behavior Experiments

This repository is a simulation lab designed to study and experiment with the behaviors of social networks. It provides a working simulation kernel and a visualization system to observe the dynamics of agents interacting within a social network. The goal is to explore and experiment with different simulation rules to see if interesting behaviors emerge.

---

## Features

### 1. **Simulation Kernel**
The simulation kernel models a social network where agents:
- **Observe** claims with noisy evidence.
- **Communicate** beliefs to other agents.
- **Verify** claims with direct evidence.
- **Update** their beliefs based on interactions and trust levels.

The kernel supports:
- Configurable agent behaviors and interaction rules.
- Dynamic belief updates based on memory and evidence.
- A simple event-driven simulation loop.

### 2. **Visualization System**
The visualization system renders the social network in real-time, allowing you to observe:
- Agent positions and belief states.
- Active edges representing communication events.
- Overlays for observed, verified, and heard agents.
- Interactive tooltips for detailed agent information.

### 3. **Extensibility**
The simulation is designed to be modular:
- Add new agent behaviors or interaction rules.
- Customize the visualization components.
- Experiment with different network structures and layouts.

---

## Getting Started

### 1. **Set Up the Environment**
Install the required dependencies using Conda:

```bash
conda env create -f environment.yml
conda activate base
```

### 2. Run the Simulation
Run the simulation with visualization:

```python
python run_viz.py
```

You can customize the simulation parameters using command-line arguments:

```python
python run_viz.py --num-agents 20 --steps 1000 --rng-seed 123
```

### 3. Experiment with Rules
Modify the simulation logic in `sim.py` to experiment with:

- Agent decision-making rules.
- Communication dynamics.
- Trust and memory mechanisms.

---

## Next Steps
The current focus is on experimenting with the simulation rules to observe emergent behaviors. Some ideas to explore:

- How do different trust dynamics affect belief propagation?
- What happens when agents have varying levels of noise in their observations?
- How does the network structure influence the spread of information?
