# Simulation Lab: Social Network Behavior Experiments

<p align="center">
  <img src="docs/assets/demo.gif" alt="Simulation demo">
</p>

This repository is a simulation lab for studying belief dynamics and information diffusion in directed social networks. It provides a simulation kernel, telemetry system, and live visualization for observing how agents interact, exchange information, and update beliefs over time.

---

## Features

### 1. **Simulation Kernel**
The simulation kernel models a directed social network where agents:

- **Receive passive observations** from the world with noisy evidence
- **Communicate** beliefs to specific neighbors
- **Broadcast** beliefs to all outgoing neighbors
- **Verify** claims with direct evidence
- **Update** beliefs from accumulated memories using trust-weighted learning

The kernel supports:

- Configurable world and agent parameters through YAML config files
- Dynamic belief updates based on memory and evidence
- Per-step snapshots capturing full agent belief state, memory sizes, and event data

### 2. **Telemetry System**
The telemetry system tracks simulation metrics across all agents and claims:

- Global belief statistics (mean, std, min, max)
- Truth-alignment metrics against known world claims
- Belief change deltas between steps
- Event counts (observations, verifications, communications, broadcasts)
- Step runtime measurement
- Initial baseline recording before the first simulation step
- Export to CSV and JSONL formats

### 3. **Visualization System**
The visualization system renders the social network in real time, allowing you to observe:

- Agent positions and belief states
- Directed communication and broadcast edges
- Overlays for observed, verified, and receiving agents
- Interactive tooltips for detailed agent information

### 4. **Configuration**
Simulation scenarios are defined in `configs/` and loaded through `config.py`.

See the [model documentation](docs/model.md) for the simulation design, and the
[configuration reference](docs/config.md) for the YAML schema and CLI options.

---

## Getting Started

Install dependencies and run:

```bash
pip install -e .
python -m simlab
```

Run tests:

```bash
pytest
```

Pass `-h` for the full list of CLI flags.

---

## Next Steps

Planned directions for the project include:

- Adding more scenario presets and config-driven experiments
- Improving telemetry metrics for action selection and belief alignment
- Writing notebook-based case studies for specific simulation runs
- Extending the decision model toward explicit utility or expected utility
- Investigating richer trust dynamics and misinformation behaviors
