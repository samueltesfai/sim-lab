# Simulation Model Design

## Overview

This document describes the current simulation kernel implemented in `sim.py`.

The model is a discrete-time, agent-based simulation of belief dynamics on a directed social network. Agents hold beliefs about claims, accumulate epistemic memories, choose actions based on simple heuristics, and update beliefs from new memories over time.

At a high level, the model combines three information channels:

1. **Observation**: passive, world-driven exposure to truth-anchored evidence
2. **Verification**: active, agent-chosen truth-checking
3. **Hearing**: social transmission from other agents through communication or broadcast

The current implementation provides a configurable simulation core for exploring information diffusion, belief updating, and action selection in networked populations.

---

## Core Entities

### `ActionType`

`ActionType` defines the set of intentional actions an agent may choose:

- `IDLE`
- `VERIFY`
- `COMMUNICATE`
- `BROADCAST`

These are **decision-level constructs**. They represent what an agent is trying to do during a tick.

### `MemoryType`

`MemoryType` defines the set of epistemic inputs that can be stored in memory:

- `OBSERVE`
- `VERIFY`
- `HEAR`

These are **evidence-level constructs**. They represent the source/channel of information that may later affect beliefs.

### `Action`

An `Action` is a parameterized decision object containing:

- `type: ActionType`
- `claim_id: int | None`
- `target_agent_id: int | None`

Validation is performed in `Action.__post_init__()` to ensure action arguments are well-formed.

### `Memory`

A `Memory` is a stored epistemic event containing:

- `id`: index in the agent's memory list
- `type`: `MemoryType`
- `timestamp`: world tick at which the memory was created
- `source`: originating agent id, if applicable
- `claim_id`: claim the memory concerns
- `evidence`: scalar support value in `[0, 1]`

---

## Belief Representation

Each agent stores a belief value for each claim:

- `belief = 0.0` means maximal confidence that the claim is false
- `belief = 0.5` means uncertainty / neutrality
- `belief = 1.0` means maximal confidence that the claim is true

Beliefs are stored as:

```python
self.beliefs: defaultdict[int, float]
```

Unseen claims default to a random value in `[0, 1]` using the agent's private RNG.

This produces heterogeneous initial beliefs without requiring explicit initialization logic.

---

## Trust Representation

Each agent maintains a trust value for other agents:

```python
self.trust: defaultdict[int, float]
```

Trust defaults to `0.5` for any unseen other agent.

Trust currently affects only social hearing (`MemoryType.HEAR`) during belief updates. In the current model, trust controls how much weight a socially received memory has in the learning rule.

---

## Network Representation

The world maintains a directed social graph:

```python
self.network: dict[int, list[int]]
```

If `j` appears in `network[i]`, then agent `i` can directly send information to agent `j`.

The current network generator is intentionally simple:

- random directed edges
- bounded out-degree
- no explicit community structure
- no guarantee of reciprocity

This is a placeholder topology intended for prototyping.

---

## Information Channels

### 1. Observation (`MemoryType.OBSERVE`)

Observation is **world-driven**, not chosen as an intentional action.

During each tick, the world may deliver passive observations to some agents. These observations are generated from objective truth plus observation noise:

```python
base = float(world.truths[claim_id])
noise = world.rng.gauss(0, world.noise[MemoryType.OBSERVE])
evidence = clamp(base + noise)
```

Interpretation:
- observation models ambient, passive exposure to truth-relevant evidence
- observation is less targeted than verification
- observation provides a truth-seeking force in the simulation

### 2. Verification (`ActionType.VERIFY` -> `MemoryType.VERIFY`)

Verification is an **intentional action** chosen by the agent.

When an agent verifies a claim, it generates a new verification memory using objective truth plus verification noise:

```python
base = float(world.truths[claim_id])
noise = world.rng.gauss(0, world.noise[MemoryType.VERIFY])
evidence = clamp(base + noise)
```

Interpretation:
- verification is an active attempt to truth-check a claim
- it is usually modeled as more reliable than passive observation
- it carries a higher action cost than idle and targeted communication

### 3. Hearing (`MemoryType.HEAR`)

Hearing is generated when one agent communicates or broadcasts a claim to another.

Heard evidence is based on the sender's current belief plus social noise:

```python
base = world.get_agent(source).beliefs[claim_id]
noise = world.rng.gauss(0, world.noise[MemoryType.HEAR])
evidence = clamp(base + noise)
```

Interpretation:
- social transmission reflects what the sender currently believes
- the model does not yet include intentional deception
- misinformation can still arise from noisy perception, noisy communication, and incorrect prior beliefs

---

## Agent Action Model

Each agent evaluates candidate actions and selects the one with the highest score.

### Candidate Action Generation

For each tick, an agent considers:

- `IDLE`
- `VERIFY(claim_id)` for each claim
- `BROADCAST(claim_id)` for each claim
- `COMMUNICATE(claim_id, neighbor_id)` for each claim and each outgoing neighbor

This is implemented in:

```python
generate_candidate_actions(world)
```

### Action Scoring

Each action is scored heuristically using the agent's preferences, action costs, and current epistemic state.

#### Helper quantities

For a claim belief `b`:

```python
confidence = abs(b - 0.5) * 2
uncertainty = 1.0 - confidence
```

These satisfy:
- `confidence in [0, 1]`
- `uncertainty in [0, 1]`

#### Disagreement

For claim `c` and neighbor `j`:

```python
disagreement(c, j) = abs(self.beliefs[c] - neighbor.beliefs[c])
```

Local disagreement is the mean disagreement over outgoing neighbors.

#### Current scoring rules

- **VERIFY**
  - high score when uncertainty is high
  - formula:
    ```python
    preference * uncertainty - cost
    ```

- **COMMUNICATE**
  - high score when the agent is confident and the target disagrees
  - formula:
    ```python
    preference * confidence * disagreement - cost
    ```

- **BROADCAST**
  - high score when the agent is confident and many local neighbors disagree
  - formula:
    ```python
    preference * confidence * local_disagreement - cost
    ```

- **IDLE**
  - baseline fallback
  - formula:
    ```python
    preference - cost
    ```

This is not yet formal expected utility. It is a **one-step heuristic desirability model**.

### Action Preferences and Costs

Agents store per-action parameters:

```python
self.action_preference: dict[ActionType, float]
self.action_cost: dict[ActionType, float]
```

These parameters allow heterogeneous behavioral styles without changing the action selection procedure.

Default values are currently:

```python
default_action_preference = {
    ActionType.IDLE: 0.0,
    ActionType.VERIFY: 0.9,
    ActionType.COMMUNICATE: 0.7,
    ActionType.BROADCAST: 0.5,
}

default_action_cost = {
    ActionType.IDLE: 0.0,
    ActionType.VERIFY: 0.35,
    ActionType.COMMUNICATE: 0.15,
    ActionType.BROADCAST: 0.30,
}
```

These should be understood as a baseline prototype regime, not a final calibrated model.

---

## Belief Update Rule

Beliefs are not updated immediately when events occur. Instead:

1. events create memories
2. memories are stored in the agent
3. the agent periodically processes unprocessed memories

This separation is a core design choice. It is also loosely inspired by real models of cognition, where humans do not update their internal understanding directly from raw events alone, but instead form internal representations of those events based on their subjective perspective. 

In this model, memories play that role: they are the agent’s internal record of what was experienced, heard, or verified, and belief updates operate over those stored representations rather than directly over the world.

### Memory backlog

Each agent tracks a memory cursor:

```python
self._mem_cursor
```

This indicates the first unprocessed memory.

### Update equation

For a memory with evidence `e`, claim belief `b`, and effective learning rate `lr`:

```python
b_new = b + lr * (e - b)
```

This is an exponential-moving-average style update toward the new evidence.

### Effective learning rates

Effective learning rate depends on memory type:

- `OBSERVE`: `eta * w_observe`
- `VERIFY`: `eta * w_verify`
- `HEAR`: `eta * w_hear * trust[source]`

Then clamped to `[0, 1]`.

### Current defaults

```python
eta = 0.1
w_observe = 0.6
w_hear = 0.3
w_verify = 1.0
```

Interpretation:
- verification is strongest
- observation is moderate
- social hearing is weakest and trust-modulated

---

## World Tick Semantics

A single tick of `World.step()` currently proceeds as follows:

1. snapshot current beliefs for logging
2. initialize `last_step`
3. deliver passive observations (`deliver_observation`)
4. let each agent choose and execute an intentional action
5. update beliefs for all agents with pending memories
6. snapshot resulting beliefs
7. advance `tick`
8. optionally log metrics

### Observation phase

The world may passively expose agents to observations. Current prototype behavior uses a fixed per-agent observation probability:

```python
if self.rng.random() < 0.1:
    ...
```

This is intentionally simple and should later become configurable.

### Action phase

Each agent selects one action via:

```python
choose_action(world)
```

and executes it via:

```python
act(world, action)
```

### Update phase

After all new memories for the tick have been created, each agent processes any pending memories. This means all epistemic inputs for the tick are accumulated before belief updating occurs.

---

## Current Model Behavior

Qualitatively, the current system exhibits the following patterns:

- agents tend to verify when uncertain
- agents tend to communicate when confident and neighbors disagree
- passive observations provide a truth-seeking background process
- in many runs, the system settles into a low-incentive equilibrium where most agents choose `IDLE`

This appears to happen when:

- uncertainty becomes low enough that verification is not worth the cost
- local disagreement becomes low enough that communication/broadcast are not worth the cost
- residual noise prevents perfect convergence

This should be understood as an emergent equilibrium of the current heuristic scoring model, not as a guaranteed or intended property of all future versions.

---

## Current Limitations

The current kernel intentionally leaves many things simple.

Not yet modeled:

- intentional deception or lying
- dynamic trust updates
- changing network topology
- community-structured graphs
- multi-step planning or expected utility
- claim salience or attention allocation
- heterogeneous world event processes
- memory decay or forgetting
- action budgets / cooldowns / fatigue

These are candidate future extensions, but were deferred in favor of stabilizing the current abstraction layer.
