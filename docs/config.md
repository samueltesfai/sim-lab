# Configuration Reference

Simulation scenarios are defined as YAML files and loaded through `simlab.config`.

A config describes two things:

1. the **world / environment** the agents inhabit
2. the **agent population** that inhabits that world

The `world` section controls truths, noise, the random seed, and passive
observation event rates. The `agent` section controls default cognition/action
parameters and the concrete profiles (subpopulations) that make up the run.

The canonical config format uses exactly three top-level pieces:

- `world`
- `agent.defaults`
- `agent.profiles`

There is no flat agent form and no separate `world.num_agents`; the population
size is the sum of the profile counts.

For the conceptual meaning of these parameters (what observation, trust,
learning, etc. actually *do*), see [`docs/model.md`](model.md). This document is
the schema/reference for writing a scenario.

## Table of Contents

- [Configuration Reference](#configuration-reference)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Minimal Config](#minimal-config)
  - [Top-Level Schema](#top-level-schema)
  - [`world`](#world)
    - [`world.rng_seed`](#worldrng_seed)
    - [`world.truths`](#worldtruths)
    - [`world.noise`](#worldnoise)
    - [`world.observation`](#worldobservation)
      - [`private_event_rate`](#private_event_rate)
      - [`global_event_rate`](#global_event_rate)
  - [`agent`](#agent)
    - [`agent.defaults`](#agentdefaults)
      - [`observation.attention`](#observationattention)
      - [`observation.bias`](#observationbias)
      - [`trust.default`](#trustdefault)
      - [`learning.rate`](#learningrate)
      - [`learning.observe_weight`](#learningobserve_weight)
      - [`learning.hear_weight`](#learninghear_weight)
      - [`learning.verify_weight`](#learningverify_weight)
      - [`action_preference`](#action_preference)
      - [`action_cost`](#action_cost)
    - [`agent.profiles`](#agentprofiles)
      - [`name`](#name)
      - [`count`](#count)
      - [profile overrides and deep-merge](#profile-overrides-and-deep-merge)
  - [Validation Rules](#validation-rules)
  - [Example: Homogeneous Population](#example-homogeneous-population)
  - [Example: Heterogeneous Population](#example-heterogeneous-population)

## Overview

Configs are loaded with `load_config(path)`, which parses the YAML and runs
`validate_config`. `build_world(cfg)` then expands `agent.defaults` +
`agent.profiles` into concrete `Agent` instances and constructs the `World`.

```python
from simlab.config import load_config, build_world

cfg = load_config("configs/heterogeneous.yaml")
world = build_world(cfg)
```

## Minimal Config

The smallest valid scenario: one true claim, no observation noise, and a single
homogeneous profile.

```yaml
world:
  rng_seed: 0
  observation:
    private_event_rate: 0.1
    global_event_rate: 0.0
  truths:
    0: true
  noise:
    OBSERVE: 0.0
    HEAR: 0.0
    VERIFY: 0.0

agent:
  defaults: {}
  profiles:
    - name: default
      count: 10
```

Any agent parameter omitted from `agent.defaults` falls back to the built-in
`Agent` defaults documented under [`agent.defaults`](#agentdefaults).

## Top-Level Schema

```yaml
world:
  rng_seed: <int>
  truths: { <claim_id:int>: <bool>, ... }
  noise: { OBSERVE: <float>, HEAR: <float>, VERIFY: <float> }
  observation:
    private_event_rate: <float>
    global_event_rate: <float>

agent:
  defaults: { <agent settings> }
  profiles:
    - name: <str>
      count: <int>
      <agent settings overrides>
```

All of `world`, `agent.defaults`, and `agent.profiles` are required.

## `world`

Describes the environment shared by all agents.

### `world.rng_seed`

Integer seed for the world's RNG. Controls network generation and world-side
event/noise draws.

Each agent is given its own derived seed of `rng_seed + agent_index + 1`, so a
single `world.rng_seed` makes the entire run reproducible while still giving
each agent an independent stream.

### `world.truths`

A mapping of `claim_id -> boolean` ground truth. Keys are integer claim IDs;
values **must** be booleans.

```yaml
truths:
  0: true
  1: false
```

Internally a `true`/`false` truth is treated as `1.0`/`0.0` when generating
truth-grounded evidence. The set of claim IDs here defines the claims that exist
in the simulation.

### `world.noise`

Standard deviation of the Gaussian noise added to each evidence channel. All
three keys are required and must be non-negative.

| Key       | Applies to                          |
| --------- | ----------------------------------- |
| `OBSERVE` | observation event evidence          |
| `VERIFY`  | verification evidence               |
| `HEAR`    | social (heard) evidence             |

```yaml
noise:
  OBSERVE: 0.1
  HEAR: 0.15
  VERIFY: 0.05
```

Observation noise is applied once, by the world, when an event is generated;
agents do not add a second perceptual noise term (they apply only a systematic
`observation.bias`). See the Observation channel in [`docs/model.md`](model.md).

### `world.observation`

Controls how often the world emits passive observation events.

#### `private_event_rate`

Required float in `[0, 1]`. The **per-agent, per-tick** probability that the
world emits a private observation event visible only to that agent. With `N`
agents the expected number of private events per tick is `N * private_event_rate`.

#### `global_event_rate`

Required float in `[0, 1]`. The **per-tick** probability that the world emits a
single global observation event visible to *every* agent. Set to `0.0` to
disable shared events.

```yaml
observation:
  private_event_rate: 0.1
  global_event_rate: 0.05
```

Visibility is not the same as perception: a visible agent still only forms a
memory if it notices the event (governed by its `observation.attention`).

## `agent`

Describes the agent population. Requires `defaults` and `profiles`.

### `agent.defaults`

Baseline parameters shared by every agent. Each profile inherits these and may
override any subset (see [deep-merge behavior](#profile-overrides-and-deep-merge)).

Every field is optional; anything omitted uses the built-in `Agent` default
shown below.

#### `observation.attention`

Float in `[0, 1]`. Probability that the agent notices an observation event it is
visible to. Default `1.0`.

#### `observation.bias`

Float in `[-1, 1]`. Systematic perceptual shift applied to noticed observation
evidence (`encoded = clamp(evidence + bias)`). Default `0.0`. Realistic configs
use small values like `-0.1`, `0.0`, or `0.1`.

#### `trust.default`

Float. Trust assigned to otherwise-unseen source agents; modulates the weight of
heard evidence. Default `0.5`.

#### `learning.rate`

Float. Global plasticity — the base learning rate applied to all belief updates
before channel weights. Default `0.1`.

#### `learning.observe_weight`

Float. Channel weight for `OBSERVE` memories. Default `0.6`.

#### `learning.hear_weight`

Float. Channel weight for `HEAR` memories (further multiplied by trust in the
source). Default `0.3`.

#### `learning.verify_weight`

Float. Channel weight for `VERIFY` memories. Default `1.0`.

#### `action_preference`

Mapping of action name -> preference in `[0, 1]`. Valid action names are
`IDLE`, `VERIFY`, `COMMUNICATE`, `BROADCAST`. Defaults:

```yaml
action_preference:
  IDLE: 0.0
  VERIFY: 0.9
  COMMUNICATE: 0.7
  BROADCAST: 0.5
```

#### `action_cost`

Mapping of action name -> non-negative cost. Same valid action names. Defaults:

```yaml
action_cost:
  IDLE: 0.0
  VERIFY: 0.35
  COMMUNICATE: 0.15
  BROADCAST: 0.30
```

A partial `action_preference` / `action_cost` map is merged onto the built-in
defaults, so you only need to list the actions you want to change.

### `agent.profiles`

A non-empty list of concrete subpopulations. Each entry is built from
`agent.defaults` plus the profile's own overrides.

#### `name`

Required string. Identifies the profile. Reported back via
`World.profile_counts` for verifying expansion and per-profile analysis.

#### `count`

Required integer greater than zero.

The total number of agents in the simulation is the **sum of all profile
counts**. There is no separate `world.num_agents` in the canonical config
format.

#### profile overrides and deep-merge

Beyond `name` and `count`, a profile may include any subset of the agent
settings (`observation`, `trust`, `learning`, `action_preference`,
`action_cost`). These are **deep-merged** onto `agent.defaults`:

- nested maps (e.g. `observation`, `learning`) merge key-by-key, so a profile
  can override `observation.attention` without restating `observation.bias`
- scalar values replace the default value
- `action_preference` / `action_cost` merge per action name

The homogeneous case is simply a single profile that adds no overrides.

## Validation Rules

`validate_config` enforces the following. A violation raises `ValueError`.

| Field                                   | Rule                                  |
| --------------------------------------- | ------------------------------------- |
| `world.observation.private_event_rate`  | in `[0, 1]`                           |
| `world.observation.global_event_rate`   | in `[0, 1]`                           |
| `world.noise.{OBSERVE,HEAR,VERIFY}`     | present and `>= 0`                     |
| `world.truths.<id>`                     | value must be a boolean               |
| `agent.defaults`                        | required                              |
| `agent.profiles`                        | required, at least one profile        |
| `agent.profiles[*].name`                | required                              |
| `agent.profiles[*].count`               | required, `> 0`                       |
| `*.observation.attention`               | in `[0, 1]`                           |
| `*.observation.bias`                    | in `[-1, 1]`                          |
| `*.action_preference.<ACTION>`          | known action; value in `[0, 1]`       |
| `*.action_cost.<ACTION>`                | known action; value `>= 0`            |

`observation`, `action_preference`, and `action_cost` rules apply to both
`agent.defaults` and every profile node.

## Example: Homogeneous Population

A single profile of 50 identical agents, overriding only action parameters:

```yaml
world:
  rng_seed: 0
  observation:
    private_event_rate: 0.1
    global_event_rate: 0.0
  truths:
    0: true
  noise:
    OBSERVE: 0.1
    HEAR: 0.15
    VERIFY: 0.05

agent:
  defaults:
    action_preference:
      IDLE: 0.0
      VERIFY: 0.9
      COMMUNICATE: 0.7
      BROADCAST: 0.5
    action_cost:
      IDLE: 0.0
      VERIFY: 0.35
      COMMUNICATE: 0.15
      BROADCAST: 0.30
  profiles:
    - name: default
      count: 50
```

## Example: Heterogeneous Population

Three subpopulations sharing common defaults, each overriding a different slice
of cognition. A non-zero `global_event_rate` adds occasional shared incidents.

```yaml
world:
  rng_seed: 0
  observation:
    private_event_rate: 0.1
    global_event_rate: 0.05
  truths:
    0: true
  noise:
    OBSERVE: 0.1
    HEAR: 0.15
    VERIFY: 0.05

agent:
  defaults:
    observation: { attention: 1.0, bias: 0.0 }
    trust: { default: 0.5 }
    learning: { rate: 0.1, observe_weight: 0.6, hear_weight: 0.3, verify_weight: 1.0 }
    action_preference: { IDLE: 0.0, VERIFY: 0.9, COMMUNICATE: 0.7, BROADCAST: 0.5 }
    action_cost: { IDLE: 0.0, VERIFY: 0.35, COMMUNICATE: 0.15, BROADCAST: 0.30 }

  profiles:
    - name: attentive
      count: 20
      observation: { attention: 0.95 }

    - name: distracted
      count: 20
      observation: { attention: 0.35 }

    - name: skeptical
      count: 10
      trust: { default: 0.25 }
      learning: { hear_weight: 0.15, verify_weight: 1.2 }
      action_preference: { VERIFY: 1.0, COMMUNICATE: 0.35, BROADCAST: 0.2 }
```

This run has `20 + 20 + 10 = 50` agents.
