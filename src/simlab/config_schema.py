"""Pydantic models validating a *materialized* (fully merged) config.

``SimConfig.model_validate(materialize_config(cfg))`` is called against the
already-merged config -- ``config.py``'s ``_deep_merge`` runs first, pydantic
validates second. This is deliberate: validating before merging (this
module's earlier shape) needs a whole parallel family of Optional-everywhere
models just to tolerate partial input, and it has a real blind spot --
pydantic never checks a field's own default against its own constraints, so
a bad value in ``agent.DEFAULT_SETTINGS``/``world.DEFAULT_NOISE`` would never
be caught as long as the config itself omitted that field. Validating the
merged result closes that gap: every value, whether it came from user YAML
or from a built-in default, is something explicitly present in the dict
being validated, so range/type checks apply to it uniformly. A quick
unrecognized-key check (an unrecognized key set via merge is still an
"extra" key on the merged result) and a deliberately-wrong built-in default
were both verified to still get rejected under this design before it was
adopted.

Only one settings shape is needed as a result: every field has a real
default, sourced directly from ``agent.DEFAULT_SETTINGS``/
``world.DEFAULT_NOISE`` (not re-declared as an independent literal) --
reading the same dict ``agent.py``/``world.py`` themselves read for their
own constructor defaults means the two can't silently disagree.

Strict types (``StrictInt``/``StrictFloat``/``StrictBool``/``StrictStr``) are
used throughout so that type-confused values -- a boolean where a rate is
expected, a string key where a claim id is expected -- are rejected here
rather than passing through and corrupting a fingerprint or crashing
downstream JSON serialization.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr

from simlab.agent import DEFAULT_SETTINGS
from simlab.world import DEFAULT_NOISE

ActionName = Literal["IDLE", "VERIFY", "COMMUNICATE", "BROADCAST"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _validate_preference_range(v: dict[str, float]) -> dict[str, float]:
    if any(not 0 <= value <= 1 for value in v.values()):
        raise ValueError("action_preference values must be in [0, 1]")
    return v


def _validate_cost_range(v: dict[str, float]) -> dict[str, float]:
    if any(value < 0 for value in v.values()):
        raise ValueError("action_cost values must be non-negative")
    return v


class ObservationSettings(_Strict):
    attention: StrictFloat = Field(
        DEFAULT_SETTINGS["observation"]["attention"], ge=0, le=1
    )
    bias: StrictFloat = Field(DEFAULT_SETTINGS["observation"]["bias"], ge=-1, le=1)


class TrustSettings(_Strict):
    default: StrictFloat = DEFAULT_SETTINGS["trust"]["default"]


class SocialSettings(_Strict):
    confidence_bound: StrictFloat = Field(
        DEFAULT_SETTINGS["social"]["confidence_bound"], ge=0, le=1
    )
    trust_update_rate: StrictFloat = Field(
        DEFAULT_SETTINGS["social"]["trust_update_rate"], ge=0, le=1
    )
    update_trust_on_rejection: StrictBool = DEFAULT_SETTINGS["social"][
        "update_trust_on_rejection"
    ]


class LearningSettings(_Strict):
    rate: StrictFloat = DEFAULT_SETTINGS["learning"]["rate"]
    observe_weight: StrictFloat = DEFAULT_SETTINGS["learning"]["observe_weight"]
    hear_weight: StrictFloat = DEFAULT_SETTINGS["learning"]["hear_weight"]
    verify_weight: StrictFloat = DEFAULT_SETTINGS["learning"]["verify_weight"]


class AgentSettings(_Strict):
    """A fully-resolved agent settings node -- every field present, as
    produced by ``config.py``'s ``_materialize_agent_profiles``."""

    observation: ObservationSettings = Field(default_factory=ObservationSettings)
    trust: TrustSettings = Field(default_factory=TrustSettings)
    social: SocialSettings = Field(default_factory=SocialSettings)
    learning: LearningSettings = Field(default_factory=LearningSettings)
    action_preference: dict[ActionName, StrictFloat] = Field(
        default_factory=lambda: dict(DEFAULT_SETTINGS["action_preference"])
    )
    action_cost: dict[ActionName, StrictFloat] = Field(
        default_factory=lambda: dict(DEFAULT_SETTINGS["action_cost"])
    )

    @field_validator("action_preference")
    @classmethod
    def _preference_range(cls, v: dict[str, float]) -> dict[str, float]:
        return _validate_preference_range(v)

    @field_validator("action_cost")
    @classmethod
    def _cost_range(cls, v: dict[str, float]) -> dict[str, float]:
        return _validate_cost_range(v)


class AgentProfile(AgentSettings):
    """A fully-materialized profile -- ``AgentSettings`` plus the
    ``name``/``count`` metadata that rides alongside settings, not merged
    with them."""

    name: StrictStr
    count: StrictInt = Field(gt=0)


class AgentSection(_Strict):
    profiles: list[AgentProfile] = Field(min_length=1)

    @field_validator("profiles")
    @classmethod
    def _unique_names(cls, profiles: list[AgentProfile]) -> list[AgentProfile]:
        seen: set[str] = set()
        for profile in profiles:
            if profile.name in seen:
                raise ValueError(f"duplicate agent profile name: {profile.name!r}")
            seen.add(profile.name)
        return profiles


class WorldObservation(_Strict):
    private_event_rate: StrictFloat = Field(ge=0, le=1)
    global_event_rate: StrictFloat = Field(ge=0, le=1)


class WorldSection(_Strict):
    rng_seed: StrictInt
    truths: dict[StrictInt, StrictBool]
    noise: dict[Literal["OBSERVE", "HEAR", "VERIFY"], StrictFloat] = Field(
        default_factory=lambda: dict(DEFAULT_NOISE)
    )
    observation: WorldObservation

    @field_validator("noise")
    @classmethod
    def _noise_nonneg(cls, v: dict[str, float]) -> dict[str, float]:
        if any(value < 0 for value in v.values()):
            raise ValueError("world.noise values must be non-negative")
        return v


class SimConfig(_Strict):
    """Validates a *materialized* config -- ``materialize_config(cfg)``'s
    output, not raw YAML."""

    world: WorldSection
    agent: AgentSection
