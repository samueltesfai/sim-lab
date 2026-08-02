"""Pydantic models validating config shape, types, and ranges.

These models are used for validation only -- ``SimConfig.model_validate(cfg)``
is called and then discarded. They deliberately do not assign default values
for omitted settings fields (every override field is ``Optional``/``None``):
that responsibility stays with ``agent.DEFAULT_SETTINGS``/``world.DEFAULT_NOISE``
and ``config.py``'s deep-merge, the single source of truth for defaults.

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

ActionName = Literal["IDLE", "VERIFY", "COMMUNICATE", "BROADCAST"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObservationOverride(_Strict):
    attention: StrictFloat | None = Field(None, ge=0, le=1)
    bias: StrictFloat | None = Field(None, ge=-1, le=1)


class TrustOverride(_Strict):
    default: StrictFloat | None = None


class SocialOverride(_Strict):
    confidence_bound: StrictFloat | None = Field(None, ge=0, le=1)
    trust_update_rate: StrictFloat | None = Field(None, ge=0, le=1)
    update_trust_on_rejection: StrictBool | None = None


class LearningOverride(_Strict):
    rate: StrictFloat | None = None
    observe_weight: StrictFloat | None = None
    hear_weight: StrictFloat | None = None
    verify_weight: StrictFloat | None = None


class AgentSettingsOverride(_Strict):
    observation: ObservationOverride | None = None
    trust: TrustOverride | None = None
    social: SocialOverride | None = None
    learning: LearningOverride | None = None
    action_preference: dict[ActionName, StrictFloat] | None = None
    action_cost: dict[ActionName, StrictFloat] | None = None

    @field_validator("action_preference")
    @classmethod
    def _preference_range(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v and any(not 0 <= value <= 1 for value in v.values()):
            raise ValueError("action_preference values must be in [0, 1]")
        return v

    @field_validator("action_cost")
    @classmethod
    def _cost_range(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v and any(value < 0 for value in v.values()):
            raise ValueError("action_cost values must be non-negative")
        return v


class AgentProfile(AgentSettingsOverride):
    name: StrictStr
    count: StrictInt = Field(gt=0)


class AgentSection(_Strict):
    defaults: AgentSettingsOverride
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
        default_factory=dict
    )
    observation: WorldObservation

    @field_validator("noise")
    @classmethod
    def _noise_nonneg(cls, v: dict[str, float]) -> dict[str, float]:
        if any(value < 0 for value in v.values()):
            raise ValueError("world.noise values must be non-negative")
        return v


class SimConfig(_Strict):
    world: WorldSection
    agent: AgentSection
