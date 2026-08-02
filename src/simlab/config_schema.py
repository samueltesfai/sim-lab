"""Pydantic models validating config shape, types, and ranges.

These models are used for validation only -- ``SimConfig.model_validate(cfg)``
is called and then discarded. ``config.py``'s deep-merge remains the single
mechanism that actually produces the fully-effective config; nothing here
replaces it.

Two families of settings models exist:

- ``*Override`` models (``ObservationOverride``, ``AgentSettingsOverride``,
  ``AgentProfile``, ...): every field is ``Optional``/``None``, since a
  profile only names the subset of settings it overrides.
- ``*Settings`` models (``ObservationSettings``, ``AgentSettings``): every
  field has a real default, sourced directly from ``agent.DEFAULT_SETTINGS``
  (not re-declared as an independent literal), used to validate
  ``agent.defaults``. Reading the same dict `agent.py` itself reads for its
  constructor defaults means the two can't silently disagree the way two
  independent default declarations could.

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


def _validate_preference_range(
    v: dict[str, float] | None,
) -> dict[str, float] | None:
    if v and any(not 0 <= value <= 1 for value in v.values()):
        raise ValueError("action_preference values must be in [0, 1]")
    return v


def _validate_cost_range(v: dict[str, float] | None) -> dict[str, float] | None:
    if v and any(value < 0 for value in v.values()):
        raise ValueError("action_cost values must be non-negative")
    return v


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
        return _validate_preference_range(v)

    @field_validator("action_cost")
    @classmethod
    def _cost_range(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        return _validate_cost_range(v)


class AgentProfile(AgentSettingsOverride):
    name: StrictStr
    count: StrictInt = Field(gt=0)


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
    """``agent.defaults``' validation model -- real defaults (unlike
    ``AgentSettingsOverride``), since ``agent.defaults`` is the resolved
    baseline profiles override, not itself an override of something else.
    """

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


class AgentSection(_Strict):
    defaults: AgentSettings
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
