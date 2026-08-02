"""Pydantic models validating a *materialized* (fully merged) config.

``SimConfig.model_validate(materialize_config(cfg))`` is called against the
already-merged config -- ``config.py``'s ``deep_merge`` runs first, pydantic
validates second. This is deliberate: validating before merging (this
module's earlier shape) needs a whole parallel family of Optional-everywhere
models just to tolerate partial input, and it has a real blind spot --
pydantic never checks a field's own default against its own constraints, so
a bad built-in default would never be caught as long as the config itself
omitted that field. Validating the merged result closes that gap: every
value, whether it came from user YAML or from a built-in default, is
something explicitly present in the dict being validated, so range/type
checks apply to it uniformly. A quick unrecognized-key check (an
unrecognized key set via merge is still an "extra" key on the merged
result) and a deliberately-wrong built-in default were both verified to
still get rejected under this design before it was adopted.

This module is the single canonical declaration of every config field --
name, type, range, and default. ``agent.py``/``world.py`` don't declare
their own parallel default dicts; they derive what they need (a
constructor's own default, or a plain dict for ``config.py``'s pre-merge
step) directly from the pydantic models here, so a field's existence and
its default value can never disagree between "what the schema accepts" and
"what the constructor/merge step assumes."

Strict types (``StrictInt``/``StrictFloat``/``StrictBool``/``StrictStr``) are
used throughout so that type-confused values -- a boolean where a rate is
expected, a string key where a claim id is expected -- are rejected here
rather than passing through and corrupting a fingerprint or crashing
downstream JSON serialization.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)

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
    attention: StrictFloat = Field(1.0, ge=0, le=1)
    bias: StrictFloat = Field(0.0, ge=-1, le=1)


class TrustSettings(_Strict):
    default: StrictFloat = 0.5


class SocialSettings(_Strict):
    confidence_bound: StrictFloat = Field(1.0, ge=0, le=1)
    trust_update_rate: StrictFloat = Field(0.0, ge=0, le=1)
    update_trust_on_rejection: StrictBool = True


class LearningSettings(_Strict):
    rate: StrictFloat = 0.1
    observe_weight: StrictFloat = 0.6
    hear_weight: StrictFloat = 0.3
    verify_weight: StrictFloat = 1.0


class AgentSettings(_Strict):
    """A fully-resolved agent settings node -- every field present, as
    produced by ``config.py``'s ``_materialize_agent_profiles``."""

    observation: ObservationSettings = Field(default_factory=ObservationSettings)
    trust: TrustSettings = Field(default_factory=TrustSettings)
    social: SocialSettings = Field(default_factory=SocialSettings)
    learning: LearningSettings = Field(default_factory=LearningSettings)
    action_preference: dict[ActionName, StrictFloat] = Field(
        default_factory=lambda: {
            "IDLE": 0.0,
            "VERIFY": 0.9,
            "COMMUNICATE": 0.7,
            "BROADCAST": 0.5,
        }
    )
    action_cost: dict[ActionName, StrictFloat] = Field(
        default_factory=lambda: {
            "IDLE": 0.0,
            "VERIFY": 0.35,
            "COMMUNICATE": 0.15,
            "BROADCAST": 0.30,
        }
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
    private_event_rate: StrictFloat = Field(0.1, ge=0, le=1)
    global_event_rate: StrictFloat = Field(0.0, ge=0, le=1)


class WorldSection(_Strict):
    rng_seed: StrictInt
    truths: dict[StrictInt, StrictBool]
    noise: dict[Literal["OBSERVE", "HEAR", "VERIFY"], StrictFloat] = Field(
        default_factory=lambda: {"OBSERVE": 0.0, "HEAR": 0.0, "VERIFY": 0.0}
    )
    observation: WorldObservation = Field(default_factory=WorldObservation)

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
