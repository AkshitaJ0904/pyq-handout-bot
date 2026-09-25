"""Shared types for the guardrail pipeline.

A guardrail is a check that runs at one of three stages -- on the user's
input, on what retrieval returned, or on what the model generated -- and
returns a verdict. The pipeline stops at the first BLOCK and reports which
guard fired and why, so a refusal is always explainable rather than the model
simply declining.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Action(str, Enum):
    ALLOW = "allow"   # nothing wrong
    WARN = "warn"     # suspicious, surface it but do not block
    BLOCK = "block"   # do not return this to the user


class Stage(str, Enum):
    INPUT = "input"
    RETRIEVAL = "retrieval"
    OUTPUT = "output"


@dataclass
class Verdict:
    guard: str
    stage: Stage
    action: Action
    reason: str = ""                 # why, for the report and the UI
    message: str = ""                # what the user is shown when blocked
    evidence: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.action is Action.BLOCK

    def to_dict(self):
        return {
            "guard": self.guard,
            "stage": self.stage.value,
            "action": self.action.value,
            "reason": self.reason,
            "message": self.message,
            "evidence": self.evidence,
        }


def allow(guard, stage, reason=""):
    return Verdict(guard=guard, stage=stage, action=Action.ALLOW, reason=reason)


def block(guard, stage, reason, message, **evidence):
    return Verdict(guard=guard, stage=stage, action=Action.BLOCK,
                   reason=reason, message=message, evidence=evidence)


def warn(guard, stage, reason, **evidence):
    return Verdict(guard=guard, stage=stage, action=Action.WARN,
                   reason=reason, evidence=evidence)


@dataclass
class PipelineResult:
    """What the app returns whether or not a guard fired."""
    allowed: bool
    answer: Optional[str]
    verdicts: List[Verdict] = field(default_factory=list)
    chunks: list = field(default_factory=list)

    @property
    def blocking_verdict(self):
        return next((v for v in self.verdicts if v.blocked), None)

    def to_dict(self):
        b = self.blocking_verdict
        return {
            "allowed": self.allowed,
            "answer": self.answer,
            "blocked_by": b.guard if b else None,
            "block_reason": b.reason if b else None,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }
