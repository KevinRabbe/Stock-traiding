from __future__ import annotations

from dataclasses import dataclass, replace

from .contracts import StrategyStage
from .registry import ProfitabilityGate, StrategyRecord, StrategyRegistry


_ALLOWED_TRANSITIONS = {
    StrategyStage.DEVELOPMENT: {StrategyStage.SHADOW, StrategyStage.RETIRED},
    StrategyStage.SHADOW: {StrategyStage.PAPER, StrategyStage.RETIRED},
    StrategyStage.PAPER: {StrategyStage.LIVE, StrategyStage.RETIRED},
    StrategyStage.LIVE: {StrategyStage.PAPER, StrategyStage.RETIRED},
    StrategyStage.RETIRED: set(),
}
_UPGRADES = {
    (StrategyStage.DEVELOPMENT, StrategyStage.SHADOW),
    (StrategyStage.SHADOW, StrategyStage.PAPER),
    (StrategyStage.PAPER, StrategyStage.LIVE),
}


@dataclass(frozen=True, slots=True)
class StrategyLifecyclePolicy:
    shadow_gate: ProfitabilityGate = ProfitabilityGate(min_trades=1)
    paper_gate: ProfitabilityGate = ProfitabilityGate(min_trades=1)
    live_gate: ProfitabilityGate = ProfitabilityGate(min_trades=1)
    require_artifact_for_paper: bool = True
    require_artifact_for_live: bool = True

    def gate_for(self, target: StrategyStage) -> ProfitabilityGate | None:
        if target is StrategyStage.SHADOW:
            return self.shadow_gate
        if target is StrategyStage.PAPER:
            return self.paper_gate
        if target is StrategyStage.LIVE:
            return self.live_gate
        return None


class StrategyLifecycleController:
    """Explicit staged promotion; stage changes never silently switch champion."""

    def __init__(
        self,
        registry: StrategyRegistry,
        policy: StrategyLifecyclePolicy | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy or StrategyLifecyclePolicy()

    def transition(self, strategy_id: str, target: StrategyStage) -> StrategyRecord:
        record = self.registry.record(strategy_id)
        if target is record.stage:
            return record
        allowed = _ALLOWED_TRANSITIONS[record.stage]
        if target not in allowed:
            raise ValueError(
                f"invalid strategy lifecycle transition {record.stage.value} -> {target.value}"
            )
        if (record.stage, target) in _UPGRADES:
            gate = self.policy.gate_for(target)
            if gate is not None and not gate.eligible(record):
                raise ValueError(
                    f"strategy {strategy_id} does not satisfy profitability gate for {target.value}"
                )
            if (
                target is StrategyStage.PAPER
                and self.policy.require_artifact_for_paper
                and not record.artifact_ref
            ):
                raise ValueError("paper promotion requires an immutable strategy artifact_ref")
            if (
                target is StrategyStage.LIVE
                and self.policy.require_artifact_for_live
                and not record.artifact_ref
            ):
                raise ValueError("live promotion requires an immutable strategy artifact_ref")
        updated = replace(record, stage=target)
        self.registry.update_record(updated)
        return updated

    def promote_champion(
        self,
        strategy_id: str,
        *,
        required_stage: StrategyStage,
    ) -> None:
        record = self.registry.record(strategy_id)
        if record.stage is not required_stage:
            raise ValueError(
                f"strategy {strategy_id} must be {required_stage.value} before champion promotion"
            )
        self.registry.loaded_strategy(strategy_id)
        self.registry.set_champion(strategy_id)
