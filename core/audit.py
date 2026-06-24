import json
from datetime import datetime, timezone
from typing import Any


class AuditTrail:
    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def record(
        self,
        task_id: str,
        event_type: str,
        detail: dict[str, Any],
    ) -> None:
        entry = {
            "task_id": task_id,
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detail": detail,
        }
        self._entries.append(entry)

    def record_topology_decision(
        self,
        task_id: str,
        topology: str,
        model_tiers: dict[str, str],
        budget: float,
        rationale: str,
        alternatives: list[dict[str, str]],
    ) -> None:
        self.record(
            task_id=task_id,
            event_type="topology_decision",
            detail={
                "topology": topology,
                "model_tiers": model_tiers,
                "budget_at_decision": budget,
                "rationale": rationale,
                "alternatives_considered": alternatives,
            },
        )

    def record_budget_band(
        self,
        task_id: str,
        band: str,
        remaining_budget: float,
        action: str,
    ) -> None:
        self.record(
            task_id=task_id,
            event_type="budget_band_crossed",
            detail={
                "band": band,
                "remaining_budget": remaining_budget,
                "action": action,
            },
        )

    def record_degradation(
        self,
        task_id: str,
        from_topology: str,
        to_topology: str,
        reason: str,
    ) -> None:
        self.record(
            task_id=task_id,
            event_type="structural_degradation",
            detail={
                "from_topology": from_topology,
                "to_topology": to_topology,
                "reason": reason,
            },
        )

    def get_task_audit(self, task_id: str) -> list[dict[str, Any]]:
        return [e for e in self._entries if e["task_id"] == task_id]

    def to_json(self, task_id: str) -> str:
        return json.dumps(self.get_task_audit(task_id), indent=2, default=str)


audit_trail = AuditTrail()


def get_audit_trail() -> AuditTrail:
    return audit_trail