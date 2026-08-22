from core.audit import AuditTrail, get_audit_trail


class TestAuditTrail:
    def test_record_and_get(self):
        audit = AuditTrail()
        audit.record("task-1", "test_event", {"key": "value"})
        events = audit.get_task_audit("task-1")
        assert len(events) == 1
        assert events[0]["event_type"] == "test_event"
        assert events[0]["detail"]["key"] == "value"
        assert events[0]["task_id"] == "task-1"

    def test_get_empty_for_unknown_task(self):
        audit = AuditTrail()
        assert audit.get_task_audit("nonexistent") == []

    def test_record_topology_decision(self):
        audit = AuditTrail()
        audit.record_topology_decision(
            task_id="t1",
            topology="pipeline",
            model_tiers={"planner": "standard"},
            budget=80.0,
            rationale="code task",
            alternatives=[],
        )
        events = audit.get_task_audit("t1")
        assert len(events) == 1
        assert events[0]["event_type"] == "topology_decision"
        assert events[0]["detail"]["topology"] == "pipeline"

    def test_record_budget_band(self):
        audit = AuditTrail()
        audit.record_budget_band("t2", "tier_downgrade", 25.0, "Downgrading tiers")
        events = audit.get_task_audit("t2")
        assert events[0]["event_type"] == "budget_band_crossed"
        assert events[0]["detail"]["band"] == "tier_downgrade"

    def test_record_degradation(self):
        audit = AuditTrail()
        audit.record_degradation("t3", "ensemble", "fanout", "budget low")
        events = audit.get_task_audit("t3")
        assert events[0]["event_type"] == "structural_degradation"
        assert events[0]["detail"]["from_topology"] == "ensemble"
        assert events[0]["detail"]["to_topology"] == "fanout"

    def test_multiple_tasks_isolated(self):
        audit = AuditTrail()
        audit.record("t1", "event_a", {})
        audit.record("t2", "event_b", {})
        assert len(audit.get_task_audit("t1")) == 1
        assert len(audit.get_task_audit("t2")) == 1

    def test_to_json(self):
        audit = AuditTrail()
        audit.record("t1", "event", {"key": "val"})
        json_str = audit.to_json("t1")
        assert "event" in json_str
        assert "key" in json_str

    def test_timestamp_present(self):
        audit = AuditTrail()
        audit.record("t1", "event", {})
        assert "timestamp" in audit.get_task_audit("t1")[0]


class TestGetAuditTrailSingleton:
    def test_returns_same_instance(self):
        a = get_audit_trail()
        b = get_audit_trail()
        assert a is b
