"""Unit tests for agents.infrastructure_agent (run: python -m pytest tests -v)."""
import json
import math

from agents.infrastructure_agent import InfrastructureAgent, analyze

HEALTHY = {"device_id": "SW1", "status": "up", "cpu_pct": 25, "memory_pct": 40,
           "temperature_c": 45, "queue_drops": 0, "interface_errors": 0}


def top(msg):
    return msg["hypotheses"][0]["hypothesis"] if msg["hypotheses"] else None


def test_normal_baseline_is_ignored():
    msg = analyze([HEALTHY])
    assert msg["status"] == "healthy"
    assert msg["hypotheses"] == [] and msg["affected_devices"] == []
    assert msg["confidence"] > 0.8


def test_buffer_drops_with_normal_cpu_flags_buffer_exhaustion():
    msg = analyze([{**HEALTHY, "queue_drops": 160}])
    assert msg["status"] == "anomaly_detected"
    assert top(msg) == "egress_buffer_exhaustion"
    assert msg["affected_devices"] == ["SW1"]
    assert any(e["metric"] == "queue_drops" and e["value"] == 160 for e in msg["supporting_evidence"])


def test_buffer_drops_with_high_cpu_is_less_confident():
    low = analyze([{**HEALTHY, "queue_drops": 160}])["hypotheses"][0]["confidence"]
    high = analyze([{**HEALTHY, "queue_drops": 160, "cpu_pct": 95}])["hypotheses"][0]["confidence"]
    assert high < low


def test_crc_errors_identify_cable_fault_with_high_confidence():
    msg = analyze([{**HEALTHY, "interface_errors": 250}])
    assert top(msg) == "physical_layer_degradation"
    assert msg["hypotheses"][0]["confidence"] >= 0.85


def test_single_crc_error_is_flagged():
    assert top(analyze([{**HEALTHY, "interface_errors": 1}])) == "physical_layer_degradation"


def test_cpu_and_memory_exhaustion_is_control_plane_overload():
    msg = analyze([{**HEALTHY, "cpu_pct": 96, "memory_pct": 95}])
    assert top(msg) == "control_plane_overload"


def test_brief_cpu_spike_gets_only_moderate_confidence():
    samples = [{**HEALTHY, "timestamp": f"2026-10-31T10:0{i}:00Z", "cpu_pct": c}
               for i, c in enumerate([20, 22, 25, 93])]
    msg = analyze(samples)
    assert top(msg) == "transient_cpu_spike"
    assert msg["hypotheses"][0]["confidence"] <= 0.5


def test_overheating_and_down_device():
    assert top(analyze([{**HEALTHY, "temperature_c": 90}])) == "thermal_stress"
    assert top(analyze([{**HEALTHY, "status": "down"}])) == "device_unreachable"


def test_aliases_and_unit_conversion_are_reported():
    msg = analyze([{"hostname": "AP1", "cpu_ratio": 0.95, "memory_utilization": 93, "temperature_f": 120}])
    assert msg["device_health"]["AP1"]["metrics"]["cpu_pct"] == 95.0
    assert top(msg) == "control_plane_overload"
    assert any("converted" in w["issue"] for w in msg["warnings"])


def test_malformed_inputs_never_raise():
    for bad in [None, 42, "oops", [], {}, [None, 3, "x"], [{"cpu_pct": 10}], {"foo": 1}]:
        msg = analyze(bad)
        assert msg["status"] in ("insufficient_evidence", "error")
        assert msg["confidence"] == 0.0
        json.dumps(msg)  # must stay serializable


def test_bad_values_are_ignored_not_trusted():
    msg = analyze([{**HEALTHY, "cpu_pct": "n/a", "memory_pct": 250, "queue_drops": float("nan"), "interface_errors": -5}])
    assert msg["status"] == "healthy"
    assert len(msg["warnings"]) >= 4
    assert all(not (isinstance(e["value"], float) and math.isnan(e["value"])) for e in msg["supporting_evidence"])


def test_missing_devices_reported_as_visibility_gaps_and_lower_confidence():
    full = analyze([HEALTHY], expected_devices=["SW1"])
    gap = analyze([HEALTHY], expected_devices=["SW1", "CORE1"])
    assert any(g["device_id"] == "CORE1" for g in gap["visibility_gaps"])
    assert gap["limitations"]


def test_missing_metrics_lower_confidence():
    full = analyze([{**HEALTHY, "queue_drops": 160}])["confidence"]
    partial = analyze([{"device_id": "SW1", "cpu_pct": 20, "queue_drops": 160}])["confidence"]
    assert partial < full


def test_output_is_standard_envelope():
    msg = InfrastructureAgent().analyze([HEALTHY], incident_id="INC-1")
    for key in ("schema_version", "message_type", "agent", "incident_id", "status", "confidence",
                "affected_devices", "hypotheses", "supporting_evidence", "visibility_gaps", "limitations"):
        assert key in msg
    assert msg["agent"] == "infrastructure_agent" and msg["incident_id"] == "INC-1"
