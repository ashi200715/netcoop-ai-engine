"""Unit tests for agents.topology_agent."""
from agents.topology_agent import analyze

TOPO = {"nodes": [
    {"id": "GW", "role": "gateway"}, {"id": "CORE", "role": "core", "parent": "GW"},
    {"id": "DIST", "role": "distribution", "parent": "CORE"},
    {"id": "SW_A", "role": "edge_switch", "parent": "DIST", "segment": "Hostel A"},
    {"id": "SW_B", "role": "edge_switch", "parent": "DIST", "segment": "Hostel B"},
    {"id": "AP1", "role": "access_point", "parent": "SW_A"}, {"id": "AP2", "role": "access_point", "parent": "SW_A"},
    {"id": "AP3", "role": "access_point", "parent": "SW_A"}, {"id": "AP4", "role": "access_point", "parent": "SW_B"},
]}


def test_single_ap_is_isolated():
    msg = analyze(TOPO, ["AP1"])
    assert msg["blast_radius"]["scope"] == "single_device"
    assert msg["root_candidates"][0]["device_id"] == "AP1"


def test_downstream_aps_are_casualties_of_failed_switch():
    msg = analyze(TOPO, ["SW_A", "AP1", "AP2", "AP3"])
    assert [r["device_id"] for r in msg["root_candidates"]] == ["SW_A"]
    assert {c["device_id"] for c in msg["casualties"]} == {"AP1", "AP2", "AP3"}
    assert msg["blast_radius"]["scope"] == "segment"


def test_unreachable_switch_shadows_children_without_alerts():
    msg = analyze(TOPO, [], unreachable=["SW_A"])
    assert msg["root_candidates"][0]["shadowed_devices"] == ["AP1", "AP2", "AP3"]


def test_silent_parent_with_many_alerted_children_is_suspected():
    msg = analyze(TOPO, ["AP1", "AP2", "AP3"])
    assert msg["suspected_common_upstream"][0]["device_id"] == "SW_A"
    assert any(g["device_id"] == "SW_A" for g in msg["visibility_gaps"])


def test_core_failure_is_upstream_aggregation():
    msg = analyze(TOPO, ["CORE", "SW_A", "SW_B", "AP1", "AP4"])
    assert [r["device_id"] for r in msg["root_candidates"]] == ["CORE"]
    assert msg["blast_radius"]["scope"] == "upstream_aggregation"
    assert len(msg["casualties"]) == 4


def test_bad_topology_and_unknown_alerts_handled():
    assert analyze(None, ["AP1"])["status"] == "insufficient_evidence"
    assert analyze({"nodes": "x"}, ["AP1"])["status"] == "insufficient_evidence"
    msg = analyze(TOPO, ["GHOST", "AP1"])
    assert any(g["device_id"] == "GHOST" for g in msg["visibility_gaps"])
    assert msg["confidence"] < 0.9


def test_no_alerts():
    assert analyze(TOPO, [])["status"] == "no_alerts"
