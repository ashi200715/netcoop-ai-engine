# Implement connectivity and network-topology analysis, including affected paths and links.
"""Topology Agent (Member 3).

Maps how devices connect (Gateway -> Core -> Distribution -> Edge Switch -> AP -> Endpoint),
works out which alerts are *root* candidates and which are merely *casualties* of an
upstream failure, and sizes the blast radius.

Input
-----
topology: {"nodes": [{"id": str, "role": str, "parent": str?, "segment": str?}],
           "links": [{"source": str, "target": str}]}      (links and/or parent fields)
alerts:   list of device ids, or dicts with "device_id" (e.g. from infrastructure_agent
          ``affected_devices`` plus the performance agent's findings)
unreachable: optional device ids known to be down; everything below them is shadowed.

Output: ``agent_finding`` message (schema_version "1.0"), same envelope as the
infrastructure agent, with topology-specific fields: ``dependency_chains``,
``root_candidates``, ``casualties``, ``suspected_common_upstream``, ``blast_radius``.

Confidence method: start at 0.90 (structure is deterministic), minus 0.10 if redundant
links make the tree ambiguous, minus 0.05 if nodes are disconnected, minus 0.10 if roles
are missing, minus 0.40 * (share of alerted devices unknown to the topology). Floor 0.10.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

SCHEMA_VERSION = "1.0"
AGENT_NAME = "topology_agent"

TIERS = ["gateway", "core", "distribution", "edge_switch", "access_point", "endpoint"]
ROLE_ALIASES = {
    "gateway": "gateway", "internet": "gateway", "firewall": "gateway", "router_gw": "gateway",
    "core": "core", "core_switch": "core", "core_router": "core",
    "distribution": "distribution", "dist": "distribution", "aggregation": "distribution", "router": "distribution",
    "edge": "edge_switch", "edge_switch": "edge_switch", "access_switch": "edge_switch", "switch": "edge_switch",
    "ap": "access_point", "access_point": "access_point", "wireless_ap": "access_point",
    "host": "endpoint", "client": "endpoint", "endpoint": "endpoint",
}
SCOPE_BY_ROLE = {
    "endpoint": "single_device", "access_point": "single_device",
    "edge_switch": "segment", "distribution": "upstream_aggregation",
    "core": "upstream_aggregation", "gateway": "upstream_aggregation",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TopologyAgent:
    def analyze(self, topology: Any, alerts: Iterable[Any], unreachable: Optional[Iterable[str]] = None,
                incident_id: Optional[str] = None) -> dict:
        """Return an ``agent_finding`` message. Never raises on bad input."""
        msg = {"schema_version": SCHEMA_VERSION, "message_type": "agent_finding", "agent": AGENT_NAME,
               "incident_id": incident_id, "timestamp": _now(),
               "confidence_method": "0.90 base minus penalties for redundancy, disconnection, missing roles, unknown alerts"}
        try:
            return self._analyze(msg, topology, alerts, set(unreachable or []))
        except Exception as exc:
            msg.update(status="error", confidence=0.0, affected_devices=[], summary="Topology agent failed internally",
                       limitations=[f"internal error: {exc!r}"], visibility_gaps=[], warnings=[])
            return msg

    # ----------------------------------------------------------------- build
    @staticmethod
    def _alert_ids(alerts: Iterable[Any]) -> Set[str]:
        ids: Set[str] = set()
        for a in alerts or []:
            if isinstance(a, str) and a.strip():
                ids.add(a.strip())
            elif isinstance(a, dict) and isinstance(a.get("device_id"), str):
                ids.add(a["device_id"].strip())
        return ids

    def _build(self, topology: Any, warnings: List[dict]):
        if not isinstance(topology, dict) or not isinstance(topology.get("nodes"), list) or not topology["nodes"]:
            return None
        roles: Dict[str, Optional[str]] = {}
        segments: Dict[str, str] = {}
        edges: List[tuple] = []
        for n in topology["nodes"]:
            if not isinstance(n, dict) or not isinstance(n.get("id"), str):
                warnings.append({"issue": "node without string id skipped", "node": repr(n)[:60]})
                continue
            role = ROLE_ALIASES.get(str(n.get("role", "")).strip().lower())
            if n.get("role") is not None and role is None:
                warnings.append({"device_id": n["id"], "issue": f"unknown role {n.get('role')!r}"})
            roles[n["id"]] = role
            if n.get("segment"):
                segments[n["id"]] = str(n["segment"])
            if isinstance(n.get("parent"), str):
                edges.append((n["parent"], n["id"]))
        for l in topology.get("links") or []:
            if isinstance(l, dict):
                s, t = l.get("source", l.get("a")), l.get("target", l.get("b"))
                if isinstance(s, str) and isinstance(t, str):
                    edges.append((s, t))
                    continue
            warnings.append({"issue": "malformed link skipped", "link": repr(l)[:60]})
        adj: Dict[str, Set[str]] = {n: set() for n in roles}
        for s, t in edges:
            if s in adj and t in adj and s != t:
                adj[s].add(t)
                adj[t].add(s)
            else:
                warnings.append({"issue": f"link {s}->{t} references unknown node; skipped"})
        if not roles:
            return None
        def tier(n): return TIERS.index(roles[n]) if roles[n] else 99
        root = min(roles, key=lambda n: (tier(n), n))
        parent: Dict[str, Optional[str]] = {root: None}
        depth = {root: 0}
        q = deque([root])
        while q:
            u = q.popleft()
            for v in sorted(adj[u]):
                if v not in parent:
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    q.append(v)
        children: Dict[str, List[str]] = {n: [] for n in parent}
        for c, p in parent.items():
            if p:
                children[p].append(c)
        n_edges = sum(len(v) for v in adj.values()) // 2
        return {"roles": roles, "segments": segments, "parent": parent, "children": children, "root": root,
                "disconnected": sorted(set(roles) - set(parent)), "redundant": n_edges > len(parent) - 1}

    @staticmethod
    def _descendants(g, node: str) -> List[str]:
        out, stack = [], list(g["children"].get(node, []))
        while stack:
            x = stack.pop()
            out.append(x)
            stack.extend(g["children"].get(x, []))
        return sorted(out)

    @staticmethod
    def _chain(g, node: str) -> List[str]:
        chain = [node]
        while g["parent"].get(chain[-1]):
            chain.append(g["parent"][chain[-1]])
        return chain

    # --------------------------------------------------------------- analyze
    def _analyze(self, msg: dict, topology: Any, alerts: Iterable[Any], unreachable: Set[str]) -> dict:
        warnings: List[dict] = []
        alerted = self._alert_ids(alerts)
        g = self._build(topology, warnings)
        if g is None:
            msg.update(status="insufficient_evidence", confidence=0.0, affected_devices=sorted(alerted),
                       summary="Topology missing or malformed; scope and dependencies cannot be determined",
                       visibility_gaps=[{"reason": "no valid topology"}], warnings=warnings,
                       limitations=["Cannot separate root causes from downstream casualties without topology"])
            return msg

        known = set(g["parent"])
        unknown_alerts = sorted(alerted - set(g["roles"]))
        alerted &= known
        unreachable &= known
        gaps = [{"device_id": d, "reason": "alerted device not present in topology"} for d in unknown_alerts]
        gaps += [{"device_id": d, "reason": "node in topology but disconnected from the gateway tree"} for d in g["disconnected"]]

        if not alerted and not unreachable:
            msg.update(status="no_alerts" if not unknown_alerts else "insufficient_evidence",
                       confidence=0.0 if unknown_alerts else 0.9, affected_devices=[],
                       summary="No alerted devices map onto the topology" if unknown_alerts else "No alerts to localize",
                       dependency_chains={}, root_candidates=[], casualties=[], suspected_common_upstream=[],
                       blast_radius={"scope": "none", "affected_count": 0, "affected_devices": []},
                       visibility_gaps=gaps, warnings=warnings, limitations=[])
            return msg

        faulty = alerted | unreachable
        def ancestors(n): return self._chain(g, n)[1:]
        roots, casualties = [], []
        for d in sorted(faulty):
            culprit = next((a for a in ancestors(d) if a in faulty), None)
            if culprit:
                casualties.append({"device_id": d, "caused_by": culprit,
                                   "reason": f"upstream device {culprit} is faulty; do not diagnose {d} independently"})
            else:
                roots.append(d)

        root_candidates = []
        for d in roots:
            desc = self._descendants(g, d)
            role = g["roles"][d]
            scope = SCOPE_BY_ROLE.get(role, "unknown")
            shadowed = [x for x in desc if x not in faulty] if d in unreachable else []
            root_candidates.append({
                "device_id": d, "role": role, "segment": g["segments"].get(d),
                "downstream_alerted": [x for x in desc if x in faulty],
                "shadowed_devices": shadowed,
                "blast_radius": {"scope": scope, "affected_count": 1 + len(desc), "affected_devices": [d] + desc},
                "reason": "no faulty device upstream of it" + ("; reported unreachable" if d in unreachable else ""),
            })

        # Common-upstream suspicion: several alerted siblings, parent silent
        suspected = []
        for p, kids in g["children"].items():
            if p in faulty or not kids:
                continue
            hit = [k for k in kids if k in faulty]
            if len(hit) >= 2 and len(hit) / len(kids) >= 0.5:
                suspected.append({
                    "device_id": p, "role": g["roles"][p], "alerted_children": hit, "total_children": len(kids),
                    "confidence": round(min(0.85, 0.5 + 0.1 * len(hit)), 2),
                    "note": f"{len(hit)}/{len(kids)} downstream devices are faulty but {p} has no alert; "
                            "possible hidden fault or missing telemetry on this device"})
                gaps.append({"device_id": p, "reason": "parent of multiple faulty devices has no telemetry/alert; verify"})

        # Overall scope
        order = ["single_device", "segment", "upstream_aggregation"]
        scopes = [r["blast_radius"]["scope"] for r in root_candidates] + (["segment"] if suspected else [])
        scope = max((s for s in scopes if s in order), key=order.index, default="unknown")
        affected = sorted({x for r in root_candidates for x in r["blast_radius"]["affected_devices"]})

        conf = 0.90
        notes = []
        if g["redundant"]:
            conf -= 0.10
            notes.append("Redundant links exist; traffic may reroute, so casualty inference is less certain")
        if g["disconnected"]:
            conf -= 0.05
        if any(r is None for r in g["roles"].values()):
            conf -= 0.10
            notes.append("Some nodes have missing/unknown roles; scope classification is approximate")
        if unknown_alerts:
            conf -= 0.40 * len(unknown_alerts) / (len(unknown_alerts) + len(alerted) or 1)
            notes.append("Some alerted devices are not in the topology")

        main = root_candidates[0]["device_id"] if root_candidates else "n/a"
        msg.update(
            status="scope_determined", confidence=round(max(conf, 0.10), 2), affected_devices=affected,
            summary=(f"{len(roots)} independent root candidate(s), {len(casualties)} downstream casualt"
                     f"{'y' if len(casualties) == 1 else 'ies'}; widest scope: {scope}; first root: {main}"),
            dependency_chains={d: self._chain(g, d) for d in sorted(faulty)},
            root_candidates=root_candidates, casualties=casualties, suspected_common_upstream=suspected,
            blast_radius={"scope": scope, "affected_count": len(affected), "affected_devices": affected},
            visibility_gaps=gaps, warnings=warnings,
            limitations=notes + (["Hierarchy inferred from the tree rooted at "
                                  f"{g['root']}; assumes traffic follows the primary path"]))
        return msg


def analyze(topology: Any, alerts: Iterable[Any], **kwargs) -> dict:
    return TopologyAgent().analyze(topology, alerts, **kwargs)
