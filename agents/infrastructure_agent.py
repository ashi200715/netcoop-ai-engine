# Implement checks for host, device, capacity, and infrastructure health.
"""Infrastructure Health Agent (Member 3).

Inspects the *machines* that carry traffic (switches, routers, APs): CPU, memory,
temperature, egress queue drops, interface/CRC errors and device status. It turns
raw counters into a standardized ``agent_finding`` message for the Arbiter.

Message contract (schema_version "1.0")
---------------------------------------
Input telemetry: list of per-device samples (or {"records"|"devices"|"telemetry": [...]},
or {device_id: {...}}). Canonical fields and units:

    device_id         str    unique device name
    timestamp         str    ISO-8601 (optional; used only for ordering)
    status            str    up | down | degraded | unknown
    cpu_pct           float  0-100  %
    memory_pct        float  0-100  %
    temperature_c     float  degrees Celsius
    queue_drops       float  egress queue drops per sample window (packets)
    interface_errors  float  CRC / input error count (errors)

Common aliases (cpu_utilization, egress_queue_drops, crc_errors, ...) and units
(cpu_ratio 0-1, temperature_f) are normalized; every conversion is reported in
``warnings`` so nothing is changed silently.

Output: one JSON-serializable dict, ``message_type = "agent_finding"``. See
``InfrastructureAgent.analyze``. The agent never raises on bad input; it returns
``status = "error"`` or ``"insufficient_evidence"`` with the reasons instead.

Confidence method (documented, deterministic)
---------------------------------------------
Each hypothesis starts from a base value reflecting how *direct* the evidence is
(CRC errors = direct proof of layer-1 trouble -> high; a CPU spike -> moderate),
then is adjusted for persistence (+) / transience (-). The finding-level
confidence = best hypothesis confidence * (0.7 + 0.3 * metric_coverage), so
missing metrics lower confidence instead of being ignored.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = "1.0"
AGENT_NAME = "infrastructure_agent"
CONFIDENCE_METHOD = (
    "base(evidence directness) +/- persistence adjustment, scaled by "
    "(0.7 + 0.3 * metric_coverage)"
)

METRICS = ("cpu_pct", "memory_pct", "temperature_c", "queue_drops", "interface_errors")

FIELD_ALIASES: Dict[str, List[str]] = {
    "device_id": ["device_id", "device", "hostname", "node_id", "node", "id", "name"],
    "timestamp": ["timestamp", "ts", "time"],
    "status": ["status", "device_status", "state"],
    "cpu_pct": ["cpu_pct", "cpu_utilization_pct", "cpu_utilization", "cpu_percent", "cpu_usage", "cpu"],
    "memory_pct": ["memory_pct", "memory_utilization_pct", "memory_utilization", "mem_pct",
                   "memory_percent", "memory_usage", "memory", "mem"],
    "temperature_c": ["temperature_c", "temp_c", "temperature", "temp"],
    "queue_drops": ["queue_drops", "egress_queue_drops", "egress_drops", "output_drops", "queue_drop_count"],
    "interface_errors": ["interface_errors", "crc_errors", "if_errors", "input_errors", "rx_errors"],
}
# (source key, target field, multiplier, offset-fn description)
UNIT_ALIASES = [
    ("cpu_ratio", "cpu_pct", lambda v: v * 100.0, "ratio 0-1 -> %"),
    ("cpu_fraction", "cpu_pct", lambda v: v * 100.0, "fraction 0-1 -> %"),
    ("memory_ratio", "memory_pct", lambda v: v * 100.0, "ratio 0-1 -> %"),
    ("memory_fraction", "memory_pct", lambda v: v * 100.0, "fraction 0-1 -> %"),
    ("temperature_f", "temperature_c", lambda v: (v - 32.0) * 5.0 / 9.0, "Fahrenheit -> Celsius"),
]
STATUS_MAP = {
    "up": "up", "online": "up", "active": "up", "ok": "up", "healthy": "up",
    "down": "down", "offline": "down", "unreachable": "down", "failed": "down",
    "degraded": "degraded", "warning": "degraded", "impaired": "degraded",
}


@dataclass(frozen=True)
class Thresholds:
    cpu_warn: float = 70.0
    cpu_crit: float = 90.0
    mem_warn: float = 80.0
    mem_crit: float = 92.0
    temp_warn: float = 70.0
    temp_crit: float = 85.0
    queue_drops_warn: float = 50.0
    queue_drops_crit: float = 150.0
    crc_warn: float = 1.0          # any non-zero CRC count is meaningful
    crc_crit: float = 100.0
    sustained_fraction: float = 0.5  # share of samples over warn to call it "sustained"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _num(value: Any) -> Optional[float]:
    """Return a finite float or None (bools, NaN, inf, junk -> None)."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        f = float(str(value).strip().rstrip("%")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _evidence(device: str, metric: str, value: float, unit: str, interp: str, severity: str) -> dict:
    return {"device_id": device, "metric": metric, "value": round(value, 2), "unit": unit,
            "interpretation": interp, "severity": severity}


class InfrastructureAgent:
    def __init__(self, thresholds: Optional[Thresholds] = None):
        self.t = thresholds or Thresholds()

    # ------------------------------------------------------------------ input
    @staticmethod
    def _extract_records(telemetry: Any) -> Tuple[List[dict], List[dict]]:
        problems: List[dict] = []
        if isinstance(telemetry, dict):
            for key in ("records", "devices", "telemetry", "samples"):
                if isinstance(telemetry.get(key), list):
                    telemetry = telemetry[key]
                    break
            else:
                if telemetry and all(isinstance(v, dict) for v in telemetry.values()):
                    telemetry = [{"device_id": k, **v} for k, v in telemetry.items()]
                else:
                    return [], [{"issue": "telemetry dict has no recognizable record list"}]
        if not isinstance(telemetry, (list, tuple)):
            return [], [{"issue": f"telemetry must be a list or dict, got {type(telemetry).__name__}"}]
        records = []
        for i, r in enumerate(telemetry):
            if isinstance(r, dict):
                records.append(r)
            else:
                problems.append({"index": i, "issue": "record is not an object; skipped"})
        return records, problems

    def _normalize(self, raw: dict, index: int, warnings: List[dict]) -> Optional[dict]:
        rec: Dict[str, Any] = {}
        lower = {str(k).lower(): v for k, v in raw.items()}
        for field, aliases in FIELD_ALIASES.items():
            for a in aliases:
                if a in lower and lower[a] is not None:
                    rec[field] = lower[a]
                    break
        for src, target, fn, note in UNIT_ALIASES:
            if target not in rec and src in lower:
                v = _num(lower[src])
                if v is not None:
                    rec[target] = fn(v)
                    warnings.append({"index": index, "field": src, "issue": f"converted {note}"})
        dev = rec.get("device_id")
        if not isinstance(dev, str) or not dev.strip():
            warnings.append({"index": index, "issue": "missing device_id; record skipped"})
            return None
        out: Dict[str, Any] = {"device_id": dev.strip(), "timestamp": rec.get("timestamp")}
        status = str(rec.get("status", "unknown")).strip().lower()
        out["status"] = STATUS_MAP.get(status, "unknown")
        for m in METRICS:
            if m not in rec:
                continue
            v = _num(rec[m])
            if v is None:
                warnings.append({"index": index, "device_id": out["device_id"], "field": m,
                                 "issue": f"non-numeric value {rec[m]!r} ignored"})
            elif v < 0 or (m in ("cpu_pct", "memory_pct") and v > 100):
                warnings.append({"index": index, "device_id": out["device_id"], "field": m,
                                 "issue": f"out-of-range value {v} ignored"})
            else:
                out[m] = v
        return out

    # --------------------------------------------------------------- analysis
    def _analyze_device(self, dev: str, samples: List[dict]) -> Tuple[List[dict], dict]:
        t = self.t
        n = len(samples)

        def series(m): return [s[m] for s in samples if m in s]
        def latest(m):
            vals = series(m)
            return vals[-1] if vals else None
        def sustained(m, warn):
            vals = series(m)
            return bool(vals) and sum(v >= warn for v in vals) / len(vals) >= t.sustained_fraction

        cpu, mem, temp = latest("cpu_pct"), latest("memory_pct"), latest("temperature_c")
        qd, crc = latest("queue_drops"), latest("interface_errors")
        status = samples[-1]["status"]
        hyps: List[dict] = []

        def add(name, conf, ev, note):
            hyps.append({"device_id": dev, "hypothesis": name, "confidence": round(min(max(conf, 0.0), 0.97), 2),
                         "evidence": ev, "note": note})

        # Direct physical-layer evidence
        if crc is not None and crc >= t.crc_warn:
            conf = 0.80 + (0.10 if crc >= t.crc_crit else 0.0)
            vals = series("interface_errors")
            if len(vals) >= 2 and vals[-1] > vals[0]:
                conf += 0.05
            add("physical_layer_degradation", conf,
                [_evidence(dev, "interface_errors", crc, "errors",
                           "Non-zero CRC/interface errors indicate a damaged cable, fiber patch or failing transceiver",
                           "critical" if crc >= t.crc_crit else "warning")],
                "Inspect/replace cable or transceiver on the erroring interface.")

        # Queue drops: buffer exhaustion vs control-plane
        cpu_normal = cpu is not None and cpu < t.cpu_warn
        if qd is not None and qd >= t.queue_drops_warn:
            ev = [_evidence(dev, "queue_drops", qd, "packets",
                            "Egress queue drops show the switch buffer is overflowing",
                            "critical" if qd >= t.queue_drops_crit else "warning")]
            if cpu_normal:
                conf = 0.75 + (0.07 if qd >= t.queue_drops_crit else 0.0)
                ev.append(_evidence(dev, "cpu_pct", cpu, "%", "CPU normal: device is not control-plane bound", "info"))
                add("egress_buffer_exhaustion", conf + (0.05 if sustained("queue_drops", t.queue_drops_warn) and n > 1 else 0), ev,
                    "Localized throughput burst or uplink congestion; review QoS/queue config and uplink utilization.")
            elif cpu is None:
                add("egress_buffer_exhaustion", 0.55, ev,
                    "CPU not reported, so control-plane overload cannot be ruled out.")
            else:
                add("egress_buffer_exhaustion", 0.55, ev,
                    "CPU is also high; drops may be a symptom of overload rather than only buffer sizing.")

        # CPU / memory
        cpu_hi = cpu is not None and cpu >= t.cpu_warn
        mem_hi = mem is not None and mem >= t.mem_warn
        if cpu_hi and mem_hi:
            crit = cpu >= t.cpu_crit or mem >= t.mem_crit
            conf = 0.65 + (0.15 if crit else 0.0) + (0.05 if sustained("cpu_pct", t.cpu_warn) and n > 1 else 0)
            add("control_plane_overload", conf,
                [_evidence(dev, "cpu_pct", cpu, "%", "High CPU utilization", "critical" if cpu >= t.cpu_crit else "warning"),
                 _evidence(dev, "memory_pct", mem, "%", "High memory utilization", "critical" if mem >= t.mem_crit else "warning")],
                "Suspect route flapping, DoS/broadcast pressure or runaway process; check control-plane logs.")
        elif cpu_hi:
            transient = n >= 3 and not sustained("cpu_pct", t.cpu_warn)
            conf = 0.40 if transient else (0.65 if cpu >= t.cpu_crit else 0.55)
            add("transient_cpu_spike" if transient else "cpu_saturation", conf,
                [_evidence(dev, "cpu_pct", cpu, "%",
                           "Brief CPU spike; most samples normal" if transient else "Sustained high CPU",
                           "warning" if cpu < t.cpu_crit else "critical")],
                "Moderate confidence: CPU alone is circumstantial evidence.")
        elif mem_hi:
            add("memory_pressure", 0.50,
                [_evidence(dev, "memory_pct", mem, "%", "High memory utilization", "warning")],
                "Possible leak or table growth; confirm with a trend over time.")

        # Thermal
        if temp is not None and temp >= t.temp_warn:
            crit = temp >= t.temp_crit
            add("thermal_stress", 0.75 if crit else 0.55,
                [_evidence(dev, "temperature_c", temp, "C", "Internal temperature above safe envelope",
                           "critical" if crit else "warning")],
                "Check fans, airflow and ambient temperature.")

        # Reported status
        if status == "down":
            add("device_unreachable", 0.85,
                [{"device_id": dev, "metric": "status", "value": "down", "unit": None,
                  "interpretation": "Device reported down; may be power/hardware failure or only a management-plane outage",
                  "severity": "critical"}],
                "Verify via a second path (neighbor link state, console, PoE).")
        elif status == "degraded":
            add("device_degraded", 0.50,
                [{"device_id": dev, "metric": "status", "value": "degraded", "unit": None,
                  "interpretation": "Device self-reports degraded state", "severity": "warning"}],
                "Review device logs for the degradation reason.")

        hyps.sort(key=lambda h: h["confidence"], reverse=True)
        sev = [e["severity"] for h in hyps for e in h["evidence"]]
        state = "critical" if "critical" in sev else "degraded" if "warning" in sev else (
            "healthy" if any(m in samples[-1] for m in METRICS) or status == "up" else "unknown")
        health = {"state": state, "status": status, "samples": n,
                  "metrics": {m: latest(m) for m in METRICS if latest(m) is not None}}
        return hyps, health

    # ------------------------------------------------------------ public API
    def analyze(self, telemetry: Any, expected_devices: Optional[Iterable[str]] = None,
                incident_id: Optional[str] = None) -> dict:
        """Return an ``agent_finding`` message. Never raises on bad input."""
        msg = {"schema_version": SCHEMA_VERSION, "message_type": "agent_finding", "agent": AGENT_NAME,
               "incident_id": incident_id, "timestamp": _now(), "confidence_method": CONFIDENCE_METHOD,
               "units": {"cpu_pct": "%", "memory_pct": "%", "temperature_c": "C",
                         "queue_drops": "packets", "interface_errors": "errors"}}
        try:
            return self._analyze(msg, telemetry, expected_devices)
        except Exception as exc:  # defensive: orchestrator must always get a message
            msg.update(status="error", confidence=0.0, affected_devices=[], hypotheses=[], device_health={},
                       supporting_evidence=[], visibility_gaps=[], warnings=[],
                       summary="Infrastructure agent failed internally", limitations=[f"internal error: {exc!r}"])
            return msg

    def _analyze(self, msg: dict, telemetry: Any, expected_devices: Optional[Iterable[str]]) -> dict:
        warnings: List[dict] = []
        raw, problems = self._extract_records(telemetry)
        warnings.extend(problems)
        by_dev: Dict[str, List[dict]] = {}
        for i, r in enumerate(raw):
            rec = self._normalize(r, i, warnings)
            if rec:
                by_dev.setdefault(rec["device_id"], []).append(rec)
        for samples in by_dev.values():
            if all(s.get("timestamp") for s in samples):
                samples.sort(key=lambda s: str(s["timestamp"]))

        usable = {d: s for d, s in by_dev.items() if any(m in x for x in s for m in METRICS) or s[-1]["status"] != "unknown"}
        gaps: List[dict] = []
        for d in sorted(set(by_dev) - set(usable)):
            gaps.append({"device_id": d, "reason": "no usable metrics or status reported"})
        if expected_devices is not None:
            for d in sorted(set(expected_devices) - set(by_dev)):
                gaps.append({"device_id": d, "reason": "no telemetry received (SNMP/stream missing or unreachable)"})

        if not usable:
            msg.update(status="insufficient_evidence", confidence=0.0, affected_devices=[], hypotheses=[],
                       device_health={}, supporting_evidence=[], visibility_gaps=gaps, warnings=warnings,
                       summary="No usable infrastructure telemetry; no diagnosis can be made",
                       limitations=["Input empty, malformed or missing required fields"])
            return msg

        hyps: List[dict] = []
        health: Dict[str, dict] = {}
        present = 0
        for d, samples in sorted(usable.items()):
            h, hl = self._analyze_device(d, samples)
            hyps.extend(h)
            health[d] = hl
            present += sum(1 for m in METRICS if any(m in s for s in samples))
            for m in METRICS:
                if not any(m in s for s in samples):
                    gaps.append({"device_id": d, "reason": f"metric '{m}' not reported"})
        coverage = present / (len(usable) * len(METRICS))
        hyps.sort(key=lambda h: h["confidence"], reverse=True)
        affected = sorted({h["device_id"] for h in hyps})
        evidence = [e for h in hyps for e in h["evidence"]]
        scale = 0.7 + 0.3 * coverage

        limitations = []
        if gaps:
            limitations.append("Some devices/metrics were not visible; absence of evidence is not evidence of health")
        if warnings:
            limitations.append("Some input values were converted or ignored (see warnings)")

        if hyps:
            top = hyps[0]
            msg.update(status="anomaly_detected", confidence=round(top["confidence"] * scale, 2),
                       summary=f"{top['hypothesis'].replace('_', ' ')} suspected on {top['device_id']}")
        else:
            msg.update(status="healthy", confidence=round(0.9 * scale, 2),
                       summary="All inspected devices are within safe operating envelopes")
        msg.update(affected_devices=affected, hypotheses=hyps, device_health=health,
                   supporting_evidence=evidence, visibility_gaps=gaps, warnings=warnings,
                   limitations=limitations)
        return msg


def analyze(telemetry: Any, **kwargs) -> dict:
    """Module-level convenience wrapper."""
    return InfrastructureAgent().analyze(telemetry, **kwargs)
