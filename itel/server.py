"""HTTP server: one JSON API plus the static dashboard.

Endpoints
---------
GET  /api/overview            scenario summary, topology, KPI headline
GET  /api/topology            switches, links, hosts (for the heat map)
GET  /api/queues?t=...        per-port queue-depth heat map at a time
GET  /api/slow                list of slow transactions
GET  /api/attribute?txn_id=   per-hop breakdown, dominant hop, co-flows
GET  /api/sampling            sampling-policy time series (rate, triggers)
GET  /api/evaluation          acceptance results (fixed vs adaptive)
GET  /api/timeseries?port=..  one port's queue depth over time
POST /api/rebuild             re-run the simulation with a new seed
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

from .collector import Collector
from .config import SLOW_TXN_US, TelemetryConfig
from .evaluate import (RunArtifacts, build_run, evaluate_policy,
                       standard_scenario, sweep_fixed)
from .sampling import (AdaptiveSampling, calibration_strata, capture_rate,
                       estimate_trigger_volume, optimize_budget,
                       total_packet_rate)
from .sim import WireTraces, materialise_wire
from .topology import MAX_PORTS

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(os.path.dirname(HERE), "web")

BUCKET_US = 20_000.0     # 20 ms buckets for every time series


@dataclass
class Session:
    seed: int = 7
    budget_bytes_per_s: float = 400_000.0
    art: Optional[RunArtifacts] = None
    adaptive: Optional[dict] = None
    fixed: Optional[dict] = None
    wire_a: Optional[WireTraces] = None
    wire_f: Optional[WireTraces] = None
    col_a: Optional[Collector] = None
    col_f: Optional[Collector] = None
    topology: dict = field(default_factory=dict)
    slow_txns: List[dict] = field(default_factory=list)
    ready: bool = False
    error: Optional[str] = None
    built_at: float = 0.0

    # ------------------------------------------------------------------
    def build(self) -> None:
        t0 = time.time()
        sc = standard_scenario()
        sc.seed = self.seed
        art = build_run(sc)
        strata = calibration_strata(art.dp, sc.duration_us)
        trig = estimate_trigger_volume(art.dp, art.tcfg)
        opt = optimize_budget(strata, art.tcfg, self.budget_bytes_per_s,
                              trigger_bytes_per_s=trig,
                              strata_pkt_rate=total_packet_rate(
                                  art.dp, sc.duration_us))
        ad = AdaptiveSampling(opt["base_prob"], opt["stratum_prob"])
        pa = ad.apply(art.dp, art.tcfg, sc.duration_us)
        a_res = evaluate_policy(art, ad, self.budget_bytes_per_s, with_collector=False)
        fx = sweep_fixed(art, a_res["volume_bytes_per_s"])
        pf = fx.apply(art.dp, art.tcfg, sc.duration_us)
        f_res = evaluate_policy(art, fx, self.budget_bytes_per_s, with_collector=False)
        wire_a = materialise_wire(art.dp, art.tcfg, pa.marking.marked,
                                  art.clocks, sc.seed)
        col_a = Collector(art.topo)
        col_a.ingest(wire_a)
        wire_f = materialise_wire(art.dp, art.tcfg, pf.marking.marked,
                                  art.clocks, sc.seed)
        col_f = Collector(art.topo)
        col_f.ingest(wire_f)

        self.art = art
        self.wire_a = wire_a
        self.wire_f = wire_f
        self.col_a = col_a
        self.col_f = col_f
        self.topology = art.topo.topology_json()
        self.adaptive = {"result": a_res, "policy": pa, "opt": opt}
        self.fixed = {"result": f_res, "policy": pf}
        self.slow_txns = self._collect_slow()
        self.ready = True
        self.built_at = time.time() - t0

    # ------------------------------------------------------------------
    def _collect_slow(self) -> List[dict]:
        """Slow transactions the collector can attribute, worst first."""
        art = self.art
        rows = []
        for tx in art.traffic.txns:
            if np.isnan(tx.true_latency_us) or tx.true_latency_us < SLOW_TXN_US:
                continue
            a = self.col_a.attribute(tx.txn_id)
            observed = "error" not in a
            rows.append({
                "txn_id": int(tx.txn_id),
                "cls": tx.cls,
                "client": tx.client, "server": tx.server,
                "t_us": round(float(tx.t0), 1),
                "latency_us": round(float(tx.true_latency_us), 1),
                "net_us": round(float(tx.net_latency_us), 1),
                # the network's share of the slowdown, which is what the
                # fabric team can actually act on
                "network_slow": bool(tx.net_latency_us >= SLOW_TXN_US),
                "observed": observed,
                "captured_by_adaptive": observed,
                "true_dominant_hop": int(tx.dominant_hop),
                "true_dominant_queue_us": round(float(tx.dominant_queue_us), 1),
                "pred_dominant_hop": (a["dominant_hop"]["hop"]
                                      if observed and a["dominant_hop"] else None),
                "pred_switch": (a["dominant_hop"]["switch"]
                                if observed and a["dominant_hop"] else None),
                "pred_queue_us": (a["dominant_hop"]["queue_us"]
                                  if observed and a["dominant_hop"] else None),
                "five_tuple": list(tx and art.traffic.flows[tx.resp_flow].five_tuple),
            })
        rows.sort(key=lambda r: (-r["network_slow"], -r["net_us"]))
        return rows


SESSION = Session()
_BUILD_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
def _bucket_index(t_us: float) -> int:
    return int(t_us // BUCKET_US)


def queue_heatmap(sess: Session, t_us: Optional[float]) -> dict:
    """Per-switch, per-port queue depth at (or just before) a time."""
    art = sess.art
    dp = art.dp
    dur = art.scenario.duration_us
    if t_us is None:
        t_us = dur * 0.5
    # find the nearest packet time slice: take records in [t, t+5ms]
    t1 = t_us + 5_000.0
    ports: Dict[int, dict] = {}
    for h in range(dp.max_h):
        m = dp.n_hops > h
        if not m.any():
            continue
        idx = np.nonzero(m & (dp.t_arrive[:, h] >= t_us)
                         & (dp.t_arrive[:, h] < t1))[0]
        if not idx.size:
            continue
        pk = dp.hop_port[idx, h]
        for p in np.unique(pk):
            sel = idx[pk == p]
            depth = float(dp.qdepth_bytes[sel, h].max())
            port = int(p) - int(dp.hop_sw[sel[0], h]) * MAX_PORTS
            sw = int(dp.hop_sw[sel[0], h])
            cur = ports.get(int(p))
            d_us = depth / float(dp.rate_eff[sel[0], h]) if dp.rate_eff[sel[0], h] > 0 else 0.0
            if cur is None or depth > cur["qdepth_bytes"]:
                ports[int(p)] = {
                    "port_key": int(p),
                    "switch": art.topo.switches[sw].switch_id,
                    "switch_idx": sw,
                    "port": port,
                    "qdepth_bytes": depth,
                    "queue_us": d_us,
                    "util": float(d_us / 16_000.0),
                }
    # switch-level aggregate for colouring the topology
    sw_level: Dict[str, float] = {}
    for p in ports.values():
        sw_level[p["switch"]] = max(sw_level.get(p["switch"], 0.0), p["queue_us"])
    return {"t_us": t_us, "ports": sorted(ports.values(),
                                          key=lambda x: -x["queue_us"])[:200],
            "switches": sw_level}


def queue_timeseries(sess: Session, port_key: Optional[int]) -> dict:
    art = sess.art
    dp = art.dp
    n_b = int(art.scenario.duration_us // BUCKET_US) + 1
    if port_key is None:
        # aggregate: the worst queue anywhere per bucket
        out = np.zeros(n_b)
        for h in range(dp.max_h):
            m = dp.n_hops > h
            if not m.any():
                continue
            b = (dp.t_arrive[m, h] // BUCKET_US).astype(np.int64)
            q = dp.qdepth_bytes[m, h]
            hull = np.zeros(n_b)
            np.maximum.at(hull, b, q)
            out = np.maximum(out, hull)
        series = out
    else:
        series = np.zeros(n_b)
        for h in range(dp.max_h):
            m = (dp.n_hops > h) & (dp.hop_port[:, h] == port_key)
            if not m.any():
                continue
            b = (dp.t_arrive[m, h] // BUCKET_US).astype(np.int64)
            np.maximum.at(series, b, dp.qdepth_bytes[m, h])
    return {
        "t_us": [round((i + 0.5) * BUCKET_US, 1) for i in range(n_b)],
        "qdepth_bytes": [round(float(x), 1) for x in series],
        "bucket_us": BUCKET_US,
    }


def _top_rules(sess: Session, limit: int = 24) -> list:
    """The sampling match-action rules, ranked by probability."""
    pol = sess.adaptive["policy"]
    d = sess.adaptive.get("opt", {}).get("stratum_prob", {})
    rows = sorted(d.items(), key=lambda kv: -kv[1])[:limit]
    return [{"class": c, "path": pth, "p": round(float(v), 5)}
            for (c, pth), v in rows]


def sampling_series(sess: Session) -> dict:
    art = sess.art
    dp = art.dp
    pol = sess.adaptive["policy"]
    n_b = int(art.scenario.duration_us // BUCKET_US) + 1
    t = dp.traffic.t_send
    b = (t // BUCKET_US).astype(np.int64)
    marked = pol.marking.marked
    boost = pol.marking.boost_mask
    trig = pol.marking.trigger_t
    eff = np.zeros(n_b)
    boost_share = np.zeros(n_b)
    for i in range(n_b):
        m = b == i
        if not m.any():
            continue
        eff[i] = marked[m].mean()
        boost_share[i] = boost[m].mean()
    trig_b = np.bincount((trig // BUCKET_US).astype(np.int64), minlength=n_b) \
        if trig.size else np.zeros(n_b)
    # fixed policy's effective rate, for the comparison line
    fx = sess.fixed["policy"]
    fm = fx.marking.marked
    eff_f = np.zeros(n_b)
    for i in range(n_b):
        m = b == i
        if m.any():
            eff_f[i] = fm[m].mean()
    return {
        "t_us": [round((i + 0.5) * BUCKET_US, 1) for i in range(n_b)],
        "adaptive_rate": [round(float(x), 5) for x in eff],
        "fixed_rate": [round(float(x), 5) for x in eff_f],
        "boost_share": [round(float(x), 5) for x in boost_share],
        "triggers": [int(x) for x in trig_b],
        "base_prob": float(np.mean(pol.marking.base_prob)),
        "n_rules": len(sess.adaptive.get("opt", {}).get("stratum_prob", {})),
        "budget_bytes_per_s": sess.budget_bytes_per_s,
        "planned_bytes_per_s": float(sess.adaptive.get("opt", {}).get(
            "expected_bytes_per_s", 0.0)),
        "actual_bytes_per_s": float(sess.adaptive["result"]["volume_bytes_per_s"]),
        "rules": _top_rules(sess),
        "trigger_window_us": art.tcfg.trigger_window_us,
        "queue_trigger_us": art.tcfg.queue_trigger_us,
    }


def evaluation_payload(sess: Session) -> dict:
    a = sess.adaptive["result"]
    f = sess.fixed["result"]
    art = sess.art
    col = sess.col_a
    # attribution + delay error on the adaptive collector
    from .evaluate import score_collector
    sc = score_collector(art, col, sess.wire_a)
    scf = score_collector(art, sess.col_f, sess.wire_f)
    return {
        "scenario": {
            "seed": int(art.scenario.seed),
            "packets": int(art.traffic.n_packets),
            "duration_s": art.scenario.duration_us / 1e6,
            "transactions": len(art.traffic.txns),
            "slow_transactions": len(sess.slow_txns),
            "int_budget_bytes": art.tcfg.budget_bytes,
            "max_records": (art.tcfg.budget_bytes - 24) // art.tcfg.hop_bytes,
            "congestion_episodes": len(art.scenario.scenario.congestion),
            "impaired_links": len(art.scenario.scenario.link_impair),
        },
        "adaptive": {**{k: a[k] for k in ("volume_bytes_per_s", "volume_mbps",
                                          "sampled_packets", "capture",
                                          "boost_packets", "triggers", "params")},
                     "volume": sess.adaptive["policy"].volume, **sc},
        "fixed": {**{k: f[k] for k in ("volume_bytes_per_s", "volume_mbps",
                                       "sampled_packets", "capture", "params")},
                  "volume": sess.fixed["policy"].volume, **scf},
        "optimizer": {k: v for k, v in sess.adaptive["opt"].items()
                      if k != "stratum_prob"},
    }


# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "INTCollector/1.0"

    def log_message(self, fmt, *args):
        pass

    def do_HEAD(self):
        # HEAD is used by health checks; answer like GET but without a body
        u = urlparse(self.path)
        path = os.path.join(WEB_DIR, (u.path.lstrip("/") or "index.html"))
        ctype = _ctype(path) if os.path.isfile(path) else "text/plain"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, default=_json_default).encode(),
                   "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path.startswith("/api/"):
                return self._api(u.path, q)
            rel = u.path.lstrip("/") or "index.html"
            if ".." in rel:
                return self._send(403, b"forbidden", "text/plain")
            path = os.path.join(WEB_DIR, rel)
            if not os.path.isfile(path):
                path = os.path.join(WEB_DIR, "index.html")
            ctype = _ctype(path)
            with open(path, "rb") as fh:
                self._send(200, fh.read(), ctype)
        except Exception:
            self._json({"error": traceback.format_exc()}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/rebuild":
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                data = json.loads(body or b"{}")
                seed = int(data.get("seed", SESSION.seed))
                budget = float(data.get("budget_bytes_per_s",
                                        SESSION.budget_bytes_per_s))
                threading.Thread(target=_rebuild, args=(seed, budget),
                                 daemon=True).start()
                return self._json({"ok": True, "building": True})
            except Exception:
                return self._json({"error": traceback.format_exc()}, 500)
        self._json({"error": "not found"}, 404)

    # ------------------------------------------------------------------
    def _api(self, path: str, q: dict):
        s = SESSION
        if path == "/api/status":
            return self._json({"ready": s.ready, "error": s.error,
                               "build_seconds": round(s.built_at, 1),
                               "seed": s.seed,
                               "budget_mbps": round(s.budget_bytes_per_s / 1e6, 3),
                               "budget_bytes_per_s": s.budget_bytes_per_s})
        if not s.ready:
            return self._json({"error": "simulation still building",
                               "ready": False}, 503)
        if path == "/api/overview":
            return self._json(_overview(s))
        if path == "/api/topology":
            return self._json(s.topology)
        if path == "/api/queues":
            t = q.get("t", [None])[0]
            return self._json(queue_heatmap(s, float(t) if t else None))
        if path == "/api/timeseries":
            p = q.get("port", [None])[0]
            return self._json(queue_timeseries(s, int(p) if p else None))
        if path == "/api/slow":
            return self._json({"slow": s.slow_txns})
        if path == "/api/attribute":
            txn = int(q.get("txn_id", ["0"])[0])
            return self._json(_attribute_payload(s, txn))
        if path == "/api/sampling":
            return self._json(sampling_series(s))
        if path == "/api/evaluation":
            return self._json(evaluation_payload(s))
        self._json({"error": "unknown endpoint"}, 404)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if v != v else v
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _ctype(path: str) -> str:
    if path.endswith(".html"):
        return "text/html; charset=utf-8"
    if path.endswith(".js"):
        return "application/javascript; charset=utf-8"
    if path.endswith(".css"):
        return "text/css; charset=utf-8"
    if path.endswith(".svg"):
        return "image/svg+xml"
    return "application/octet-stream"


def _overview(s: Session) -> dict:
    art = s.art
    ev = evaluation_payload(s)
    tr = art.traffic
    lat = np.array([tx.true_latency_us for tx in tr.txns
                    if not np.isnan(tx.true_latency_us)])
    return {
        "seed": s.seed,
        "duration_us": art.scenario.duration_us,
        "packets": int(tr.n_packets),
        "flows": len(tr.flows),
        "transactions": len(tr.txns),
        "latency_ms": {"p50": round(float(np.percentile(lat, 50)) / 1000, 3),
                       "p99": round(float(np.percentile(lat, 99)) / 1000, 3),
                       "p999": round(float(np.percentile(lat, 99.9)) / 1000, 3),
                       "max": round(float(lat.max()) / 1000, 3)},
        "slow_transactions": len(s.slow_txns),
        "faults": {
            "congestion": art.scenario.scenario.congestion,
            "shapers": getattr(art.scenario.scenario, "shapers", []),
            "impaired_links": art.scenario.scenario.link_impair,
            "clock_biases": art.scenario.scenario.clock_bias_ppb,
        },
        "evaluation_headline": {
            "adaptive_capture": ev["adaptive"]["capture"]["capture_rate"],
            "fixed_capture": ev["fixed"]["capture"]["capture_rate"],
            "adaptive_mbps": ev["adaptive"]["volume_mbps"],
            "fixed_mbps": ev["fixed"]["volume_mbps"],
            "attribution_accuracy": ev["adaptive"]["attribution_accuracy"],
            "attribution_accuracy_big": ev["adaptive"]["attribution_accuracy_big"],
            "attribution_accuracy_net": ev["adaptive"]["attribution_accuracy_net"],
            "net_slow_transactions": sum(1 for r in s.slow_txns
                                         if r.get("network_slow")),
            "delay_mae_us": ev["adaptive"]["delay_mae_us"],
        },
    }


def _attribute_payload(s: Session, txn_id: int) -> dict:
    art = s.art
    a = s.col_a.attribute(txn_id)
    if "error" in a:
        return a
    # ground truth, so the dashboard can show right/wrong
    tx = next((t for t in art.traffic.txns if t.txn_id == txn_id), None)
    if tx is not None:
        a["truth"] = {
            "latency_us": round(float(tx.true_latency_us), 1),
            "net_us": round(float(tx.net_latency_us), 1),
            "dominant_hop": int(tx.dominant_hop),
            "dominant_queue_us": round(float(tx.dominant_queue_us), 1),
        }
        a["txn_start_us"] = round(float(tx.t0), 1)
        a["server_thinking_us"] = round(float(tx.server_delay_us), 1)
    a["trigger_events"] = _nearby_triggers(s, a.get("t_start_us", 0.0))
    a["packet_timeline"] = _packet_timeline(s, txn_id)
    a["txn_summary"] = _txn_summary(s, txn_id)
    return a


def _packet_timeline(s: Session, txn_id: int) -> dict:
    """When each response packet left the server and reached the client."""
    art = s.art
    tr = art.traffic
    rows = np.nonzero((tr.txn_id == txn_id) & (tr.role == 1))[0]
    if not rows.size:
        return {}
    t0 = float(tr.t_send[rows].min())
    return {
        "t0_us": t0,
        "send_us": [round(float(x - t0), 2) for x in tr.t_send[rows]],
        "deliver_us": [round(float(x - t0), 2) for x in art.dp.t_deliver[rows]],
        "queue_us": [round(float(np.nansum(art.dp.qdelay[r])), 2) for r in rows],
    }


def _txn_summary(s: Session, txn_id: int) -> dict:
    """The 5-tuple and byte counts an operator would paste into a ticket."""
    art = s.art
    tr = art.traffic
    tx = next((t for t in tr.txns if t.txn_id == txn_id), None)
    if tx is None:
        return {}
    resp = np.nonzero((tr.txn_id == txn_id) & (tr.role == 1))[0]
    req = np.nonzero((tr.txn_id == txn_id) & (tr.role == 0))[0]
    ft = tr.flows[tx.resp_flow].five_tuple
    return {
        "five_tuple": list(ft),
        "client": tx.client,
        "server": tx.server,
        "request_bytes": int(tx.req_bytes),
        "response_bytes": int(tx.resp_bytes),
        "response_packets": int(resp.size),
        "server_processing_us": round(float(tx.server_delay_us), 1),
        "queueing_us": round(float(np.nansum(art.dp.qdelay[resp])), 1),
        "e2e_us": round(float(tx.true_latency_us), 1),
    }


def _nearby_triggers(s: Session, t_us: float, window: float = 200_000.0) -> list:
    mk = s.adaptive["policy"].marking
    if mk.trigger_t.size == 0:
        return []
    m = np.abs(mk.trigger_t - t_us) <= window
    out = []
    for sw, t, port in zip(mk.trigger_sw[m], mk.trigger_t[m],
                           mk.trigger_port[m]):
        out.append({"switch": s.art.topo.switches[int(sw)].switch_id,
                    "t_us": round(float(t), 1),
                    "port": int(port) - int(sw) * MAX_PORTS,
                    "dt_us": round(float(t - t_us), 1)})
    out.sort(key=lambda x: abs(x["dt_us"]))
    return out[:8]


def _rebuild(seed: int, budget: float) -> None:
    with _BUILD_LOCK:
        SESSION.ready = False
        SESSION.error = None
        SESSION.seed = seed
        SESSION.budget_bytes_per_s = budget
        try:
            SESSION.build()
        except Exception:
            SESSION.error = traceback.format_exc()
            SESSION.ready = False


def serve(host: str = "127.0.0.1", port: int = 8765, seed: int = 7,
          budget: float = 400_000.0) -> None:
    SESSION.seed = seed
    SESSION.budget_bytes_per_s = budget
    with _BUILD_LOCK:
        SESSION.ready = False
        SESSION.build()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"INT telemetry console on http://{host}:{port}  "
          f"(seed {seed}, {SESSION.art.traffic.n_packets} packets, "
          f"{SESSION.built_at:.1f}s build)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("bye")
