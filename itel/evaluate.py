"""Acceptance evaluation: fixed vs adaptive sampling, attribution, delay error.

Produces a JSON report that both the CLI and the dashboard read.

Metrics
-------
capture_rate      slow transactions with *any* telemetry on the response leg
attribution_acc   predicted dominant hop equals the true dominant hop
                  (reported overall and for queueing > 1 ms)
delay_mae_us      per-hop queueing delay error (collector vs simulator)
volume_mbps       collector ingress bytes / wall second
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .collector import Collector, PacketTrace
from .config import (DEFAULT_INT_BUDGET_BYTES, SLOW_TXN_US, ScenarioConfig,
                     TelemetryConfig, TrafficConfig)
from .sampling import (AdaptiveSampling, FixedSampling, StratifiedSampling,
                       calibration_strata, capture_rate, estimate_trigger_volume,
                       estimate_volume, optimize_budget, total_packet_rate)
from .sim import (DataPlane, WireTraces, annotate_transactions,
                  build_impairment, make_clocks, materialise_wire,
                  run_data_plane)
from .topology import Topology, build_fat_tree
from .traffic import Traffic, build_traffic, default_classes


# ---------------------------------------------------------------------------
def default_scenario() -> ScenarioConfig:
    """Faults that reproduce the operational complaint.

    Three intermittent episodes on the leaf egress facing the hot storage
    node, one congested spine uplink, and one lossy link.  All are time
    bounded so a policy that samples uniformly has a fair chance to see the
    quiet periods too.
    """
    return ScenarioConfig(
        # one port was shaped to 40% of line rate by a bad change window and
        # nobody noticed: a persistent configuration fault
        shapers=[("agg-1-0", 2, 0.40)],
        congestion=[
            # a port that degrades intermittently: drain rate drops for a
            # bounded window, then recovers
            ("leaf-2-1", 5, 1_500_000, 1_650_000, 0.45),
        ],
        link_impair=[
            # (src, dst, t0, t1, retransmit probability)
            ("spine-1", "agg-3-0", 300_000, 900_000, 0.03),
            ("leaf-3-0", "agg-3-0", 1_200_000, 1_800_000, 0.02),
        ],
        clock_bias_ppb={"spine-0": 80.0, "super-spine-1": -60.0},
    )


@dataclass
class Scenario:
    name: str
    duration_us: float
    flows_per_second: float
    episodes: List[dict]
    scenario: ScenarioConfig
    seed: int = 7


def standard_scenario() -> Scenario:
    return Scenario(
        name="dc-incident",
        duration_us=2_000_000.0,
        flows_per_second=2_500.0,
        episodes=[
            # incast onto hot storage node 0.  The episode's duty cycle is
            # low (a few percent) because that is what "occasional" means
            # operationally: most of the time the fabric is clean.
            {"t_start": 200_000, "t_end": 1_800_000, "server_idx": 0,
             "period_us": 40_000, "burst_us": 2_400, "n_senders": 18,
             "sender_rate_bpus": 122.0},
            # a second, lighter burst target so the fabric has >1 hot spot
            {"t_start": 900_000, "t_end": 1_400_000, "server_idx": 2,
             "period_us": 55_000, "burst_us": 1_800, "n_senders": 11,
             "sender_rate_bpus": 120.0},
        ],
        scenario=default_scenario(),
    )


@dataclass
class RunArtifacts:
    topo: Topology
    traffic: Traffic
    dp: DataPlane
    clocks: object
    scenario: Scenario
    tcfg: TelemetryConfig


def build_run(sc: Scenario, tcfg: Optional[TelemetryConfig] = None,
              classes=None) -> RunArtifacts:
    topo = build_fat_tree()
    cls = classes or default_classes(topo)
    tcfg = tcfg or TelemetryConfig()
    traffic = build_traffic(
        TrafficConfig(classes=cls, duration_us=sc.duration_us,
                      flows_per_second=sc.flows_per_second, seed=sc.seed),
        topo, sc.episodes)
    clocks = make_clocks(topo, sc.scenario.clock_bias_ppb, sc.seed)
    imp = build_impairment(topo, sc.scenario)
    dp = run_data_plane(traffic, topo, imp, clocks, sc.seed)
    annotate_transactions(dp)
    return RunArtifacts(topo, traffic, dp, clocks, sc, tcfg)


# ---------------------------------------------------------------------------
def evaluate_policy(art: RunArtifacts, policy, budget_bytes_per_s: float,
                    with_collector: bool = True) -> dict:
    """Run one policy end to end and score it."""
    dp, tcfg = art.dp, art.tcfg
    t0 = time.time()
    res = policy.apply(dp, tcfg, art.scenario.duration_us)
    t_mark = time.time() - t0
    cap = capture_rate(dp, res.marking.marked)
    out = {
        "policy": res.name,
        "volume_bytes_per_s": res.volume["bytes_per_s"],
        "volume_mbps": res.volume["bytes_per_s"] * 8 / 1e6,
        "sampled_packets": res.volume["packets"],
        "capture": cap,
        "params": res.params,
        "boost_packets": int(res.marking.boost_mask.sum()),
        "triggers": int(res.marking.trigger_t.size),
        "mark_seconds": round(t_mark, 2),
    }
    if not with_collector:
        return out
    wire = materialise_wire(dp, tcfg, res.marking.marked, art.clocks, art.scenario.seed)
    col = Collector(art.topo)
    t1 = time.time()
    col.ingest(wire)
    out["ingest_seconds"] = round(time.time() - t1, 2)
    out.update(score_collector(art, col, wire))
    return out


def score_collector(art: RunArtifacts, col: Collector,
                    wire: WireTraces) -> dict:
    """Attribution accuracy and per-hop delay error against ground truth."""
    dp = art.dp
    tr = art.traffic
    # ---- attribution ---------------------------------------------------
    # Two cohorts matter, and they are not the same question:
    #  * every slow request  -> can we explain *any* slow request?
    #  * network-caused slow  -> can we nail the hop the fabric team owns?
    # A request that stalled in the server's own processing has no honest
    # fabric culprit, so mixing the two understates the fabric diagnosis.
    n = n_hit = 0
    n_big = n_big_hit = 0
    n_net = n_net_hit = 0
    delay_err: List[float] = []
    est_by_hop: Dict[tuple, List[float]] = {}
    true_by_hop: Dict[tuple, List[float]] = {}
    for tx in tr.txns:
        if np.isnan(tx.true_latency_us) or tx.true_latency_us < SLOW_TXN_US:
            continue
        a = col.attribute(tx.txn_id)
        if "error" in a or a["dominant_hop"] is None:
            continue
        pred = a["dominant_hop"]["hop"]
        hit = pred == tx.dominant_hop
        n += 1
        n_hit += hit
        if tx.dominant_queue_us >= 1_000.0:
            n_big += 1
            n_big_hit += hit
        if tx.net_latency_us >= SLOW_TXN_US:
            n_net += 1
            n_net_hit += hit
    # ---- per-hop delay error -------------------------------------------
    for t in col.traces:
        pkt = wire.b_pkt[t.packet_id]
        for h in t.hops:
            if not h.observed or not (h.depth_queue_us == h.depth_queue_us):
                continue
            true_q = float(dp.qdelay[pkt, h.hop])
            delay_err.append(float(h.depth_queue_us) - true_q)
            key = (h.sw_id, h.out_port)
            est_by_hop.setdefault(key, []).append(float(h.depth_queue_us))
            true_by_hop.setdefault(key, []).append(true_q)
    delay_err = np.asarray(delay_err) if delay_err else np.zeros(0)
    per_hop = []
    for key in sorted(est_by_hop, key=lambda k: -len(est_by_hop[k]))[:12]:
        e = np.asarray(est_by_hop[key])
        t = np.asarray(true_by_hop[key])
        per_hop.append({
            "switch": key[0], "out_port": key[1], "records": int(e.size),
            "true_mean_us": round(float(t.mean()), 2),
            "est_mean_us": round(float(e.mean()), 2),
            "true_p99_us": round(float(np.percentile(t, 99)), 2),
            "est_p99_us": round(float(np.percentile(e, 99)), 2),
        })
    stats = col.stats()
    clock_err = None
    if col.clocks is not None and art.clocks is not None:
        clock_err = float(np.median(np.abs(col.clocks.offset_us
                                           - art.clocks.offset_us)))
    return {
        "attribution_txns": n,
        "attribution_accuracy": (n_hit / n) if n else float("nan"),
        "attribution_txns_big": n_big,
        "attribution_accuracy_big": (n_big_hit / n_big) if n_big else float("nan"),
        "attribution_txns_net": n_net,
        "attribution_accuracy_net": (n_net_hit / n_net) if n_net else float("nan"),
        "delay_mae_us": float(np.mean(np.abs(delay_err))) if delay_err.size else float("nan"),
        "delay_p95_abs_us": float(np.percentile(np.abs(delay_err), 95)) if delay_err.size else float("nan"),
        "delay_samples": int(delay_err.size),
        "per_hop": per_hop,
        "collector": stats,
        "clock_median_abs_error_us": clock_err,
    }


# ---------------------------------------------------------------------------
def sweep_fixed(art: RunArtifacts, target_bytes_per_s: float,
                iters: int = 22) -> FixedSampling:
    """Binary-search the fixed probability that matches a target volume."""
    lo, hi = 1e-7, 0.9
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p = FixedSampling(mid)
        r = p.apply(art.dp, art.tcfg, art.scenario.duration_us)
        if r.volume["bytes_per_s"] < target_bytes_per_s:
            lo = mid
        else:
            hi = mid
    return FixedSampling(0.5 * (lo + hi))


def run_acceptance(sc: Optional[Scenario] = None,
                   budgets_bytes_per_s: Sequence[float] = (80_000.0, 250_000.0, 800_000.0),
                   include_collector: bool = True,
                   seed: Optional[int] = None) -> dict:
    sc = sc or standard_scenario()
    if seed is not None:
        sc.seed = seed
    art = build_run(sc)
    strata = calibration_strata(art.dp, sc.duration_us)
    trig_bytes = estimate_trigger_volume(art.dp, art.tcfg)
    report: dict = {
        "scenario": {
            "name": sc.name,
            "duration_s": sc.duration_us / 1e6,
            "packets": int(art.traffic.n_packets),
            "flows": len(art.traffic.flows),
            "transactions": len(art.traffic.txns),
            "slow_transactions": sum(
                1 for tx in art.traffic.txns
                if not np.isnan(tx.true_latency_us)
                and tx.true_latency_us >= SLOW_TXN_US),
            "faults": {
                "congestion_episodes": len(sc.scenario.congestion),
                "impaired_links": len(sc.scenario.link_impair),
                "clock_biases": sc.scenario.clock_bias_ppb,
            },
            "int_budget_bytes": art.tcfg.budget_bytes,
            "trigger_bytes_per_s": trig_bytes,
            "max_records": (art.tcfg.budget_bytes - 24) // art.tcfg.hop_bytes,
        },
        "budgets": [],
    }
    for budget in budgets_bytes_per_s:
        opt = optimize_budget(strata, art.tcfg, budget_bytes_per_s=budget,
                              trigger_bytes_per_s=trig_bytes,
                              strata_pkt_rate=total_packet_rate(
                                  art.dp, sc.duration_us))
        adaptive = AdaptiveSampling(opt["base_prob"], opt["stratum_prob"])
        a_res = evaluate_policy(art, adaptive, budget, include_collector)
        fixed = sweep_fixed(art, a_res["volume_bytes_per_s"])
        f_res = evaluate_policy(art, fixed, budget, include_collector)
        report["budgets"].append({
            "budget_bytes_per_s": budget,
            "adaptive": a_res,
            "fixed": f_res,
            "optimizer": {k: v for k, v in opt.items() if k != "stratum_prob"},
            "n_rules": opt.get("n_rules", 0),
        })
    return report


def run_control(seed: int = 7) -> dict:
    """Same topology and traffic, no injected faults.

    This is the honesty check on the delay estimator: with no congestion the
    true queueing is near zero, so any reported delay is estimator error.
    """
    sc = standard_scenario()
    sc.seed = seed
    sc.scenario = ScenarioConfig()      # no shapers, no congestion, no bad links
    sc.episodes = []
    art = build_run(sc)
    tcfg = TelemetryConfig()
    strata = calibration_strata(art.dp, sc.duration_us)
    trig = estimate_trigger_volume(art.dp, tcfg)
    opt = optimize_budget(strata, tcfg, 1_000_000.0, trigger_bytes_per_s=trig,
                          strata_pkt_rate=total_packet_rate(art.dp, sc.duration_us))
    ad = AdaptiveSampling(opt["base_prob"], opt["stratum_prob"])
    res = evaluate_policy(art, ad, 1_000_000.0, with_collector=True)
    return {
        "seed": seed,
        "packets": int(art.traffic.n_packets),
        "delay_mae_us": res.get("delay_mae_us"),
        "delay_p95_abs_us": res.get("delay_p95_abs_us"),
        "true_hop_queue_p99_us": float(np.percentile(
            art.dp.qdelay[np.arange(art.dp.max_h)[None, :] < art.dp.n_hops[:, None]],
            99)),
        "volume_bytes_per_s": res["volume_bytes_per_s"],
    }


def run_scenarios(seeds: Sequence[int] = (7, 11, 23),
                  budgets_bytes_per_s: Sequence[float] = (250_000.0,)) -> dict:
    """Repeat the acceptance across seeds to show it is not a lucky draw."""
    rows = []
    for seed in seeds:
        sc = standard_scenario()
        sc.seed = seed
        art = build_run(sc)
        strata = calibration_strata(art.dp, sc.duration_us)
        for budget in budgets_bytes_per_s:
            trig = estimate_trigger_volume(art.dp, art.tcfg)
            opt = optimize_budget(strata, art.tcfg, budget_bytes_per_s=budget,
                                  trigger_bytes_per_s=trig,
                                  strata_pkt_rate=total_packet_rate(
                                      art.dp, sc.duration_us))
            ad = AdaptiveSampling(opt["base_prob"], opt["stratum_prob"])
            a = evaluate_policy(art, ad, budget, True)
            fx = sweep_fixed(art, a["volume_bytes_per_s"])
            f = evaluate_policy(art, fx, budget, True)
            rows.append({"seed": seed, "budget": budget,
                         "adaptive": {k: a[k] for k in (
                             "volume_bytes_per_s", "capture",
                             "attribution_accuracy",
                             "attribution_accuracy_big",
                             "attribution_accuracy_net", "delay_mae_us")},
                         "fixed": {k: f[k] for k in (
                             "volume_bytes_per_s", "capture",
                             "attribution_accuracy",
                             "attribution_accuracy_big",
                             "attribution_accuracy_net", "delay_mae_us")}})
    return {"rows": rows}
