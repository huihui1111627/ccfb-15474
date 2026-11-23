"""Tests for the INT testbed.

Run with:  python3 -m pytest tests -q
"""
from __future__ import annotations

import numpy as np
import pytest

from itel.collector import (Collector, estimate_clocks, known_path,
                            _records_by_packet)
from itel.config import (DEFAULT_INT_BUDGET_BYTES, SLOW_TXN_US, ScenarioConfig,
                         TelemetryConfig, TrafficConfig)
from itel.evaluate import build_run, standard_scenario
from itel.sampling import (AdaptiveSampling, FixedSampling, capture_rate,
                           calibration_strata, estimate_trigger_volume,
                           metadata_bytes, optimize_budget, total_packet_rate)
from itel.sim import (INT_BASE_BYTES, QFLAG_TRUNC, annotate_transactions,
                      build_impairment, make_clocks, materialise_wire,
                      mark_packets, run_data_plane)
from itel.topology import MAX_PORTS, build_fat_tree
from itel.traffic import build_traffic, default_classes


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def topo():
    return build_fat_tree()


@pytest.fixture(scope="module")
def std_art(topo):
    """The reference incident, built once and shared by the expensive tests."""
    art = build_run(standard_scenario())
    return art.traffic, art.dp


@pytest.fixture(scope="module")
def small_run(topo):
    """A short run: tests do not need the full 2 s incident."""
    traffic = build_traffic(
        TrafficConfig(classes=default_classes(topo), duration_us=60_000.0,
                      flows_per_second=1_500.0, seed=3),
        topo, [{"t_start": 10_000, "t_end": 50_000, "server_idx": 0}])
    clocks = make_clocks(topo, {}, 3)
    imp = build_impairment(topo, ScenarioConfig())
    dp = run_data_plane(traffic, topo, imp, clocks, 3)
    annotate_transactions(dp)
    return traffic, dp, clocks


# ---------------------------------------------------------------------------
# topology / routing
# ---------------------------------------------------------------------------
def test_fat_tree_shape(topo):
    kinds = [s.switch_id.split("-")[0] for s in topo.switches]
    assert kinds.count("leaf") == 8
    assert kinds.count("agg") == 8
    assert kinds.count("spine") == 4
    assert kinds.count("super") == 2
    assert len(topo.hosts) == 64


def test_path_lengths(topo):
    """One/two/three tier paths are 1, 3 and 7 hops."""
    same_leaf = topo.path_for("host-1-0-0", "host-1-0-3", 12345)
    same_pod = topo.path_for("host-1-0-0", "host-1-1-3", 12345)
    cross = topo.path_for("host-1-0-0", "host-3-1-3", 12345)
    assert same_leaf.n_hops == 1
    assert same_pod.n_hops == 3
    assert cross.n_hops == 7


def test_ecmp_diversity(topo):
    """Different flow hashes must reach different fabric paths."""
    sigs = {topo.path_for("host-0-0-1", "host-3-1-2", k).signature
            for k in range(60)}
    assert len(sigs) >= 4, "ECMP should spread a leaf pair over many paths"


def test_path_is_connected(topo):
    """Every consecutive hop must hand off through a real link."""
    p = topo.path_for("host-0-0-1", "host-3-1-2", 999)
    for a, b in zip(p.hops, p.hops[1:]):
        link = topo.links[a.link]
        assert link.src == topo.switches[a.sw].switch_id
        assert link.dst == topo.switches[b.sw].switch_id


def test_path_for_is_stable(topo):
    a = topo.path_for("host-0-0-1", "host-2-1-4", 4242)
    b = topo.path_for("host-0-0-1", "host-2-1-4", 4242)
    assert a is b


# ---------------------------------------------------------------------------
# data plane
# ---------------------------------------------------------------------------
def test_queueing_is_fifo_and_monotone(topo, small_run):
    """Arrival order at a port must be service order: delays never invert."""
    traffic, dp, _ = small_run
    # pick the busiest port and check its packets' departure order
    h = 6
    m = dp.n_hops > h
    ports = dp.hop_port[m, h]
    pk = np.bincount(ports).argmax()
    rows = np.nonzero(m)[0][ports == pk]
    order = np.argsort(dp.t_arrive[rows, h])
    dep = dp.t_depart[rows[order], h]
    assert np.all(np.diff(dep) >= -1e-9), "departures must be non-decreasing"


def test_serialisation_matches_link_rate(topo, small_run):
    traffic, dp, _ = small_run
    h = 2
    m = dp.n_hops > h
    sel = np.nonzero(m)[0][:500]
    for i in sel:
        rate = dp.rate_eff[i, h]
        assert rate > 0
        assert dp.ser_us[i, h] == pytest.approx(
            float(traffic.size[i]) / rate, rel=1e-6)


def test_queue_depth_is_backlog_bytes(topo, small_run):
    """qdepth = queueing delay x drain rate, capped by the port buffer."""
    traffic, dp, _ = small_run
    h = 6
    m = dp.n_hops > h
    sel = np.nonzero(m)[0][:500]
    for i in sel:
        cap = topo.port_cap_bytes[dp.hop_port[i, h]]
        expect = min(dp.qdelay[i, h] * dp.rate_eff[i, h], cap)
        assert dp.qdepth_bytes[i, h] == pytest.approx(expect, rel=1e-6)


def test_slowdown_raises_rate_eff(topo):
    """A shaped port must show a lower effective drain rate, not more bytes."""
    traffic = build_traffic(
        TrafficConfig(classes=default_classes(topo), duration_us=80_000.0,
                      flows_per_second=2_500.0, seed=5),
        topo, [{"t_start": 5_000, "t_end": 75_000, "server_idx": 0}])
    clocks = make_clocks(topo, {}, 5)
    sc = ScenarioConfig(shapers=[("leaf-0-0", 3, 0.4)])
    imp = build_impairment(topo, sc)
    dp = run_data_plane(traffic, topo, imp, clocks, 5)
    pk = topo.ports[(topo.sw_index("leaf-0-0"), 3)]
    rows = np.nonzero(np.any(dp.hop_port == pk, axis=1))[0]
    assert rows.size > 0
    rates = []
    for i in rows:
        cols = np.nonzero(dp.hop_port[i] == pk)[0]
        rates.append(dp.rate_eff[i, cols[0]])
    assert max(rates) == pytest.approx(125.0 * 0.4, rel=1e-6)


def test_retransmit_flag_and_penalty(topo):
    traffic = build_traffic(
        TrafficConfig(classes=default_classes(topo), duration_us=80_000.0,
                      flows_per_second=1_500.0, seed=9),
        topo, [])
    clocks = make_clocks(topo, {}, 9)
    sc = ScenarioConfig(link_impair=[("spine-0", "agg-0-0", 0, 80_000, 0.5)])
    imp = build_impairment(topo, sc)
    dp = run_data_plane(traffic, topo, imp, clocks, 9)
    assert dp.retx.sum() > 0, "a 50% loss link must produce retransmissions"
    # a retransmitted hop's service time must exceed the bare serialisation
    bad = np.nonzero(dp.retx.any(axis=1))[0]
    i = bad[0]
    h = int(np.nonzero(dp.retx[i])[0][0])
    assert dp.ser_us[i, h] > float(traffic.size[i]) / dp.rate_eff[i, h]


# ---------------------------------------------------------------------------
# in-band metadata
# ---------------------------------------------------------------------------
def test_metadata_cap_is_respected(topo):
    tcfg = TelemetryConfig()
    max_recs = (tcfg.budget_bytes - INT_BASE_BYTES) // tcfg.hop_bytes
    traffic = build_traffic(
        TrafficConfig(classes=default_classes(topo), duration_us=60_000.0,
                      flows_per_second=1_500.0, seed=11),
        topo, [{"t_start": 5_000, "t_end": 55_000, "server_idx": 0}])
    clocks = make_clocks(topo, {}, 11)
    dp = run_data_plane(traffic, topo, build_impairment(topo, ScenarioConfig()),
                        clocks, 11)
    annotate_transactions(dp)
    marked = np.zeros(traffic.n_packets, dtype=bool)
    marked[::7] = True
    wire = materialise_wire(dp, tcfg, marked, clocks, 11)
    per_pkt = np.bincount(wire.pkt, minlength=traffic.n_packets)[wire.b_pkt]
    assert per_pkt.max() <= max_recs
    assert wire.wire_bytes() <= marked.sum() * tcfg.budget_bytes


def test_truncation_keeps_last_hop_and_flags(topo):
    tcfg = TelemetryConfig(truncation_policy="key_hops")
    traffic = build_traffic(
        TrafficConfig(classes=default_classes(topo), duration_us=40_000.0,
                      flows_per_second=1_500.0, seed=13),
        topo, [{"t_start": 5_000, "t_end": 35_000, "server_idx": 0}])
    clocks = make_clocks(topo, {}, 13)
    dp = run_data_plane(traffic, topo, build_impairment(topo, ScenarioConfig()),
                        clocks, 13)
    long_pkt = int(np.nonzero(dp.n_hops == 7)[0][0])
    marked = np.zeros(traffic.n_packets, dtype=bool)
    marked[long_pkt] = True
    wire = materialise_wire(dp, tcfg, marked, clocks, 13)
    hops = sorted(wire.hop[wire.pkt == long_pkt].tolist())
    assert hops[-1] == 6, "the final hop proves delivery and must survive"
    assert len(hops) < 7
    assert int((wire.flags & QFLAG_TRUNC).max()) == QFLAG_TRUNC
    assert wire.b_dropped[0] == 7 - len(hops)


def test_truncation_policies_differ(topo):
    traffic = build_traffic(
        TrafficConfig(classes=default_classes(topo), duration_us=40_000.0,
                      flows_per_second=1_500.0, seed=17),
        topo, [])
    clocks = make_clocks(topo, {}, 17)
    dp = run_data_plane(traffic, topo, build_impairment(topo, ScenarioConfig()),
                        clocks, 17)
    marked = np.zeros(traffic.n_packets, dtype=bool)
    marked[np.nonzero(dp.n_hops == 7)[0][:40]] = True
    first = materialise_wire(dp, TelemetryConfig(truncation_policy="first_k"),
                             marked, clocks, 17)
    key = materialise_wire(dp, TelemetryConfig(truncation_policy="key_hops"),
                           marked, clocks, 17)
    assert set(first.hop.tolist()) != set(key.hop.tolist())


def test_metadata_bytes_shrinks_with_budget(topo):
    hops = np.array([1, 3, 7] * 10)
    big = metadata_bytes(hops, TelemetryConfig(budget_bytes=200))
    small = metadata_bytes(hops, TelemetryConfig(budget_bytes=60))
    assert np.all(small <= big)
    assert small[2] == 24 + ((60 - 24) // 12) * 12


# ---------------------------------------------------------------------------
# collector
# ---------------------------------------------------------------------------
def test_collector_recovers_path_through_holes(topo, small_run):
    traffic, dp, clocks = small_run
    marked = np.zeros(traffic.n_packets, dtype=bool)
    marked[::5] = True
    wire = materialise_wire(dp, TelemetryConfig(), marked, clocks, 3)
    col = Collector(topo)
    col.ingest(wire)
    assert col.traces
    for t in col.traces[:200]:
        assert len(t.hops) in (1, 3, 7), "path shape must be reconstructed"
        if t.n_missing:
            assert any(not h.observed for h in t.hops)


def test_clock_estimate_is_accurate(topo, small_run):
    traffic, dp, clocks = small_run
    marked = np.zeros(traffic.n_packets, dtype=bool)
    marked[::2] = True
    wire = materialise_wire(dp, TelemetryConfig(), marked, clocks, 3)
    _, recs = _records_by_packet(wire)
    est = estimate_clocks(recs, wire, topo)
    # Absolute offsets are only identifiable up to a common gauge (there is no
    # synchronised reference in a real fabric).  What matters for per-hop
    # delay is the *difference* between two switches' offsets.
    true_diff = clocks.offset_us[:, None] - clocks.offset_us[None, :]
    est_diff = est.offset_us[:, None] - est.offset_us[None, :]
    err = np.abs(est_diff - true_diff)
    assert np.median(err) < 2.0, f"median offset-difference error {np.median(err):.2f} us"
    # the 60 ms fixture gives every switch pair only a few hundred samples,
    # so a switch that appears on few paths stays coarser; that is the
    # built-in cost of sparse telemetry, and the collector reports it
    assert np.percentile(err, 90) < 12.0, f"p90 {np.percentile(err, 90):.2f} us"


def test_hop_delay_estimate_matches_truth(topo, small_run):
    traffic, dp, clocks = small_run
    marked = np.zeros(traffic.n_packets, dtype=bool)
    marked[::4] = True
    wire = materialise_wire(dp, TelemetryConfig(), marked, clocks, 3)
    col = Collector(topo)
    col.ingest(wire)
    errs = []
    for t in col.traces:
        pkt = wire.b_pkt[t.packet_id]
        for h in t.hops:
            if h.observed and h.queue_best == h.queue_best:
                errs.append(h.queue_best - float(dp.qdelay[pkt, h.hop]))
    errs = np.abs(np.asarray(errs))
    assert errs.size > 1000
    assert np.median(errs) < 3.0
    assert np.percentile(errs, 95) < 25.0


def test_attribution_finds_the_congested_hop(topo):
    """With a hard, sustained incast, attribution must name the victim port."""
    traffic = build_traffic(
        TrafficConfig(classes=default_classes(topo), duration_us=200_000.0,
                      flows_per_second=2_500.0, seed=21),
        topo, [{"t_start": 20_000, "t_end": 180_000, "server_idx": 0,
                "period_us": 30_000, "burst_us": 1_500, "n_senders": 18,
                "sender_rate_bpus": 120.0}])
    clocks = make_clocks(topo, {}, 21)
    dp = run_data_plane(traffic, topo, build_impairment(topo, ScenarioConfig()),
                        clocks, 21)
    annotate_transactions(dp)
    marked = np.ones(traffic.n_packets, dtype=bool)   # full capture: pure logic test
    wire = materialise_wire(dp, TelemetryConfig(budget_bytes=400), marked, clocks, 21)
    col = Collector(topo)
    col.ingest(wire)
    slow = [tx for tx in traffic.txns
            if not np.isnan(tx.net_latency_us) and tx.net_latency_us >= SLOW_TXN_US]
    assert slow, "the injected incast must produce network-slow requests"
    hit = 0
    for tx in slow:
        a = col.attribute(tx.txn_id)
        if "error" not in a and a["dominant_hop"]:
            hit += a["dominant_hop"]["hop"] == tx.dominant_hop
    assert hit / len(slow) >= 0.9, f"{hit}/{len(slow)} attributed correctly"


def test_attribution_names_co_flows(topo):
    traffic = build_traffic(
        TrafficConfig(classes=default_classes(topo), duration_us=200_000.0,
                      flows_per_second=2_500.0, seed=23),
        topo, [{"t_start": 20_000, "t_end": 180_000, "server_idx": 0,
                "period_us": 30_000, "burst_us": 1_500, "n_senders": 18,
                "sender_rate_bpus": 120.0}])
    clocks = make_clocks(topo, {}, 23)
    dp = run_data_plane(traffic, topo, build_impairment(topo, ScenarioConfig()),
                        clocks, 23)
    annotate_transactions(dp)
    marked = np.ones(traffic.n_packets, dtype=bool)
    wire = materialise_wire(dp, TelemetryConfig(budget_bytes=400), marked, clocks, 23)
    col = Collector(topo)
    col.ingest(wire)
    # the incast victim is the host whose *egress* port is congested; slow
    # responses delivered to it were queued behind the incast senders
    victim = traffic.server_pool[0]
    slow = [tx for tx in traffic.txns
            if tx.client == victim and not np.isnan(tx.net_latency_us)
            and tx.net_latency_us >= SLOW_TXN_US]
    assert slow, "the incast must stall responses delivered to the victim"
    found = []
    for tx in slow:
        a = col.attribute(tx.txn_id)
        for cf in a.get("co_flows", []):
            if cf["dst"] == victim and cf["src"] != victim:
                found.append(cf)
    assert found, "co-flow list must name the incast senders"
    assert all(cf["cls"] for cf in found)


# ---------------------------------------------------------------------------
# sampling / budget
# ---------------------------------------------------------------------------
def test_fixed_sampling_rate_matches_probability(topo, small_run):
    traffic, dp, _ = small_run
    r = FixedSampling(0.05).apply(dp, TelemetryConfig(), 60_000.0)
    assert abs(r.marking.marked.mean() - 0.05) < 0.01


def test_trigger_marks_more_than_base(topo, small_run):
    traffic, dp, _ = small_run
    n = traffic.n_packets
    m = mark_packets(dp, TelemetryConfig(), np.zeros(n),
                     np.zeros(n, dtype=np.int8), use_trigger=True)
    assert m.boost_mask.any(), "a hot queue must arm at least one switch"
    # every boosted packet must be marked
    assert np.all(m.marked[m.boost_mask])


def test_optimizer_respects_budget(topo, small_run):
    traffic, dp, _ = small_run
    strata = calibration_strata(dp, 60_000.0)
    tcfg = TelemetryConfig()
    trig = estimate_trigger_volume(dp, tcfg)
    for budget in (trig * 1.5, trig + 3e5, trig + 3e6):
        opt = optimize_budget(strata, tcfg, budget,
                              trigger_bytes_per_s=trig,
                              strata_pkt_rate=total_packet_rate(dp, 60_000.0))
        ad = AdaptiveSampling(opt["base_prob"], opt["stratum_prob"])
        r = ad.apply(dp, tcfg, 60_000.0)
        assert r.volume["bytes_per_s"] <= budget * 1.10, (
            f"budget {budget:.0f} exceeded: {r.volume['bytes_per_s']:.0f}")


def test_adaptive_beats_fixed_at_equal_volume(topo, std_art):
    """The headline claim: same bytes, more captured slow requests.

    Uses the reference incident so the comparison is the one the acceptance
    report makes, not a toy.
    """
    traffic, dp = std_art
    tcfg = TelemetryConfig()
    strata = calibration_strata(dp, 2_000_000.0)
    trig = estimate_trigger_volume(dp, tcfg)
    opt = optimize_budget(strata, tcfg, 2_000_000.0, trigger_bytes_per_s=trig,
                          strata_pkt_rate=total_packet_rate(dp, 2_000_000.0))
    ad = AdaptiveSampling(opt["base_prob"], opt["stratum_prob"])
    ra = ad.apply(dp, tcfg, 2_000_000.0)
    cap_a = capture_rate(dp, ra.marking.marked)["capture_rate"]

    # match the fixed policy's volume to the adaptive one
    lo, hi = 1e-7, 0.9
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        v = FixedSampling(mid).apply(dp, tcfg, 2_000_000.0).volume["bytes_per_s"]
        if v < ra.volume["bytes_per_s"]:
            lo = mid
        else:
            hi = mid
    rf = FixedSampling(0.5 * (lo + hi)).apply(dp, tcfg, 2_000_000.0)
    cap_f = capture_rate(dp, rf.marking.marked)["capture_rate"]
    assert abs(rf.volume["bytes_per_s"] / ra.volume["bytes_per_s"] - 1) < 0.02
    assert cap_a >= cap_f + 0.30, f"adaptive {cap_a:.2f} vs fixed {cap_f:.2f}"


def test_sampling_is_deterministic(topo, small_run):
    traffic, dp, _ = small_run
    r1 = FixedSampling(0.1).apply(dp, TelemetryConfig(), 60_000.0)
    r2 = FixedSampling(0.1).apply(dp, TelemetryConfig(), 60_000.0)
    assert np.array_equal(r1.marking.marked, r2.marking.marked)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
def test_report_shape():
    rep = _run_short_acceptance()
    assert rep["scenario"]["packets"] > 10_000
    assert rep["budgets"], "the report must contain at least one budget point"
    row = rep["budgets"][0]
    for side in ("adaptive", "fixed"):
        assert 0.0 <= row[side]["capture"]["capture_rate"] <= 1.0
        assert row[side]["volume_bytes_per_s"] > 0


def _run_short_acceptance():
    from itel.evaluate import run_acceptance
    sc = standard_scenario()
    sc.duration_us = 300_000.0
    sc.episodes = [{"t_start": 20_000, "t_end": 280_000, "server_idx": 0,
                    "period_us": 25_000, "burst_us": 1_500, "n_senders": 16,
                    "sender_rate_bpus": 120.0}]
    return run_acceptance(sc, budgets_bytes_per_s=(300_000.0,),
                          include_collector=False)
