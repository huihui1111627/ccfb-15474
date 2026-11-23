"""The telemetry collector.

Input is exactly what arrived on the wire (:class:`~itel.sim.WireTraces`):
switch-clock nanosecond timestamps, tiny queue-depth counters, and no ground
truth.  The collector does five things:

1. **Path reconstruction** -- order each packet's hop records and fill the
   holes left by truncation.  The collector holds the fabric's topology and
   the ECMP hash seed, so the path for a 5-tuple is *determined*, not guessed;
   a dropped record is a hole in the observation, not in the path.
2. **Clock skew removal** -- each switch has an independent oscillator with a
   fixed offset and a linear drift.  A global least-squares fit recovers
   ``offset_sw`` and ``rate_sw`` using per-packet consecutive-hop timestamp
   differences, which are immune to absolute (unobservable) time.  Queueing
   only ever *adds* delay, so the fit is re-weighted on the low-delay decile
   to keep congestion out of the clock estimate.
3. **Per-hop residence decomposition** -- residence minus nominal propagation
   minus serialisation minus switching gives the queueing component, which is
   what an operator cares about.
4. **Flow aggregation** -- records grouped by 5-tuple; transactions closed on
   the response leg; rolling percentiles per flow for anomaly scoring.
5. **Attribution** -- for a slow transaction, name the hop that carried the
   queueing and list the other flows queued at that same egress port in the
   same window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import TelemetryConfig
from .sim import (INT_BASE_BYTES, QFLAG_DEPTH_SAT, QFLAG_KEY, QFLAG_RETX,
                  QFLAG_TRUNC, WireTraces)
from .topology import MAX_PORTS, Topology, canonical_flow_key, tuple_hash5

NS_PER_US = 1_000.0
#: switching cost assumed by the collector (it cannot see ASIC internals)
NOMINAL_SWITCH_US = 1.5
#: a packet arrival at the collector is a "slow request" beyond this
SLOW_THRESHOLD_US = 5_000.0
#: half-width of the co-flow attribution window.  A queue that took `d` us to
#: build was fed by traffic arriving over that whole interval, so the window
#: scales with the observed delay rather than being a fixed constant.
COFLOW_MIN_WINDOW_US = 200.0
COFLOW_MAX_WINDOW_US = 60_000.0
COFLOW_WINDOW_FACTOR = 3.0
#: minimum samples before a switch pair is trusted for a clock estimate
PAIR_MIN_SAMPLES = 400


# ---------------------------------------------------------------------------
@dataclass
class HopEstimate:
    hop: int
    sw: int
    sw_id: str
    in_port: int
    out_port: int
    port_key: int
    observed: bool
    residence_us: float
    queue_us: float             # queueing delay from the timestamp differential
    qdepth_bytes: float
    depth_queue_us: float       # queueing delay implied by the byte counter
    retx: bool
    span_hops: int = 1          # hops covered by this residence measurement
    rate_bpus: float = float("nan")   # port rate used to convert the counter
    size: float = float("nan")        # packet size, needed by the rate fit

    @property
    def known(self) -> bool:
        return self.observed and self.queue_us == self.queue_us

    @property
    def queue_best(self) -> float:
        """Best available queueing estimate.

        The timestamp differential is a direct measurement of the residence,
        so it wins whenever two adjacent records exist.  When truncation left
        a hole, the byte counter is the only signal that still attributes the
        delay to this hop, and it is used with the rate the collector learned
        for this port at this time.
        """
        if self.queue_us == self.queue_us:
            return float(self.queue_us)
        return float(self.depth_queue_us)


@dataclass
class PacketTrace:
    packet_id: int
    flow_key: int
    five_tuple: Tuple[str, str, int, int, int]
    src: str
    dst: str
    cls: str
    seq: int
    txn_id: int
    role: int
    send_us: float
    hops: List[HopEstimate]
    n_missing: int
    measured_us: float
    path_sig: str

    @property
    def complete(self) -> bool:
        return self.n_missing == 0

    @property
    def total_queue_us(self) -> float:
        return float(sum(h.queue_best for h in self.hops if h.observed
                         and h.queue_best == h.queue_best))

    def dominant(self) -> Optional[HopEstimate]:
        known = [h for h in self.hops
                 if h.observed and h.queue_best == h.queue_best]
        if not known:
            return None
        return max(known, key=lambda h: h.queue_best)

    def path_switch_ids(self) -> Tuple[str, ...]:
        return tuple(h.sw_id for h in self.hops)


@dataclass
class ClockEstimate:
    offset_us: np.ndarray
    rate: np.ndarray
    resid_ns: np.ndarray
    n_samples: np.ndarray

    def correct_us(self, sw: int, ts_ns) -> np.ndarray:
        return np.asarray(ts_ns) / NS_PER_US / (1.0 + self.rate[sw]) - self.offset_us[sw]

    def offset_spread_us(self, sw_a: int, sw_b: int) -> float:
        """Offset difference between two switches: the *only* identifiable part.

        Absolute offsets need a synchronised reference; a collector has none.
        Every per-hop delay uses a strain of (offset_b - offset_a), so this is
        the quantity that matters.
        """
        return float(self.offset_us[sw_a] - self.offset_us[sw_b])


# ---------------------------------------------------------------------------
# wire -> per-packet observed records
# ---------------------------------------------------------------------------
def _records_by_packet(wire: WireTraces) -> Tuple[np.ndarray, List[List[dict]]]:
    order = np.lexsort((wire.hop, wire.pkt))
    pkt = wire.pkt[order]
    starts = np.nonzero(np.r_[True, pkt[1:] != pkt[:-1]])[0] if pkt.size else np.empty(0, int)
    ends = np.r_[starts[1:], pkt.size] if pkt.size else np.empty(0, int)
    out: List[List[dict]] = []
    for b in range(wire.n_packets):
        rows = order[starts[b]:ends[b]] if b < starts.size else np.empty(0, int)
        recs = [dict(hop=int(wire.hop[r]), sw=int(wire.sw[r]),
                     in_port=int(wire.in_port[r]), out_port=int(wire.out_port[r]),
                     ts=int(wire.ts_ns[r]), depth=float(wire.qdepth_bytes[r]),
                     flags=int(wire.flags[r]), size=float(wire.b_size[b]))
                for r in rows]
        recs.sort(key=lambda d: d["hop"])
        out.append(recs)
    return order, out


# ---------------------------------------------------------------------------
# 2. clock skew removal
# ---------------------------------------------------------------------------
@dataclass
class _Edge:
    sw_a: int
    sw_b: int
    resid_ns: float
    mid_us: float
    weight: float = 1.0


def estimate_clocks(recs_by_pkt: List[List[dict]], wire: WireTraces,
                    topo: Topology, n_iter: int = 4) -> ClockEstimate:
    """Estimate every switch's clock offset against a reference switch.

    Model: ``ts_sw(t) = t + offset_sw`` (see the note on drift below).

    The collector observes, for each marked packet, the departure timestamps
    of its retained hop records.  For two *adjacent-on-path* records a -> b
    the true spacing is known from the topology (propagation + serialisation
    + switching) to within queueing, so

        ts_b - ts_a = gap + (offset_b - offset_a) + queueing

    Queueing is non-negative, so the low tail of ``(ts_b - ts_a) - gap`` over
    many packets converges on the clock difference.  Building one such
    estimate per switch pair and solving all pairs at once least-squares
    gives the offsets up to a common gauge, which the reference switch pins.

    Records separated by a truncated hop are *not* used: their "gap" hides the
    queueing at every skipped switch, so their low tail is not the clock
    difference.  For a 7-hop path with a 4-record budget that rules out most
    pairs, which is exactly why the budget matters to a collector.
    """
    n_sw = topo.n_switches
    ea_l, eb_l, er_l = [], [], []
    for b_i, recs in enumerate(recs_by_pkt):
        path = known_path(topo, wire.b_src[b_i], wire.b_dst[b_i],
                          wire.b_five_tuple[b_i])
        for a, b in zip(recs, recs[1:]):
            if b["hop"] != a["hop"] + 1:
                continue
            gap = _gap_along(topo, path, a, b, float(wire.b_size[b_i]))
            ea_l.append(a["sw"])
            eb_l.append(b["sw"])
            er_l.append((b["ts"] - a["ts"]) - gap * NS_PER_US)
        # the ingress stamp is taken on the first switch's own clock, so
        # (first departure - ingress stamp) is another same-clock observation
        if recs and recs[0]["hop"] == 0:
            r0 = recs[0]
            ea_l.append(r0["sw"])
            eb_l.append(r0["sw"])
            er_l.append((r0["ts"] - int(wire.b_src_ts_ns[b_i]))
                        - _ingress_gap(topo, wire, b_i, r0) * NS_PER_US)
    if not ea_l:
        return ClockEstimate(np.zeros(n_sw), np.zeros(n_sw), np.zeros(n_sw),
                             np.zeros(n_sw, dtype=np.int64))

    ea = np.asarray(ea_l, dtype=np.int64)
    eb = np.asarray(eb_l, dtype=np.int64)
    er = np.asarray(er_l, dtype=np.float64)
    synced = ea != eb

    # sigma-clipped per-pair low quantile, fitted iteratively.  Each round
    # uses the current offsets to measure how far each pair's low tail sits
    # from the global floor, and drops pairs that are chronically congested
    # (their floor cannot be trusted as a clock reference).
    unk = [sw for sw in range(n_sw) if sw != 0]
    pos = {sw: k for k, sw in enumerate(unk)}
    if synced.any():
        keys = ea[synced] * n_sw + eb[synced]
        res_s = er[synced]
        uniq, inv = np.unique(keys, return_inverse=True)
        ua = uniq // n_sw
        ub = uniq % n_sw
        order = np.argsort(inv, kind="stable")
        r_sorted = res_s[order]
        inv_sorted = inv[order]
        starts = np.searchsorted(inv_sorted, np.arange(uniq.size), side="left")
        ends = np.r_[starts[1:], inv_sorted.size]
        n_per = ends - starts
        lo = np.array([np.percentile(r_sorted[starts[j]:ends[j]], 10)
                       for j in range(uniq.size)])
        base = np.maximum(n_per, 1).astype(np.float64)
    else:
        ua = ub = np.zeros(0, dtype=np.int64)
        lo = np.zeros(0)
        base = np.zeros(0)

    ref = 0
    off_k = np.zeros(n_sw)
    for it in range(n_iter):
        # residual of each pair against the current offsets
        rr = lo - (off_k[ub] - off_k[ua]) * NS_PER_US if ua.size else np.zeros(0)
        if ua.size:
            well = n_per >= PAIR_MIN_SAMPLES
            floor = np.percentile(rr[well], 10) if well.any() else np.median(rr)
            keep = rr <= floor + 150.0
            if keep.sum() < max(4, n_sw - 1):
                keep = np.ones(ua.size, dtype=bool)
        else:
            keep = np.zeros(0, dtype=bool)
        H = np.zeros((len(unk), len(unk)))
        g = np.zeros(len(unk))
        for j in np.nonzero(keep)[0]:
            w = base[j] ** 0.5
            h = np.zeros(len(unk))
            if ua[j] != ref:
                h[pos[ua[j]]] = -1.0
            if ub[j] != ref:
                h[pos[ub[j]]] += 1.0
            nz = np.nonzero(h)[0]
            if nz.size == 0:
                continue
            H[np.ix_(nz, nz)] += w * np.outer(h[nz], h[nz])
            g[nz] += w * h[nz] * lo[j]
        if ua.size == 0 or keep.sum() < max(4, n_sw - 1):
            break
        H[np.diag_indices_from(H)] += 1e-6 * max(1.0, float(np.trace(H)))
        sol = np.linalg.lstsq(H, g, rcond=None)[0]
        for sw in unk:
            off_k[sw] = sol[pos[sw]] / NS_PER_US

    pred = (off_k[eb] - off_k[ea]) * NS_PER_US
    resid = er - pred
    n_samp = np.bincount(eb, minlength=n_sw)
    med = np.zeros(n_sw)
    for sw in range(n_sw):
        m = eb == sw
        if m.any():
            med[sw] = float(np.median(resid[m]))
    return ClockEstimate(off_k, np.zeros(n_sw), med, n_samp)


def _gap_along(topo: Topology, path, a: dict, b: dict, size: float) -> float:
    """True spacing between two *observed* hop departures, over the full path.

    Handles holes: if records a and b are k hops apart, the gap covers
    propagation out of a, then serialisation + switching at every switch
    strictly between them (at their egress rate), then b's own serialisation.
    """
    ha, hb = a["hop"], b["hop"]
    gap = NOMINAL_SWITCH_US * (hb - ha)
    gap += path.hops[ha].prop_us
    for k in range(ha + 1, hb + 1):
        rate = topo.port_rate[path.hops[k].port_key]
        if rate > 0:
            gap += size / rate
    return gap


def _ingress_gap(topo: Topology, wire: WireTraces, b: int, r0: dict) -> float:
    """Ingress stamp -> egress departure at the source switch (us)."""
    gap = NOMINAL_SWITCH_US
    if r0["out_port"] >= 0:
        rate = topo.port_rate[r0["sw"] * MAX_PORTS + r0["out_port"]]
        if rate > 0:
            gap += float(wire.b_size[b]) / rate
    return gap


def _nominal_gap(topo: Topology, a: dict, b: dict) -> float:
    """True spacing between two consecutive hop departures (us).

    departure(a) -> propagation over a's egress link -> serialisation at b at
    b's egress rate -> switching at b -> departure(b).
    """
    gap = NOMINAL_SWITCH_US
    if a["out_port"] >= 0:
        link = _port_link(topo, a["sw"], a["out_port"])
        if link is not None:
            gap += link.base_latency_us
    if b["out_port"] >= 0:
        rate = topo.port_rate[b["sw"] * MAX_PORTS + b["out_port"]]
        if rate > 0:
            gap += b.get("size", 1400.0) / rate
    return gap


def _port_link(topo: Topology, sw: int, port: int):
    pk = topo.ports.get((sw, port))
    if pk is None:
        return None
    return topo.links[topo.port_link[pk]]


# ---------------------------------------------------------------------------
# 1 + 3. path reconstruction and per-hop decomposition
# ---------------------------------------------------------------------------
_PATH_BY_SIG: Dict[tuple, object] = {}


def known_path(topo: Topology, src: str, dst: str,
               five_tuple: Tuple[str, str, int, int, int]):
    """The path a 5-tuple takes: topology + ECMP seed determine it exactly."""
    key = (src, dst, five_tuple)
    p = _PATH_BY_SIG.get(key)
    if p is None:
        p = topo.path_for(src, dst, tuple_hash5(five_tuple) & 0xFFFFFFFF)
        _PATH_BY_SIG[key] = p
    return p


def estimate_port_rates(traces: List[PacketTrace], topo: Topology,
                        bucket_us: float = 50_000.0
                        ) -> Dict[int, List[Tuple[float, float]]]:
    """Recover each egress port's *effective* drain rate, per time bucket.

    The switch reports backlog in bytes; converting that to a delay needs the
    port's real drain rate, and a rate-limited (shaped or degraded) port
    drains well below its configured line rate -- exactly the failure that
    makes a "queue is small" reading lie.  The rate is observable: for a
    packet whose previous hop is also on the wire,

        residence = propagation + size/rate + switching + queueing

    Queueing is non-negative, so the low tail of residence over many packets
    gives ``propagation + size/rate + switching``; dividing observed packet
    sizes by that tail recovers the rate.  Estimating per time bucket lets
    the collector follow a port that degrades and recovers, rather than
    averaging the two regimes into one wrong number.

    Returns ``port_key -> [(t_bucket_centre_us, rate_bpus), ...]``.
    """
    acc: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for t in traces:
        for h in t.hops:
            if not h.observed or h.span_hops != 1:
                continue
            if not (h.residence_us == h.residence_us):
                continue
            if not (h.qdepth_bytes == h.qdepth_bytes) or not (h.size == h.size):
                continue
            acc.setdefault(int(h.port_key), []).append(
                (float(t.send_us), float(h.size), float(h.residence_us),
                 float(h.qdepth_bytes)))
    out: Dict[int, List[Tuple[float, float]]] = {}
    for pk, rows in acc.items():
        if pk < 0 or pk >= topo.port_rate.size:
            continue
        nominal = float(topo.port_rate[pk])
        if nominal <= 0:
            continue
        link = topo.links[topo.port_link[pk]] if pk in topo.port_link else None
        prop = float(link.base_latency_us) if link is not None else 0.0
        ts = np.array([r[0] for r in rows])
        sizes = np.array([r[1] for r in rows])
        resid = np.array([r[2] for r in rows])
        depth = np.array([r[3] for r in rows])
        # service-only part of the residence; queueing sits on top of it
        service = resid - prop - NOMINAL_SWITCH_US
        buckets = (ts // bucket_us).astype(np.int64)
        series: List[Tuple[float, float]] = []
        last = nominal
        for bidx in np.unique(buckets):
            m = buckets == bidx
            if m.sum() < 6:
                series.append(((bidx + 0.5) * bucket_us, last))
                continue
            # A packet with backlog q waited at least q/rate, so
            #     q / (residence - prop - switching) <= rate
            # always.  The largest ratio over a window is the tightest lower
            # bound and reaches equality once the queue dominates the
            # residence, which is precisely what happens when a port degrades.
            q = depth[m]
            serv = service[m]
            ok = (q > 2_000.0) & (serv > 0.5)
            if ok.sum() >= 2:
                ratio = q[ok] / serv[ok]
                est = float(np.percentile(ratio, 75))
            else:
                # no queueing in this window: only a serialisation bound is
                # available, from the largest packets
                big = sizes[m] >= np.percentile(sizes[m], 75)
                ser = float(np.percentile(service[m][big], 5)) if big.any() else 0.0
                est = float(sizes[m][big].mean()) / ser if ser > 1e-6 else nominal
            est = float(np.clip(est, 0.02 * nominal, nominal))
            last = est
            series.append(((bidx + 0.5) * bucket_us, est))
        if series:
            out[pk] = series
    return out


def rate_at(series: List[Tuple[float, float]], t_us: float,
            nominal: float) -> float:
    """Effective rate at a time, nearest in the learned series."""
    if not series:
        return nominal
    ts = [x[0] for x in series]
    i = int(np.searchsorted(ts, t_us))
    i = min(max(i, 0), len(series) - 1)
    if i > 0 and abs(ts[i - 1] - t_us) < abs(ts[i] - t_us):
        i -= 1
    return series[i][1]


def _recompute_depth_delays(traces: List[PacketTrace],
                            rates: Dict[int, List[Tuple[float, float]]],
                            topo: Topology) -> None:
    """Re-derive every hop's byte-counter delay from the learned rate curve."""
    for t in traces:
        for h in t.hops:
            if not h.observed:
                continue
            nominal = float(h.rate_bpus)
            if not (nominal == nominal) or nominal <= 0:
                continue
            rate = rate_at(rates.get(int(h.port_key), []), t.send_us, nominal)
            h.rate_bpus = rate
            if h.qdepth_bytes == h.qdepth_bytes:
                h.depth_queue_us = float(h.qdepth_bytes) / rate


def build_traces(wire: WireTraces, topo: Topology, clocks: ClockEstimate,
                 recs_by_pkt: Optional[List[List[dict]]] = None) -> List[PacketTrace]:
    """Turn wire records into per-hop delay decompositions.

    Each retained record carries two independent delay signals:

    * the egress backlog in bytes, which converts to a queueing delay through
      the port's known rate -- no clock involved, exact to the 128 B counter
      granularity;
    * the departure timestamp, which measures residence but needs two records
      on two different switch clocks.

    The trace reports both, so the collector can cross-check a clock fit
    against a measurement that has no clock in it at all.
    """
    if recs_by_pkt is None:
        _, recs_by_pkt = _records_by_packet(wire)
    n = wire.n_packets
    # resolve every packet's path once
    paths = [None] * n
    for b in range(n):
        paths[b] = known_path(topo, wire.b_src[b], wire.b_dst[b],
                              wire.b_five_tuple[b])
    path_arr = {}
    for b, p in enumerate(paths):
        path_arr.setdefault(p.path_id, p)

    traces: List[PacketTrace] = []
    for b in range(n):
        recs = recs_by_pkt[b]
        path = paths[b]
        n_h = path.n_hops
        size = float(wire.b_size[b])
        by_hop = {r["hop"]: r for r in recs}
        hops: List[HopEstimate] = []
        prev_obs: Optional[int] = None
        for h, hp in enumerate(path.hops):
            r = by_hop.get(h)
            rate = topo.port_rate[hp.port_key]
            if r is None:
                hops.append(HopEstimate(h, hp.sw, hp.sw_id, hp.in_port,
                                        hp.out_port, hp.port_key, False,
                                        float("nan"), float("nan"),
                                        float("nan"), float("nan"), False,
                                        1, rate, size))
                continue
            depth = float(r["depth"])
            # a saturated counter is a lower bound, not a measurement: report
            # it as unusable so the timestamp estimate is used instead
            if int(r["flags"]) & QFLAG_DEPTH_SAT:
                depth_queue = float("nan")
            else:
                depth_queue = depth / rate if rate > 0 else float("nan")
            if prev_obs is None:
                residence = float("nan")
                queue = depth_queue
                span = h + 1
            else:
                t_cur = (r["ts"] / NS_PER_US / (1.0 + clocks.rate[int(r["sw"])])
                         - clocks.offset_us[int(r["sw"])])
                prev = by_hop[prev_obs]
                t_prev = (prev["ts"] / NS_PER_US / (1.0 + clocks.rate[int(prev["sw"])])
                          - clocks.offset_us[int(prev["sw"])])
                residence = t_cur - t_prev
                span = h - prev_obs
                if span > 1:
                    # the timestamp differential spans truncated hops, so it
                    # mixes in their queueing and cannot be decomposed; fall
                    # back to the byte counter for this hop
                    queue = float("nan")
                else:
                    ser = size / rate if rate > 0 else 0.0
                    queue = (residence - path.hops[prev_obs].prop_us - ser
                             - NOMINAL_SWITCH_US)
            hops.append(HopEstimate(h, hp.sw, hp.sw_id, hp.in_port, hp.out_port,
                                    hp.port_key, True, residence, queue,
                                    depth, depth_queue,
                                    bool(r["flags"] & QFLAG_RETX), span, rate,
                                    size))
            prev_obs = h
        first = next((x.hop for x in hops if x.observed), None)
        last = next((x.hop for x in reversed(hops) if x.observed), None)
        measured = 0.0
        if first is not None and last is not None and last > first:
            measured = float(hops[last].residence_us or 0.0) + sum(
                path.hops[k].prop_us + NOMINAL_SWITCH_US
                + size / topo.port_rate[path.hops[k].port_key]
                for k in range(first, last))
        traces.append(PacketTrace(
            packet_id=b, flow_key=canonical_flow_key(wire.b_five_tuple[b]),
            five_tuple=wire.b_five_tuple[b], src=wire.b_src[b],
            dst=wire.b_dst[b], cls=wire.b_cls[b], seq=int(wire.b_seq[b]),
            txn_id=int(wire.b_txn[b]), role=int(wire.b_role[b]),
            send_us=float(wire.b_src_ts_ns[b] / NS_PER_US), hops=hops,
            n_missing=n_h - len(recs), measured_us=measured,
            path_sig=path.signature))
    return traces


# ---------------------------------------------------------------------------
# 4. flow aggregation
# ---------------------------------------------------------------------------
@dataclass
class FlowAgg:
    flow_key: int
    five_tuple: Tuple[str, str, int, int, int]
    cls: str
    src: str
    dst: str
    path_sig: str
    packet_ids: List[int] = field(default_factory=list)
    queue_us: List[float] = field(default_factory=list)
    dominant_hops: List[int] = field(default_factory=list)
    n_packets: int = 0
    n_marked: int = 0

    def summary(self) -> dict:
        q = np.asarray(self.queue_us)
        return {
            "flow_key": self.flow_key,
            "five_tuple": list(self.five_tuple),
            "cls": self.cls, "src": self.src, "dst": self.dst,
            "path_sig": self.path_sig,
            "packets": self.n_packets, "marked": self.n_marked,
            "queue_total_us": float(q.sum()) if q.size else 0.0,
            "queue_p95_us": float(np.percentile(q, 95)) if q.size else 0.0,
        }


class Collector:
    """Stateful collector: ingest a batch of wire traces, answer queries."""

    def __init__(self, topo: Topology):
        self.topo = topo
        self.traces: List[PacketTrace] = []
        self.by_packet: Dict[int, PacketTrace] = {}
        self.flows: Dict[int, FlowAgg] = {}
        self.txns: Dict[int, List[PacketTrace]] = {}
        self.clocks: Optional[ClockEstimate] = None
        self.wire_bytes = 0
        self.n_packets = 0
        # per-port queue history: port_key -> list of (t_us, queue_us, flow_key)
        self.port_history: Dict[int, List[Tuple[float, float, int]]] = {}
        self.port_rates: Dict[int, List[Tuple[float, float]]] = {}

    # ------------------------------------------------------------------
    def ingest(self, wire: WireTraces) -> None:
        _, recs_by_pkt = _records_by_packet(wire)
        self.clocks = estimate_clocks(recs_by_pkt, wire, self.topo)
        traces = build_traces(wire, self.topo, self.clocks, recs_by_pkt)
        rates = estimate_port_rates(traces, self.topo)
        _recompute_depth_delays(traces, rates, self.topo)
        self.port_rates = rates
        self.wire_bytes += wire.wire_bytes()
        self.n_packets += wire.n_packets
        base = len(self.traces)
        self.traces.extend(traces)
        for i, t in enumerate(traces):
            t.packet_id = base + i
            self.by_packet[t.packet_id] = t
            fa = self.flows.get(t.flow_key)
            if fa is None:
                fa = FlowAgg(t.flow_key, t.five_tuple, t.cls, t.src, t.dst,
                             t.path_sig)
                self.flows[t.flow_key] = fa
            fa.packet_ids.append(t.packet_id)
            if t.txn_id >= 0:
                self.txns.setdefault(t.txn_id, []).append(t)
            best = [(h.queue_best, h.hop) for h in t.hops
                    if h.observed and h.queue_best == h.queue_best]
            fa.queue_us.append(float(sum(q for q, _ in best)))
            fa.dominant_hops.append(max(best)[1] if best else -1)
            fa.n_marked += 1
            for h in t.hops:
                if h.observed and h.queue_us == h.queue_us:
                    self.port_history.setdefault(h.port_key, []).append(
                        (t.send_us, h.queue_us, t.flow_key))

    # ------------------------------------------------------------------
    def transaction_for(self, five_tuple: Tuple[str, str, int, int, int],
                        t_us: float) -> Optional[int]:
        """Find the transaction whose response crossed the fabric at t_us."""
        key = canonical_flow_key(five_tuple)
        best, best_dt = None, float("inf")
        for txn_id, traces in self.txns.items():
            for t in traces:
                if t.flow_key != key:
                    continue
                dt = abs(t.send_us - t_us)
                if dt < best_dt:
                    best_dt, best = dt, txn_id
        return best

    def trace_window(self, txn_id: int) -> List[PacketTrace]:
        return self.txns.get(txn_id, [])

    # ------------------------------------------------------------------
    def attribute(self, txn_id: int, window_us: float = 400.0,
                  top_k: int = 6) -> dict:
        """Break a transaction's latency down hop by hop and name the culprit.

        Returns the per-hop decomposition of the *slowest* response packet,
        the dominant hop, and the other flows that shared that egress port in
        the same window.
        """
        traces = sorted(self.txns.get(txn_id, []), key=lambda t: t.send_us)
        if not traces:
            return {"error": f"transaction {txn_id} not observed"}
        # Attribute on the packet that paid the most, but report the *mean*
        # queueing per hop over the response train: a burst makes the first
        # packets pay the full queue and later ones progressively less, and
        # the operator's question is "how much of this request's latency was
        # queued at hop k", not "which single packet was worst".
        slowest = max(traces, key=lambda t: t.total_queue_us)
        dom = slowest.dominant()
        hop_rows = []
        for h in slowest.hops:
            hop_rows.append({
                "hop": h.hop, "switch": h.sw_id,
                "in_port": h.in_port, "out_port": h.out_port,
                "observed": bool(h.observed),
                "queue_us": None if h.depth_queue_us != h.depth_queue_us
                             else round(float(h.depth_queue_us), 2),
                "queue_ts_us": None if h.queue_us != h.queue_us
                                else round(float(h.queue_us), 2),
                "residence_us": None if h.residence_us != h.residence_us
                                 else round(float(h.residence_us), 2),
                "qdepth_bytes": None if h.qdepth_bytes != h.qdepth_bytes
                                 else round(float(h.qdepth_bytes), 1),
                "retx": bool(h.retx),
            })
        culprits = []
        if dom is not None:
            culprits = self.co_flows(slowest.packet_id, dom, window_us, top_k)
        # mean per-hop queueing over every observed packet of the transaction
        n_h = max((len(t.hops) for t in traces), default=0)
        per_hop_sum = np.zeros(n_h)
        per_hop_n = np.zeros(n_h)
        for t in traces:
            for h in t.hops:
                if h.observed and h.queue_best == h.queue_best:
                    per_hop_sum[h.hop] += float(h.queue_best)
                    per_hop_n[h.hop] += 1.0
        per_hop_mean = np.where(per_hop_n > 0, per_hop_sum / np.maximum(per_hop_n, 1),
                                np.nan)
        for row in hop_rows:
            k = row["hop"]
            if k < n_h and per_hop_n[k] > 0:
                row["mean_queue_us"] = round(float(per_hop_mean[k]), 2)
                row["samples"] = int(per_hop_n[k])
            else:
                row["mean_queue_us"] = None
                row["samples"] = 0
        total_q = float(np.nansum(per_hop_mean))
        return {
            "txn_id": int(txn_id),
            "five_tuple": list(slowest.five_tuple),
            "flow_key": slowest.flow_key,
            "cls": slowest.cls,
            "src": slowest.src, "dst": slowest.dst,
            "path": list(slowest.path_switch_ids()),
            "path_sig": slowest.path_sig,
            "packets_observed": len(traces),
            "missing_hops": slowest.n_missing,
            "t_start_us": round(traces[0].send_us, 1),
            "hop_breakdown": hop_rows,
            "queue_total_us": round(total_q, 2),
            "dominant_hop": None if dom is None else {
                "hop": dom.hop, "switch": dom.sw_id,
                "out_port": dom.out_port, "port_key": dom.port_key,
                "queue_us": round(float(per_hop_mean[dom.hop]), 2)
                if per_hop_n[dom.hop] > 0 else round(float(dom.queue_best), 2),
                "share": (round(float(per_hop_mean[dom.hop]) / total_q, 3)
                          if total_q > 0 and per_hop_n[dom.hop] > 0 else 0.0),
            },
            "co_flows": culprits,
        }

    def co_flows(self, packet_id: int, dom: HopEstimate, window_us: float,
                 top_k: int) -> List[dict]:
        """Flows that shared the congested egress port in the same window.

        The window is sized from the delay being explained: a queue that
        added 16 ms of latency was fed by traffic arriving across those 16 ms,
        so a fixed 400 us window would miss almost all of it.
        """
        t = self.by_packet[packet_id]
        hist = self.port_history.get(dom.port_key)
        if not hist:
            return []
        delay = float(dom.queue_best)
        if not (delay == delay) or delay <= 0:
            delay = float(dom.residence_us) if dom.residence_us == dom.residence_us else 0.0
        half = min(COFLOW_MAX_WINDOW_US,
                   max(COFLOW_MIN_WINDOW_US, COFLOW_WINDOW_FACTOR * delay))
        arr = np.asarray([h[0] for h in hist])
        q = np.asarray([h[1] for h in hist])
        fk = np.asarray([h[2] for h in hist])
        lo, hi = t.send_us - half, t.send_us + half
        m = (arr >= lo) & (arr <= hi) & (fk != t.flow_key)
        if not m.any():
            return []
        out: Dict[int, float] = {}
        cnt: Dict[int, int] = {}
        for f, qq in zip(fk[m], q[m]):
            out[int(f)] = out.get(int(f), 0.0) + float(qq)
            cnt[int(f)] = cnt.get(int(f), 0) + 1
        ranked = sorted(out.items(), key=lambda kv: -kv[1])[:top_k]
        res = []
        for f, qq in ranked:
            fa = self.flows.get(f)
            res.append({
                "flow_key": f,
                "five_tuple": list(fa.five_tuple) if fa else None,
                "cls": fa.cls if fa else None,
                "src": fa.src if fa else None,
                "dst": fa.dst if fa else None,
                "packets": cnt[f],
                "queue_sum_us": round(qq, 2),
            })
        return res

    # ------------------------------------------------------------------
    def flow_table(self, limit: Optional[int] = None) -> List[dict]:
        rows = [f.summary() for f in self.flows.values()]
        rows.sort(key=lambda r: -r["queue_total_us"])
        return rows[:limit] if limit else rows

    def stats(self) -> dict:
        n_hops = sum(len(t.hops) for t in self.traces)
        n_obs = sum(1 for t in self.traces for h in t.hops if h.observed)
        return {
            "packets": self.n_packets,
            "records": n_obs,
            "expected_records": n_hops,
            "truncation_loss": 1.0 - (n_obs / n_hops) if n_hops else 0.0,
            "flows": len(self.flows),
            "txns": len(self.txns),
            "wire_bytes": self.wire_bytes,
            "complete_paths": sum(1 for t in self.traces if t.complete),
        }
