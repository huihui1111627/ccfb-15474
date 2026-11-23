"""Data-plane simulation + in-band telemetry marking.

The data plane is computed *analytically* rather than by a per-packet event
loop, which keeps a 250k-packet, 7-hop fabric run under a second:

For every egress port (a FIFO with a fixed drain rate) the finish time of the
last packet already queued when packet *p* arrives is enough to know both

* the queueing delay of *p*  = ``max(0, finish_prev - service_start)``
* the backlog in bytes at its arrival = ``delay * drain_rate``

(the port is work-conserving and FIFO, so bytes-backlog = time-to-drain x
rate, exactly).  Packets are processed hop by hop; within a hop, packets are
grouped per egress port and sorted by service start, so the running maximum is
a segmented ``np.maximum.accumulate``.  No Python loop over packets.

Faults
------
* intermittent port slowdown: an egress port's drain rate is multiplied for a
  time window (a congested / mis-shaped port)
* link errors: a packet traversing a bad link is retransmitted with
  probability p, paying re-serialisation + a fast-retransmit penalty, and the
  hop record carries a ``RETX`` flag
* clock skew: every switch timestamp is ``(1 + drift) * t + offset`` on its
  own clock, quantised to 1 ns

Telemetry marking happens in a second pass, over the same trace, so that the
sampling policy never changes the data plane (both policies see identical
traffic).  A switch that observes a queue above the trigger threshold (or a
hop residence above the latency threshold) marks **every** packet traversing
it for the next ``trigger_window_us``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import INT_HOP_BYTES, TelemetryConfig, ScenarioConfig
from .topology import HOST_PROP_US, MAX_PORTS, Topology
from .traffic import HOST_RATE_BPUS, SWITCH_PROC_US, Traffic

INT_BASE_BYTES = 24
QFLAG_RETX = 0x01
QFLAG_TRUNC = 0x02
QFLAG_KEY = 0x04
QFLAG_FULLMARK = 0x08
#: the 16-bit queue-depth field saturates; the flag says so explicitly so the
#: collector can fall back to the timestamp instead of believing a clipped value
QFLAG_DEPTH_SAT = 0x10

#: Queue depth encoding: 16 bits, 128 B units -> 8.3 MB maximum, which covers
#: a 16 ms deep-buffer port at 40 Gbps.  Anything larger saturates and is
#: flagged; silently clipping it would understate a multi-millisecond stall.
QDEPTH_UNIT_BYTES = 128
QDEPTH_MAX_UNITS = 65535


# ---------------------------------------------------------------------------
# clock model
# ---------------------------------------------------------------------------
@dataclass
class ClockModel:
    """Per-switch clock: ts = (1 + drift) * t + offset (us), ns-quantised."""

    offset_us: np.ndarray          # per switch index
    drift: np.ndarray              # dimensionless (ppb * 1e-9)
    jitter_ns: float = 30.0        # read jitter, stddev

    def stamp(self, sw: np.ndarray, t_true: np.ndarray,
              rng: np.random.Generator) -> np.ndarray:
        """Return switch-clock timestamp in nanoseconds (int64, 48-bit)."""
        ts = (t_true + self.offset_us[sw]
              + self.drift[sw] * t_true
              + rng.normal(0.0, self.jitter_ns / 1000.0, t_true.shape))
        ns = np.rint(ts * 1000.0).astype(np.int64)
        return ns & ((1 << 48) - 1)

    def true_stamp(self, sw: np.ndarray, t_true: np.ndarray) -> np.ndarray:
        ts = t_true + self.offset_us[sw] + self.drift[sw] * t_true
        return np.rint(ts * 1000.0).astype(np.int64)


def make_clocks(topo: Topology, bias_ppb: Dict[str, float], seed: int) -> ClockModel:
    rng = np.random.default_rng(seed ^ 0xC10CC)
    n = topo.n_switches
    offset = rng.uniform(-25.0, 25.0, n)              # +/-25 us
    drift = rng.uniform(-120e-9, 120e-9, n)           # +/-120 ppb
    for sid, ppb in (bias_ppb or {}).items():
        i = topo.sw_index(sid)
        drift[i] = ppb * 1e-9
        offset[i] = 0.0
    return ClockModel(offset_us=offset, drift=drift)


# ---------------------------------------------------------------------------
# impairment
# ---------------------------------------------------------------------------
@dataclass
class Impairment:
    #: (port_key, t0, t1, slowdown factor in (0,1])
    port_slowdowns: List[Tuple[int, float, float, float]] = field(default_factory=list)
    #: (link_id, t0, t1, retx probability)
    link_retx: List[Tuple[int, float, float, float]] = field(default_factory=list)

    def slowdown_factor(self, port_key: int) -> Tuple[np.ndarray, np.ndarray]:
        return np.empty(0), np.empty(0)

    @staticmethod
    def factor_at(events: Sequence[Tuple[float, float, float]], t: np.ndarray,
                  default: float = 1.0, combine: str = "min") -> np.ndarray:
        """Value of a timed impairment at time ``t``.

        ``combine='min'`` suits a *slowdown factor* (worst impairment wins);
        ``combine='max'`` suits a *probability* (any impairment can trigger).
        """
        out = np.full(t.shape, default, dtype=np.float64)
        for t0, t1, val in events:
            inside = (t >= t0) & (t < t1)
            if combine == "min":
                out = np.where(inside, np.minimum(out, val), out)
            else:
                out = np.where(inside, np.maximum(out, val), out)
        return out


def build_impairment(topo: Topology, scenario: ScenarioConfig) -> Impairment:
    imp = Impairment()
    for spec in getattr(scenario, "shapers", []) or []:
        # (switch_id, out_port, factor) for the whole run
        sid, port, factor = spec[0], int(spec[1]), float(spec[2])
        pk = topo.ports[(topo.sw_index(sid), port)]
        imp.port_slowdowns.append((pk, -1e18, 1e18, factor))
    for spec in scenario.congestion:
        # spec: (switch_id, out_port, t0, t1, factor)
        sid, port, t0, t1 = spec[0], int(spec[1]), float(spec[2]), float(spec[3])
        factor = float(spec[4]) if len(spec) > 4 else 0.35
        pk = topo.ports[(topo.sw_index(sid), port)]
        imp.port_slowdowns.append((pk, t0, t1, factor))
    for spec in scenario.link_impair:
        # spec: (src_id, dst_id, t0, t1, retx_prob)
        a, b, t0, t1 = spec[0], spec[1], float(spec[2]), float(spec[3])
        p = float(spec[4]) if len(spec) > 4 else 0.02
        li = _find_link(topo, a, b)
        imp.link_retx.append((li, t0, t1, p))
    return imp


def _find_link(topo: Topology, a: str, b: str) -> int:
    for l in topo.links:
        if l.src == a and l.dst == b:
            return l.link_id
    for l in topo.links:
        if l.src == b and l.dst == a:
            return l.link_id
    raise KeyError((a, b))


# ---------------------------------------------------------------------------
# data plane
# ---------------------------------------------------------------------------
@dataclass
class DataPlane:
    traffic: Traffic
    topo: Topology
    max_h: int
    #: (n_packets, max_h) -- NaN / padded beyond a packet's path length
    t_arrive: np.ndarray        # true arrival at each switch (us, real time)
    t_depart: np.ndarray        # true departure from each switch
    qdelay: np.ndarray          # queueing delay at each hop (us)
    ser_us: np.ndarray          # serialisation time at each hop's egress
    qdepth_bytes: np.ndarray    # backlog seen on arrival (bytes)
    qdepth_us: np.ndarray       # same backlog expressed as drain time (us)
    retx: np.ndarray            # 0/1
    hop_port: np.ndarray        # port_key per hop
    hop_sw: np.ndarray
    hop_link: np.ndarray
    rate_eff: np.ndarray        # drain rate used (bytes/us)
    n_hops: np.ndarray          # per packet
    t_deliver: np.ndarray       # true arrival at the destination host
    port_max_depth: Dict[int, float] = field(default_factory=dict)
    port_busy: Dict[int, float] = field(default_factory=dict)


def _path_tables(topo: Topology, traffic: Traffic, max_h: int):
    n_p = len(topo._paths)
    sw = np.full((n_p, max_h), -1, dtype=np.int32)
    port = np.full((n_p, max_h), -1, dtype=np.int32)
    link = np.full((n_p, max_h), -1, dtype=np.int32)
    rate = np.zeros((n_p, max_h))
    prop_in = np.zeros((n_p, max_h))
    cap = np.zeros((n_p, max_h))
    plen = np.zeros(n_p, dtype=np.int32)
    for p in topo._paths:
        plen[p.path_id] = p.n_hops
        for h, hop in enumerate(p.hops):
            sw[p.path_id, h] = hop.sw
            port[p.path_id, h] = hop.port_key
            link[p.path_id, h] = hop.link
            rate[p.path_id, h] = hop.rate_bpus
            cap[p.path_id, h] = hop.queue_cap_bytes
            prop_in[p.path_id, h] = HOST_PROP_US if h == 0 else p.hops[h - 1].prop_us
    return sw, port, link, rate, prop_in, cap, plen


def run_data_plane(traffic: Traffic, topo: Topology, imp: Impairment,
                   clocks: ClockModel, seed: int,
                   proc_us: float = SWITCH_PROC_US) -> DataPlane:
    rng = np.random.default_rng(seed ^ 0xDA7A)
    n = traffic.n_packets
    max_h = max(p.n_hops for p in topo._paths)
    (sw_t, port_t, link_t, rate_t, prop_t, cap_t,
     plen_t) = _path_tables(topo, traffic, max_h)

    pid = traffic.path_id
    n_hops = plen_t[pid]
    size = traffic.size.astype(np.float64)

    t_arrive = np.full((n, max_h), np.nan)
    t_depart = np.full((n, max_h), np.nan)
    qdelay = np.zeros((n, max_h))
    ser_us = np.zeros((n, max_h))
    qdepth = np.zeros((n, max_h))
    retx = np.zeros((n, max_h), dtype=np.uint8)
    act_mask = (np.arange(max_h)[None, :] < plen_t[pid][:, None])
    hop_port = np.where(act_mask, port_t[pid], -1).astype(np.int32)
    hop_sw = np.where(act_mask, sw_t[pid], -1).astype(np.int32)
    hop_link = np.where(act_mask, link_t[pid], -1).astype(np.int32)
    rate_eff_all = np.zeros((n, max_h))
    u_retx = rng.random((n, max_h))

    # port slowdown lookup tables (port_key -> [(t0,t1,factor)])
    slow: Dict[int, List[Tuple[float, float, float]]] = {}
    for pk, t0, t1, f in imp.port_slowdowns:
        slow.setdefault(pk, []).append((t0, t1, f))
    retx_link: Dict[int, List[Tuple[float, float, float]]] = {}
    for li, t0, t1, p in imp.link_retx:
        retx_link.setdefault(li, []).append((t0, t1, p))

    order_flat = np.arange(n)
    prev_dep = traffic.t_send.astype(np.float64)     # leaves the source host

    for h in range(max_h):
        act = np.nonzero(n_hops > h)[0]
        if act.size == 0:
            break
        ports = hop_port[act, h]
        links = hop_link[act, h]
        rate = rate_t[pid[act], h]
        arr = prev_dep[act] + prop_t[pid[act], h]
        t_arrive[act, h] = arr

        # --- effective drain rate (intermittent port slowdown) -------------
        fac = np.ones(act.size)
        for pk in np.unique(ports):
            ev = slow.get(int(pk))
            if not ev:
                continue
            m = ports == pk
            fac[m] = Impairment.factor_at(ev, arr[m])
        rate_e = rate * fac
        ser = size[act] / rate_e

        # --- retransmission on a bad link ---------------------------------
        p_retx = np.zeros(act.size)
        for li in np.unique(links):
            ev = retx_link.get(int(li))
            if not ev:
                continue
            m = links == li
            p_retx[m] = Impairment.factor_at(ev, arr[m], default=0.0,
                                             combine="max")
        is_retx = (u_retx[act, h] < p_retx)
        penalty = 2.0 * ser + 5.0
        ser_total = ser + np.where(is_retx, penalty, 0.0)

        svc = arr + proc_us
        # --- FIFO per egress port ------------------------------------------
        # For a work-conserving FIFO with packet service times se_k and service
        # start s_k, the finish time is
        #     fin_i = C_i + max_{j<=i} (s_j - C_{j-1}),   C_i = sum_{k<=i} se_k
        # and queueing delay is fin_i - se_i - s_i.  Computing it per port group
        # (there are at most a few hundred groups per hop) is exact and fast.
        ordidx = np.lexsort((svc, ports))
        p_sorted = ports[ordidx]
        s_sorted = svc[ordidx]
        ser_sorted = ser_total[ordidx]
        # buffer capacity as a drain time, per packet in sorted order
        cap_bytes_sorted = cap_t[pid[act], h][ordidx]
        rate_sorted = rate_e[ordidx]
        cap_by_port = np.where(rate_sorted > 0,
                               cap_bytes_sorted / np.maximum(rate_sorted, 1e-12),
                               np.inf)
        qd = np.empty_like(s_sorted)
        newg = np.empty(p_sorted.size, dtype=bool)
        newg[0] = True
        newg[1:] = p_sorted[1:] != p_sorted[:-1]
        starts = np.nonzero(newg)[0]
        ends = np.append(starts[1:], p_sorted.size)
        for g0, g1 in zip(starts, ends):
            ss = s_sorted[g0:g1]
            ee = ser_sorted[g0:g1]
            cs = np.cumsum(ee)
            cprev = cs - ee
            w = ss - cprev
            runmax = np.maximum.accumulate(w)
            fin = cs + runmax
            busy = fin - ee - ss
            # A real egress buffer holds a finite number of bytes: once the
            # backlog exceeds the buffer the switch tail-drops, so no packet
            # ever waits longer than one buffer-drain time.  Without this the
            # model would let a 40 ms queue build behind a 16 ms buffer, and
            # the byte counter (capped at the buffer) could never explain it.
            pk = int(p_sorted[g0])
            cap_us = float(cap_by_port[g0])
            qd[g0:g1] = np.minimum(busy, cap_us)
        dep_sorted = s_sorted + ser_sorted + qd
        qd_orig = np.empty_like(qd)
        dep_orig = np.empty_like(dep_sorted)
        qd_orig[ordidx] = qd
        dep_orig[ordidx] = dep_sorted

        # --- write back ------------------------------------------------------
        t_depart[act, h] = dep_orig
        qdelay[act, h] = qd_orig
        ser_us[act, h] = ser_total
        qdepth[act, h] = np.minimum(qd_orig * rate_e, cap_t[pid[act], h])
        retx[act, h] = is_retx.astype(np.uint8)
        rate_eff_all[act, h] = rate_e
        prev_dep[act] = dep_orig

    # deliver at destination host
    last_h = n_hops - 1
    rows = np.arange(n)
    t_deliver = t_depart[rows, last_h] + HOST_PROP_US

    dp = DataPlane(traffic=traffic, topo=topo, max_h=max_h,
                   t_arrive=t_arrive, t_depart=t_depart, qdelay=qdelay,
                   ser_us=ser_us, qdepth_bytes=qdepth,
                   # a byte counter can only ever report up to the buffer
                   # capacity, so its drain-time equivalent is bounded too
                   qdepth_us=np.minimum(qdelay, np.where(rate_eff_all > 0,
                                                         cap_t[pid] / np.maximum(rate_eff_all, 1e-12),
                                                         qdelay)),
                   retx=retx,
                   hop_port=hop_port, hop_sw=hop_sw, hop_link=hop_link,
                   rate_eff=rate_eff_all, n_hops=n_hops, t_deliver=t_deliver)
    dp.port_max_depth = _port_maxima(dp)
    return dp


def _port_maxima(dp: DataPlane) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for h in range(dp.max_h):
        m = dp.n_hops > h
        if not m.any():
            continue
        pk = dp.hop_port[m, h]
        qd = dp.qdepth_bytes[m, h]
        for p in np.unique(pk):
            v = float(qd[pk == p].max())
            if v > out.get(int(p), 0.0):
                out[int(p)] = v
    return out


# ---------------------------------------------------------------------------
# source timestamps / true task latency
# ---------------------------------------------------------------------------
def annotate_transactions(dp: DataPlane) -> None:
    """Fill in ground-truth latency for every request/response transaction."""
    tr = dp.traffic
    rows = np.arange(tr.n_packets)
    deliver = dp.t_deliver
    n_hops = dp.n_hops
    # last-hop departure is a better "wire" reference for the response leg
    resp_rows = np.nonzero(tr.role == 1)[0]
    first = {}
    last = {}
    for r in resp_rows:
        txn = int(tr.txn_id[r])
        if txn not in first:
            first[txn] = r
        last[txn] = r
    req_rows = {}
    for r in np.nonzero(tr.role == 0)[0]:
        req_rows[int(tr.txn_id[r])] = r
    resp_rows_of: Dict[int, List[int]] = {}
    for r in resp_rows:
        resp_rows_of.setdefault(int(tr.txn_id[r]), []).append(int(r))
    for tx in tr.txns:
        r0 = req_rows.get(tx.txn_id)
        r1 = last.get(tx.txn_id)
        if r0 is None or r1 is None:
            continue
        t_done = deliver[r1]
        tx.true_latency_us = float(t_done - float(tr.t_send[r0]))
        tx.net_latency_us = float(tx.true_latency_us - tx.server_delay_us)
        # dominant hop over the whole response leg.  Total queueing summed
        # over packets is the wrong yardstick: a burst inflates the first
        # packets' delay and every later packet pays a *smaller* delay that is
        # really the same queue draining.  The honest measure is the
        # queueing the *last* packet paid -- that is the queue depth the
        # transaction actually had to sit behind -- plus, for a stall that
        # clears mid-transaction, the peak instantaneous delay across hops.
        rows = resp_rows_of.get(tx.txn_id, [])
        if rows:
            rows = np.asarray(rows, dtype=np.int64)
            max_h = int(n_hops[rows].max())
            q = np.zeros(max_h)
            n_at = np.zeros(max_h)
            for h in range(max_h):
                m = n_hops[rows] > h
                if m.any():
                    q[h] = float(dp.qdelay[rows[m], h].sum())
                    n_at[h] = float(m.sum())
            mean_q = q / np.maximum(n_at, 1.0)
            k = int(np.argmax(mean_q))
            tx.dominant_hop = k
            tx.dominant_port = int(dp.hop_port[rows[0], k]) if k < n_hops[rows[0]] else -1
            tx.dominant_queue_us = float(mean_q[k])
            order = np.argsort(-mean_q)
            tx.worst_hops = tuple(int(i) for i in order[:3])


# ---------------------------------------------------------------------------
# telemetry marking (pass 2)
# ---------------------------------------------------------------------------
@dataclass
class MarkingResult:
    #: packet indices that were marked at the source
    marked: np.ndarray
    #: per-packet probability actually used
    prob: np.ndarray
    #: per-packet reason codes (0 base, 1 class, 2 path, 3 trigger)
    reason: np.ndarray
    #: per-switch trigger times (true us) for re-play
    trigger_times_by_sw: Dict[int, np.ndarray]
    #: trigger events as arrays: sw, t_us, port_key, qdepth_bytes, qdelay_us
    trigger_sw: np.ndarray
    trigger_t: np.ndarray
    trigger_port: np.ndarray
    trigger_qdepth: np.ndarray
    trigger_qdelay: np.ndarray
    base_prob: np.ndarray
    boost_mask: np.ndarray


#: Number of distinct random streams, so that a policy's sample set does not
#: depend on which other policies were evaluated.
N_STREAMS = 64


def proto_uniform(traffic: Traffic) -> np.ndarray:
    """A fixed pseudo-random draw per packet, in [0,1).

    Keyed by (flow 5-tuple, sequence number) so that *every* policy sees the
    same draw for the same packet.  Pooling: the flow id is itself a hash of
    the 5-tuple, so this is a 5-tuple-keyed draw.
    """
    from .topology import splitmix64
    out = np.empty(traffic.n_packets, dtype=np.float64)
    flow = traffic.flow_id.astype(np.int64)
    seq = traffic.seq.astype(np.int64)
    step = 4096
    for i in range(0, traffic.n_packets, step):
        f = flow[i:i + step]
        s = seq[i:i + step]
        x = f.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) \
            + s.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9) \
            + np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(30))
        x = x * np.uint64(0xBF58476D1CE4E5B9)
        x = x ^ (x >> np.uint64(27))
        x = x * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
        out[i:i + step] = (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return out


def compute_triggers(dp: DataPlane, tcfg: TelemetryConfig):
    """Switch-side trigger events, from local queue monitoring only.

    A switch fires when a packet arrives at an egress port whose backlog drain
    time exceeds ``queue_trigger_us``, or whose hop residence exceeds
    ``hop_latency_trigger_us``.  Once armed it full-marks all traffic for
    ``trigger_window_us``; successive firings inside the window are collapsed
    into one event per switch, which is what the hardware timer does.
    """
    times_by_sw: Dict[int, np.ndarray] = {}
    ev_sw, ev_t, ev_port, ev_qd, ev_ql = [], [], [], [], []
    for h in range(dp.max_h):
        m = dp.n_hops > h
        if not m.any():
            continue
        hot = m & ((dp.qdepth_us[:, h] > tcfg.queue_trigger_us) |
                   (dp.qdelay[:, h] > tcfg.hop_latency_trigger_us))
        if not hot.any():
            continue
        idx = np.nonzero(hot)[0]
        sws = dp.hop_sw[idx, h]
        ts = dp.t_arrive[idx, h]
        ports = dp.hop_port[idx, h]
        order = np.lexsort((ts, sws))
        sws, ts, ports = sws[order], ts[order], ports[order]
        keep = np.zeros(ts.size, dtype=bool)
        last_t: Dict[int, float] = {}
        for i in range(ts.size):
            s = int(sws[i])
            if (i == 0 or s != int(sws[i - 1])
                    or ts[i] - last_t[s] > tcfg.trigger_window_us):
                keep[i] = True
            last_t[s] = ts[i]
        sws, ts, ports = sws[keep], ts[keep], ports[keep]
        sel = idx[order][keep]
        for s in np.unique(sws):
            k = sws == s
            prev = times_by_sw.get(int(s))
            times_by_sw[int(s)] = (ts[k] if prev is None
                                   else np.sort(np.concatenate([prev, ts[k]])))
        ev_sw.append(sws); ev_t.append(ts); ev_port.append(ports)
        ev_qd.append(dp.qdepth_bytes[sel, h])
        ev_ql.append(dp.qdelay[sel, h])
    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0)
    return (times_by_sw,
            np.concatenate(ev_sw) if ev_sw else empty_i,
            np.concatenate(ev_t) if ev_t else empty_f,
            np.concatenate(ev_port) if ev_port else empty_i,
            np.concatenate(ev_qd) if ev_qd else empty_f,
            np.concatenate(ev_ql) if ev_ql else empty_f)


def trigger_state(times_by_sw: Dict[int, np.ndarray], sw: np.ndarray,
                  t: np.ndarray, window_us: float) -> np.ndarray:
    """True where the switch fired a trigger within ``window_us`` before t."""
    out = np.zeros(t.shape, dtype=bool)
    for s in np.unique(sw):
        m = sw == s
        tt = times_by_sw.get(int(s))
        if tt is None or tt.size == 0:
            continue
        idx = np.searchsorted(tt, t[m], side="right") - 1
        ok = idx >= 0
        res = np.zeros(int(m.sum()), dtype=bool)
        res[ok] = (t[m][ok] - tt[idx[ok]]) <= window_us
        out[m] = res
    return out


def mark_packets(dp: DataPlane, tcfg: TelemetryConfig, prob: np.ndarray,
                 reason: np.ndarray, use_trigger: bool = True) -> MarkingResult:
    """Decide which packets carry telemetry.

    ``prob`` is the source policy's per-packet probability (before triggers).
    With ``use_trigger`` a switch that has fired a trigger marks *everything*
    it forwards for ``trigger_window_us`` -- the in-band header of those
    packets carries the trigger bit, and every switch on the path that is
    armed appends its record.
    """
    n = dp.traffic.n_packets
    u = proto_uniform(dp.traffic)
    (times_by_sw, ev_sw, ev_t, ev_port,
     ev_qd, ev_ql) = compute_triggers(dp, tcfg) if use_trigger else (
        {}, np.empty(0, dtype=np.int64), np.empty(0), np.empty(0, dtype=np.int64),
        np.empty(0), np.empty(0))
    boost = np.zeros(n, dtype=bool)
    for h in range(dp.max_h):
        m = dp.n_hops > h
        if not m.any():
            continue
        idx = np.nonzero(m)[0]
        st = trigger_state(times_by_sw, dp.hop_sw[idx, h], dp.t_arrive[idx, h],
                           tcfg.trigger_window_us)
        boost[idx[st]] = True
    p_final = np.where(boost, 1.0, prob)
    marked = u < p_final
    rs = np.where(boost, 3, reason).astype(np.int8)
    return MarkingResult(marked=marked, prob=p_final, reason=rs,
                         trigger_times_by_sw=times_by_sw,
                         trigger_sw=ev_sw, trigger_t=ev_t, trigger_port=ev_port,
                         trigger_qdepth=ev_qd, trigger_qdelay=ev_ql,
                         base_prob=prob, boost_mask=boost)


# ---------------------------------------------------------------------------
# in-band metadata records (what actually rides in the header)
# ---------------------------------------------------------------------------
def hop_records(dp: DataPlane, pkt: int, clocks: ClockModel,
                rng: np.random.Generator, tcfg: TelemetryConfig) -> List[dict]:
    """Build the wire records a marked packet accumulates, then truncate.

    Each record is 12 bytes: sw_id(1) in_port(1) out_port(1) flags(1)
    qdepth(2, units of 128 B) ts(6, 48-bit ns).  Records are stamped with the
    **switch clock** and carry only what hardware can observe: its own index,
    the ports, the egress backlog and the departure timestamp.
    """
    n_h = int(dp.n_hops[pkt])
    recs = []
    for h in range(n_h):
        sw = int(dp.hop_sw[pkt, h])
        pk = int(dp.hop_port[pkt, h])
        out_port = pk - sw * MAX_PORTS
        in_port = _in_port(dp, pkt, h)
        depth = float(dp.qdepth_bytes[pkt, h])
        units = int(round(depth / QDEPTH_UNIT_BYTES))
        q16 = min(QDEPTH_MAX_UNITS, units)
        flags = 0
        if dp.retx[pkt, h]:
            flags |= QFLAG_RETX
        if units > QDEPTH_MAX_UNITS:
            flags |= QFLAG_DEPTH_SAT
        if depth > tcfg.queue_trigger_us * float(dp.rate_eff[pkt, h]):
            flags |= QFLAG_KEY
        ts_ns = int(clocks.true_stamp(np.array([sw]),
                                      np.array([dp.t_depart[pkt, h]]))[0])
        ts_ns = int(ts_ns + rng.normal(0.0, clocks.jitter_ns)) & ((1 << 48) - 1)
        recs.append({
            "hop": h, "sw": sw, "in_port": in_port, "out_port": out_port,
            "port_key": pk, "qdepth_bytes": q16 * QDEPTH_UNIT_BYTES,
            "ts_ns": ts_ns, "flags": flags, "link": int(dp.hop_link[pkt, h]),
            "true_qdelay": float(dp.qdelay[pkt, h]),
            "true_arrive": float(dp.t_arrive[pkt, h]),
            "true_depart": float(dp.t_depart[pkt, h]),
            "size": float(dp.traffic.size[pkt]),
            "rate": float(dp.rate_eff[pkt, h]),
        })
    return apply_budget(recs, tcfg)


def _in_port(dp: DataPlane, pkt: int, h: int) -> int:
    path = dp.topo._paths[int(dp.traffic.path_id[pkt])]
    return int(path.hops[h].in_port)


def apply_budget(recs: List[dict], tcfg: TelemetryConfig) -> List[dict]:
    """Enforce the metadata length cap according to the truncation policy.

    ``key_hops`` keeps the final hop (it proves the packet's arrival and
    closes the path) plus the deepest-queue hops, which is what makes an
    over-budget path still attributable.
    """
    max_recs = max(1, (tcfg.budget_bytes - INT_BASE_BYTES) // tcfg.hop_bytes)
    if len(recs) <= max_recs:
        return recs
    pol = tcfg.truncation_policy
    if pol == "first_k":
        out = recs[:max_recs]
    elif pol == "last_k":
        out = recs[-max_recs:]
    else:  # key_hops
        tail = recs[-1]
        others = sorted(recs[:-1], key=lambda r: (-r["qdepth_bytes"], r["hop"]))
        out = others[: max_recs - 1] + [tail]
        out.sort(key=lambda r: r["hop"])
    for r in out:
        r["flags"] |= QFLAG_TRUNC
    return out


# ---------------------------------------------------------------------------
# wire materialisation: exactly the bytes the collector receives
# ---------------------------------------------------------------------------
@dataclass
class WireTraces:
    """Columnar form of all INT records that reached the collector.

    One row per (marked packet, retained hop record).  ``ts_ns`` is the
    switch's own clock, quantised to 1 ns; nothing true-time ever appears in
    this structure -- it is what a collector on the wire could observe.
    """

    pkt: np.ndarray             # index into the traffic packet array
    hop: np.ndarray
    sw: np.ndarray
    in_port: np.ndarray
    out_port: np.ndarray
    port_key: np.ndarray
    link: np.ndarray
    ts_ns: np.ndarray
    qdepth_bytes: np.ndarray
    flags: np.ndarray
    # ---- base header (one per marked packet) ----
    b_pkt: np.ndarray
    b_flow: np.ndarray
    b_seq: np.ndarray
    b_txn: np.ndarray
    b_role: np.ndarray
    b_src_ts_ns: np.ndarray     # ingress stamp at the first switch (its clock)
    b_src_sw: np.ndarray        # index of that first switch
    b_size: np.ndarray
    b_src_interval_us: np.ndarray   # source-stamped inter-packet gap
    b_five_tuple: List[Tuple[str, str, int, int, int]]
    b_src: List[str]
    b_dst: List[str]
    b_cls: List[str]
    b_dropped: np.ndarray       # number of records the source had to drop

    @property
    def n_packets(self) -> int:
        return int(self.b_pkt.size)

    @property
    def n_records(self) -> int:
        return int(self.pkt.size)

    def wire_bytes(self) -> int:
        return self.n_packets * INT_BASE_BYTES + self.n_records * INT_HOP_BYTES


def materialise_wire(dp: DataPlane, tcfg: TelemetryConfig, marked: np.ndarray,
                     clocks: ClockModel, seed: int) -> WireTraces:
    """Build the collector's view of the INT stream for a marking mask."""
    tr = dp.traffic
    rng = np.random.default_rng(seed ^ 0x11E5)
    pkts = np.nonzero(marked)[0]
    n_hops = dp.n_hops[pkts]
    counts = n_hops.astype(np.int64)
    total = int(counts.sum())
    pkt_rep = np.repeat(pkts, counts)
    hop_rep = np.concatenate([np.arange(c, dtype=np.int32) for c in counts]) \
        if total else np.empty(0, dtype=np.int32)
    # vectorised hop index construction
    if total:
        offs = np.repeat(np.cumsum(counts) - counts, counts)
        hop_rep = (np.arange(total, dtype=np.int32) - offs.astype(np.int32))

    sw = dp.hop_sw[pkt_rep, hop_rep]
    port = dp.hop_port[pkt_rep, hop_rep]
    link = dp.hop_link[pkt_rep, hop_rep]
    depth = dp.qdepth_bytes[pkt_rep, hop_rep]
    flags = np.zeros(total, dtype=np.uint8)
    flags |= np.where(dp.retx[pkt_rep, hop_rep] > 0, QFLAG_RETX, 0).astype(np.uint8)
    key = depth > tcfg.queue_trigger_us * dp.rate_eff[pkt_rep, hop_rep]
    flags |= np.where(key, QFLAG_KEY, 0).astype(np.uint8)

    # switch-clock departure timestamps, ns, with read jitter
    ts_true = dp.t_depart[pkt_rep, hop_rep]
    ts = ts_true + clocks.offset_us[sw] + clocks.drift[sw] * ts_true
    ts = ts * 1000.0 + rng.normal(0.0, clocks.jitter_ns, total)
    ts_ns = np.rint(ts).astype(np.int64) & ((1 << 48) - 1)

    # in_port comes from the path table; a switch knows it, so it is in the record
    path_of_pkt = tr.path_id[pkt_rep]
    in_port = np.array([dp.topo._paths[int(p)].hops[int(h)].in_port
                        for p, h in zip(path_of_pkt, hop_rep)], dtype=np.int32) \
        if total else np.empty(0, dtype=np.int32)

    out_port = port - sw * MAX_PORTS
    units = np.rint(depth / QDEPTH_UNIT_BYTES).astype(np.int64)
    saturated = units > QDEPTH_MAX_UNITS
    flags = flags | np.where(saturated, QFLAG_DEPTH_SAT, 0).astype(np.uint8)
    qdepth = np.minimum(units, QDEPTH_MAX_UNITS) * QDEPTH_UNIT_BYTES

    # ---- truncation -----------------------------------------------------
    max_recs = max(1, (tcfg.budget_bytes - INT_BASE_BYTES) // tcfg.hop_bytes)
    keep = np.ones(total, dtype=bool)
    drops = np.zeros(pkts.size, dtype=np.int64)
    over = n_hops > max_recs
    if over.any():
        big = pkts[over]
        sel_idx = np.nonzero(np.isin(pkt_rep, big))[0]
        pol = tcfg.truncation_policy
        if pol == "first_k":
            # keep hops 0..max_recs-1 of each oversized packet
            pos_in_pkt = np.empty(total, dtype=np.int64)
            offs = np.repeat(np.cumsum(counts) - counts, counts)
            pos_in_pkt = np.arange(total, dtype=np.int64) - offs
            sel_idx = sel_idx[pos_in_pkt[sel_idx] >= max_recs]
            keep[sel_idx] = False
        elif pol == "last_k":
            pos_in_pkt = np.empty(total, dtype=np.int64)
            offs = np.repeat(np.cumsum(counts) - counts, counts)
            pos_in_pkt = np.arange(total, dtype=np.int64) - offs
            last_pos = np.repeat(counts - 1, counts)
            sel_idx = sel_idx[pos_in_pkt[sel_idx] < last_pos[sel_idx] - max_recs + 1]
            keep[sel_idx] = False
        else:  # key_hops: deepest-queue interior hops + the final hop
            n_int = max_recs - 1
            rows = sel_idx                      # rows of oversized packets
            pr = pkt_rep[rows]
            order = np.lexsort((hop_rep[rows], -qdepth[rows], pr))
            si = rows[order]
            prs = pkt_rep[si]
            starts = np.nonzero(np.r_[True, prs[1:] != prs[:-1]])[0]
            grp = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, si.size]))
            last_hop = n_hops[np.searchsorted(pkts, prs)] - 1
            is_last = hop_rep[si] == last_hop
            nf = np.nonzero(~is_last)[0]
            grp_nf = grp[nf]
            st_nf = np.nonzero(np.r_[True, grp_nf[1:] != grp_nf[:-1]])[0]
            rep = np.repeat(np.arange(st_nf.size), np.diff(np.r_[st_nf, nf.size]))
            rk = np.arange(nf.size) - st_nf[rep]      # rank within its packet
            tmp = np.zeros(si.size, dtype=bool)
            tmp[nf] = rk < n_int
            tmp |= is_last
            keep[si] = tmp
        sel_idx = np.nonzero(np.isin(pkt_rep, big) & ~keep)[0]
        drops = np.bincount(np.searchsorted(pkts, pkt_rep[sel_idx]),
                            minlength=pkts.size)
        # set the truncation bit on every record that survived from a cut packet
        cut = np.isin(pkt_rep, pkts[np.nonzero(drops)[0]])
        flags[cut] |= QFLAG_TRUNC
    keep_idx = np.nonzero(keep)[0]

    # base headers
    flows = tr.flows
    b_src = [flows[int(f)].src for f in tr.flow_id[pkts]]
    b_dst = [flows[int(f)].dst for f in tr.flow_id[pkts]]
    b_cls = [flows[int(f)].cls for f in tr.flow_id[pkts]]
    b_ft = [flows[int(f)].five_tuple for f in tr.flow_id[pkts]]
    # the source node stamps the packet on ingress with *its own* clock: this
    # is the first switch on the path, so the collector can correct it
    first_sw = dp.hop_sw[pkts, 0]
    src_ts = dp.t_arrive[pkts, 0]
    b_src_ns = np.rint((src_ts + clocks.offset_us[first_sw]
                        + clocks.drift[first_sw] * src_ts) * 1000.0
                       ).astype(np.int64) & ((1 << 48) - 1)

    return WireTraces(
        pkt=pkt_rep[keep_idx], hop=hop_rep[keep_idx], sw=sw[keep_idx],
        in_port=in_port[keep_idx], out_port=out_port[keep_idx],
        port_key=port[keep_idx], link=link[keep_idx], ts_ns=ts_ns[keep_idx],
        qdepth_bytes=qdepth[keep_idx], flags=flags[keep_idx],
        b_pkt=pkts, b_flow=tr.flow_id[pkts], b_seq=tr.seq[pkts],
        b_txn=tr.txn_id[pkts], b_role=tr.role[pkts], b_src_ts_ns=b_src_ns,
        b_src_sw=first_sw,
        b_size=tr.size[pkts],
        b_src_interval_us=np.r_[0.0, np.diff(tr.t_send[pkts])],
        b_five_tuple=b_ft, b_src=b_src, b_dst=b_dst,
        b_cls=b_cls, b_dropped=drops)
