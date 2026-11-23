"""Traffic model: request/response transactions plus bursty bulk flows.

Everything is generated analytically (no per-packet Python loop) so a few
seconds of a 64-host fabric is a few hundred thousand packets.

A *flow* is one direction of one L4 connection.  ``request_response`` classes
create transactions that pair a request flow (client -> server) with a
response flow (server -> client).  The transaction is what the business sees,
so its latency is measured end to end.

Burstiness
----------
Every flow carries a ``burstiness`` knob.  A train is generated from
exponential gaps at the flow's mean rate; then random runs of packets are
compressed to the host line rate, creating microbursts like a NIC's initial
congestion window.  That is what makes egress queues sawtooth even without
injected faults.

Hot servers
-----------
``FlowClass.dst_pool`` restricts a class to a handful of "storage nodes".
They are the incast targets, and the places the scenario injects congestion
into.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import FlowClass, TrafficConfig
from .topology import Topology, tuple_hash5

MSS = 1400.0
HOST_RATE_BPUS = 125.0          # 1 Gbps host NIC, bytes/us
SWITCH_PROC_US = 1.5


@dataclass
class FlowMeta:
    flow_id: int
    txn_id: int
    cls: str
    src: str
    dst: str
    path_id: int
    five_tuple: Tuple[str, str, int, int, int]
    role: str                  # 'req' | 'resp' | 'bulk'
    t_start: float
    size_bytes: float
    rate_bpus: float


@dataclass
class TxnMeta:
    txn_id: int
    cls: str
    client: str
    server: str
    t0: float
    req_flow: int
    resp_flow: int
    req_bytes: float
    resp_bytes: float
    server_delay_us: float
    #: filled in after simulation
    true_latency_us: float = np.nan
    net_latency_us: float = np.nan
    dominant_hop: int = -1          # hop index (into response path) or -1
    dominant_port: int = -1
    dominant_queue_us: float = 0.0
    worst_hops: Tuple[int, ...] = ()


@dataclass
class EpisodeFlow:
    """A burst source that is part of an injected congestion episode."""

    t_start: float
    t_end: float
    src: str
    dst: str
    path_id: int
    flow_id: int
    rate_bpus: float


@dataclass
class Traffic:
    # packet arrays, sorted by t_send
    t_send: np.ndarray
    size: np.ndarray
    flow_id: np.ndarray
    path_id: np.ndarray
    cls: np.ndarray
    txn_id: np.ndarray
    role: np.ndarray            # 0=req 1=resp 2=bulk 3=episode
    seq: np.ndarray
    flows: List[FlowMeta]
    txns: List[TxnMeta]
    episodes: List[EpisodeFlow]
    rng_seed: int
    #: cold-server pool used for congestion target selection
    server_pool: Tuple[str, ...] = ()

    @property
    def n_packets(self) -> int:
        return self.t_send.size


def _sizes_for(n: int, total: float, rng: np.random.Generator) -> np.ndarray:
    """Split ``total`` bytes into ``n`` packets with an MSS cap."""
    sizes = np.full(n, MSS, dtype=np.float64)
    sizes[-1] = max(64.0, total - MSS * (n - 1))
    if sizes[-1] <= 64.0:
        sizes = sizes[:-1]
        sizes[-1] = max(64.0, sizes[-1] + 64.0)
    return sizes


def _train_times(n: int, sizes: np.ndarray, rate_bpus: float,
                 burstiness: float, t0: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Packet send times for a flow with mean rate ``rate_bpus``."""
    if n == 0:
        return np.empty(0)
    mean_gap = float(np.mean(sizes)) / max(rate_bpus, 1e-6)
    gaps = rng.exponential(mean_gap, n)
    gaps[0] = 0.0
    if burstiness > 0.0 and n > 2:
        # compress runs of packets to host line rate
        n_bursts = max(1, int(round(n * burstiness / 8.0)))
        starts = rng.integers(1, n, size=n_bursts)
        for s in starts:
            k = int(rng.integers(3, 12))
            e = min(n, s + k)
            gaps[s:e] = sizes[s:e] / HOST_RATE_BPUS
    return t0 + np.cumsum(gaps)


def _pick_hosts(rng: np.random.Generator, hosts: Sequence[str], n: int,
                avoid: Optional[str] = None) -> List[str]:
    out = []
    for _ in range(n):
        while True:
            h = hosts[int(rng.integers(0, len(hosts)))]
            if h != avoid:
                out.append(h)
                break
    return out


def _hot_servers(topo: Topology, n: int, rng: np.random.Generator) -> Tuple[str, ...]:
    # concentrate on pod 0 hosts so that a single leaf's uplink is shared,
    # which is what a "hot storage pod" looks like operationally
    pod0 = [h for h in topo.host_order if h.startswith("host-0-")]
    idx = rng.choice(len(pod0), size=min(n, len(pod0)), replace=False)
    return tuple(sorted(pod0[int(i)] for i in idx))


def default_classes(topo: Topology) -> Tuple[FlowClass, ...]:
    all_hosts = tuple(topo.host_order)
    return (
        FlowClass("web", weight=0.50, size_bytes=9_000, rate_bpus=40,
                  burstiness=0.6, importance=1.0, request_response=True,
                  src_pool=all_hosts, dst_pool=all_hosts),
        FlowClass("rpc", weight=0.22, size_bytes=28_000, rate_bpus=110,
                  burstiness=0.5, importance=1.0, request_response=True,
                  src_pool=all_hosts, dst_pool=all_hosts),
        FlowClass("cache", weight=0.16, size_bytes=120_000, rate_bpus=260,
                  burstiness=0.4, importance=1.5, request_response=True,
                  src_pool=all_hosts, dst_pool=all_hosts),   # hot pool patched below
        FlowClass("storage", weight=0.08, size_bytes=800_000, rate_bpus=420,
                  burstiness=0.8, importance=0.6, request_response=False,
                  src_pool=all_hosts, dst_pool=all_hosts),
        FlowClass("backup", weight=0.04, size_bytes=2_200_000, rate_bpus=650,
                  burstiness=0.9, importance=0.3, request_response=False,
                  src_pool=all_hosts, dst_pool=all_hosts),
    )


def build_traffic(cfg: TrafficConfig, topo: Topology,
                  episode_specs: Sequence[dict] = ()) -> Traffic:
    rng = np.random.default_rng(cfg.seed)
    classes = list(cfg.classes)
    servers = _hot_servers(topo, 8, rng)
    # patch the cache class to hit the hot servers
    classes = [
        FlowClass(c.name, c.weight, c.size_bytes, c.rate_bpus, c.burstiness,
                  c.importance, c.request_response, c.src_pool,
                  servers if c.name == "cache" else c.dst_pool)
        for c in classes
    ]
    weights = np.asarray([c.weight for c in classes], dtype=np.float64)
    weights = weights / weights.sum()

    flows: List[FlowMeta] = []
    txns: List[TxnMeta] = []
    pk_t_send: List[np.ndarray] = []
    pk_size: List[np.ndarray] = []
    pk_flow: List[np.ndarray] = []
    pk_path: List[np.ndarray] = []
    pk_cls: List[np.ndarray] = []
    pk_txn: List[np.ndarray] = []
    pk_role: List[np.ndarray] = []
    pk_seq: List[np.ndarray] = []

    cls_names = [c.name for c in classes]
    n_flows = int(rng.poisson(cfg.flows_per_second * cfg.duration_us / 1e6))
    t_starts = rng.uniform(0.0, cfg.duration_us, size=n_flows)
    cls_idx = rng.choice(len(classes), size=n_flows, p=weights)

    def add_packets(times, sizes, flow_id, path_id, cls_i, txn_id, role, seqs):
        pk_t_send.append(times)
        pk_size.append(sizes)
        pk_flow.append(np.full(times.size, flow_id, dtype=np.int32))
        pk_path.append(np.full(times.size, path_id, dtype=np.int32))
        pk_cls.append(np.full(times.size, cls_i, dtype=np.int8))
        pk_txn.append(np.full(times.size, txn_id, dtype=np.int32))
        pk_role.append(np.full(times.size, role, dtype=np.int8))
        pk_seq.append(seqs)

    txn_id = 0
    for i in range(n_flows):
        cls = classes[int(cls_idx[i])]
        ci = int(cls_idx[i])
        t0 = float(t_starts[i])
        src = cls.src_pool[int(rng.integers(0, len(cls.src_pool)))]
        dst = cls.dst_pool[int(rng.integers(0, len(cls.dst_pool)))]
        if dst == src:
            continue
        sport = 20000 + int(rng.integers(0, 40000))
        dport = 8000 + ci
        proto = 6

        if cls.request_response:
            # ---- request leg -------------------------------------------------
            req_bytes = float(rng.integers(200, 1400))
            req_flow_id = len(flows)
            req_ft = (topo.hosts[src].ip, topo.hosts[dst].ip, sport, dport, proto)
            req_path = topo.path_for(src, dst, tuple_hash5(req_ft) & 0xFFFFFFFF)
            flows.append(FlowMeta(req_flow_id, txn_id, cls.name, src, dst,
                                  req_path.path_id, req_ft, "req", t0, req_bytes,
                                  min(cls.rate_bpus, HOST_RATE_BPUS)))
            times = np.array([t0])
            add_packets(times, np.array([req_bytes]), req_flow_id,
                        req_path.path_id, ci, txn_id, 0, np.array([0], dtype=np.int16))

            # ---- server think time ------------------------------------------
            # lognormal with a heavy tail: most requests are served from cache
            # in tens of microseconds, ~1% hit a slow path in the server
            sd = float(rng.lognormal(mean=np.log(90.0), sigma=0.55))
            if rng.random() < 0.01:
                sd += float(rng.uniform(8_000.0, 25_000.0))

            # ---- response leg ------------------------------------------------
            resp_flow_id = len(flows)
            resp_ft = (topo.hosts[dst].ip, topo.hosts[src].ip, dport, sport, proto)
            resp_path = topo.path_for(dst, src, tuple_hash5(resp_ft) & 0xFFFFFFFF)
            n_resp = max(1, int(np.ceil(cls.size_bytes / MSS)))
            sizes = _sizes_for(n_resp, cls.size_bytes, rng)
            resp_rate = min(cls.rate_bpus, HOST_RATE_BPUS)
            req_path_us = req_path.total_prop_us() + 40.0
            resp_t0 = t0 + req_bytes / HOST_RATE_BPUS + req_path_us + sd
            times = _train_times(n_resp, sizes, resp_rate, cls.burstiness,
                                 resp_t0, rng)
            seqs = np.arange(n_resp, dtype=np.int16)
            add_packets(times, sizes, resp_flow_id, resp_path.path_id, ci,
                        txn_id, 1, seqs)
            flows.append(FlowMeta(resp_flow_id, txn_id, cls.name, dst, src,
                                  resp_path.path_id, resp_ft, "resp", resp_t0,
                                  cls.size_bytes, resp_rate))
            txns.append(TxnMeta(txn_id, cls.name, src, dst, t0, req_flow_id,
                                resp_flow_id, req_bytes, cls.size_bytes, sd))
            txn_id += 1
        else:
            # ---- bulk transfer ------------------------------------------------
            flow_id = len(flows)
            ft = (topo.hosts[src].ip, topo.hosts[dst].ip, sport, dport, proto)
            path = topo.path_for(src, dst, tuple_hash5(ft) & 0xFFFFFFFF)
            n_pk = int(max(1, min(4000, np.ceil(cls.size_bytes / MSS))))
            sizes = _sizes_for(n_pk, cls.size_bytes, rng)
            rate = min(cls.rate_bpus, HOST_RATE_BPUS * 0.95)
            times = _train_times(n_pk, sizes, rate, cls.burstiness, t0, rng)
            seqs = np.arange(n_pk, dtype=np.int16)
            add_packets(times, sizes, flow_id, path.path_id, ci, -1, 2, seqs)
            flows.append(FlowMeta(flow_id, -1, cls.name, src, dst,
                                  path.path_id, ft, "bulk", t0,
                                  cls.size_bytes, rate))

    # ------------------------------------------------------------------
    # injected congestion episodes: many-to-one microbursts at a hot server
    # ------------------------------------------------------------------
    episodes: List[EpisodeFlow] = []
    ep_id = 0
    for spec in episode_specs:
        dst_host = spec.get("dst_host") or servers[spec.get("server_idx", 0) % len(servers)]
        t0 = float(spec["t_start"])
        t1 = float(spec["t_end"])
        period = float(spec.get("period_us", 50_000.0))
        burst = float(spec.get("burst_us", 500.0))
        n_send = int(spec.get("n_senders", 10))
        rate = float(spec.get("sender_rate_bpus", 118.0))
        t = t0 + float(rng.uniform(0, period * 0.4))
        while t < t1:
            senders = _pick_hosts(rng, topo.host_order, n_send, avoid=dst_host)
            for s in senders:
                ft = (topo.hosts[s].ip, topo.hosts[dst_host].ip,
                      30000 + int(rng.integers(0, 20000)), 9000, 17)
                path = topo.path_for(s, dst_host, tuple_hash5(ft) & 0xFFFFFFFF)
                n_pk = int(burst * rate / MSS)
                sizes = np.full(n_pk, MSS)
                # all senders fire inside the burst window: they overlap at
                # the victim, but the episode is short, so the victim's
                # egress queue drains again between episodes
                times = t + rng.uniform(0.0, burst, n_pk)
                times = np.sort(times)
                fid = len(flows)
                add_packets(times, sizes, fid, path.path_id, -1, -1, 3,
                            np.arange(n_pk, dtype=np.int16))
                flows.append(FlowMeta(fid, -1, "bulk", s, dst_host,
                                      path.path_id, ft, "episode", t,
                                      n_pk * MSS, rate))
                episodes.append(EpisodeFlow(t, t + burst, s, dst_host,
                                            path.path_id, fid, rate))
                ep_id += 1
            t += period * float(rng.uniform(0.85, 1.15))

    # ------------------------------------------------------------------
    t_all = np.concatenate(pk_t_send)
    order = np.argsort(t_all, kind="stable")
    traffic = Traffic(
        t_send=t_all[order].astype(np.float64),
        size=np.concatenate(pk_size).astype(np.float32)[order],
        flow_id=np.concatenate(pk_flow)[order],
        path_id=np.concatenate(pk_path)[order],
        cls=np.concatenate(pk_cls)[order],
        txn_id=np.concatenate(pk_txn)[order],
        role=np.concatenate(pk_role)[order],
        seq=np.concatenate(pk_seq)[order],
        flows=flows,
        txns=txns,
        episodes=episodes,
        rng_seed=cfg.seed,
        server_pool=servers,
    )
    return traffic


def flow_class_lookup(classes: Sequence[FlowClass]) -> Dict[str, FlowClass]:
    return {c.name: c for c in classes}
