r"""A structurally realistic 3-tier fat-tree data-centre fabric.

::

    super-spine-0  super-spine-1          (2 boxes, 4 x 40G down each)
        |   \        /   |
      spine-0 ... spine-3                 (4 boxes, 8 x 20G down, 2 x 40G up)
       /  \        /  \
    agg-0-0 agg-0-1 ... agg-3-1           (2 per pod, 2 x 10G down, 4 x 20G up)
       |             |
    leaf-0-0 ... leaf-3-1                 (2 per pod, 8 hosts + 2 x 10G up)
       |
    host-0-0-0 ...                        (64 hosts, 1G each)

Path shapes
-----------
* same leaf            : 1 hop
* same pod, other leaf : leaf -> agg -> leaf                        (3 hops)
* cross pod            : leaf -> agg -> spine -> super-spine ->
                         spine -> agg -> leaf                       (7 hops)

That 7-hop worst case is what makes the INT metadata budget bite.

Port map
--------
leaf-{p}-{l}  : 0..7 hosts, 8..9 uplinks to agg-{p}-0/1
agg-{p}-{a}   : 0..1 downlinks to leaf-{p}-0/1, 2..5 uplinks to spine-0..3
spine-{s}     : 0..7 downlinks to agg-{p}-{a} (port = 2*p+a), 8..9 uplinks
                to super-spine-0/1
super-spine-{k}: 0..3 downlinks to spine-0..3

Every directed link owns an *egress queue* on its source switch, indexed by a
global ``port_key = switch_idx * MAX_PORTS + port``.  Hosts are pure endpoints:
the host NIC paces itself, so host -> leaf is unqueued.  leaf -> host **is**
queued, which is where many-to-one incast shows up.

Each flow gets a 5-tuple hash and an ECMP decision at every branching switch.
Between two leaves in different pods there are up to 2*4*2*4*2 = 128 distinct
paths, exactly the path diversity a real fabric has (and the reason a per-path
sampling policy has something to say).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .config import LinkSpec, SwitchSpec

MAX_PORTS = 16
HOST_BPS = 1e9
LEAF_UPLINK_BPS = 10e9
AGG_UPLINK_BPS = 20e9
SPINE_UPLINK_BPS = 40e9
HOST_PROP_US = 2.0
LEAF_AGG_PROP_US = 3.0
AGG_SPINE_PROP_US = 5.0
SPINE_SUPER_PROP_US = 8.0

N_PODS = 4
LEAVES_PER_POD = 2
AGGS_PER_POD = 2
N_SPINES = 4
N_SUPERSPINES = 2
HOSTS_PER_LEAF = 8

QUEUE_SLACK_US = 16_000.0   # egress buffer = 16 ms of line rate
N_HOSTS = N_PODS * LEAVES_PER_POD * HOSTS_PER_LEAF


def bps_to_bpus(bps: float) -> float:
    return bps / 8.0 / 1e6


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def tuple_hash5(five_tuple) -> int:
    raw = "|".join(str(v) for v in five_tuple).encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def canonical_flow_key(five_tuple) -> int:
    """Bidirectional flow key: both legs of a transaction hash the same."""
    sip, dip, sport, dport, proto = five_tuple
    a = (sip, sport)
    b = (dip, dport)
    if a > b:
        a, b = b, a
    return tuple_hash5((a[0], a[1], b[0], b[1], proto)) & 0xFFFFFFFF


def leaf_sw_idx(pod: int, leaf: int) -> int:
    return pod * LEAVES_PER_POD + leaf


def agg_sw_idx(pod: int, a: int) -> int:
    return N_PODS * LEAVES_PER_POD + pod * AGGS_PER_POD + a


def spine_sw_idx(s: int) -> int:
    return N_PODS * LEAVES_PER_POD + N_PODS * AGGS_PER_POD + s


def superspine_sw_idx(k: int) -> int:
    return (N_PODS * LEAVES_PER_POD + N_PODS * AGGS_PER_POD
            + N_SPINES + k)


@dataclass(frozen=True)
class HostInfo:
    name: str
    leaf_idx: int
    leaf_port: int
    ip: str


@dataclass(frozen=True)
class Hop:
    index: int
    sw: int
    sw_id: str
    in_port: int
    out_port: int
    port_key: int
    link: int
    prop_us: float
    rate_bpus: float
    queue_cap_bytes: float


@dataclass
class Path:
    path_id: int
    src_host: str
    dst_host: str
    hops: Tuple[Hop, ...]
    signature: str
    switch_ids: Tuple[str, ...]

    @property
    def n_hops(self) -> int:
        return len(self.hops)

    def total_prop_us(self) -> float:
        return sum(h.prop_us for h in self.hops)


@dataclass
class Topology:
    switches: List[SwitchSpec]
    links: List[LinkSpec]
    ports: Dict[Tuple[int, int], int]
    port_link: Dict[int, int]
    port_rate: np.ndarray
    port_cap_bytes: np.ndarray
    hosts: Dict[str, HostInfo]
    host_order: List[str]
    _path_cache: Dict[tuple, Path] = field(default_factory=dict)
    _paths: List[Path] = field(default_factory=list)

    @property
    def n_switches(self) -> int:
        return len(self.switches)

    def sw_index(self, sw_id: str) -> int:
        for i, s in enumerate(self.switches):
            if s.switch_id == sw_id:
                return i
        raise KeyError(sw_id)

    def n_port_keys(self) -> int:
        return self.n_switches * MAX_PORTS

    # ------------------------------------------------------------------
    def path_for(self, src_host: str, dst_host: str, flow_key: int) -> Path:
        sh, dh = self.hosts[src_host], self.hosts[dst_host]
        k1 = splitmix64(flow_key ^ 0xA5A5) & 0xFFFF
        k2 = splitmix64(flow_key ^ 0x5A5A) & 0xFFFF
        k3 = splitmix64(flow_key ^ 0xC0FFEE) & 0xFFFF
        cache_key = (src_host, dst_host, k1, k2, k3)
        hit = self._path_cache.get(cache_key)
        if hit is None:
            hit = self._build_path(sh, dh, k1, k2, k3)
            self._path_cache[cache_key] = hit
            self._paths.append(hit)
        return hit

    def _build_path(self, sh: HostInfo, dh: HostInfo, k1: int, k2: int, k3: int) -> Path:
        hops: List[Hop] = []
        sl, dl = sh.leaf_idx, dh.leaf_idx
        ps, pd = sl // LEAVES_PER_POD, dl // LEAVES_PER_POD
        if sl == dl:
            hops.append(self._hop(sl, sh.leaf_port, dh.leaf_port))
        elif ps == pd:
            a = k1 & 1
            agg = agg_sw_idx(ps, a)
            hops.append(self._hop(sl, sh.leaf_port, 8 + a))
            hops.append(self._hop(agg, sl % LEAVES_PER_POD, dl % LEAVES_PER_POD))
            hops.append(self._hop(dl, 8 + a, dh.leaf_port))
        else:
            a_s = k1 & 1
            s_up = (k1 >> 1) % N_SPINES
            ss = (k2 >> 1) % N_SUPERSPINES
            s_down = (k2 >> 3) % N_SPINES
            a_d = k3 & 1
            agg_s = agg_sw_idx(ps, a_s)
            agg_d = agg_sw_idx(pd, a_d)
            hops.append(self._hop(sl, sh.leaf_port, 8 + a_s))
            hops.append(self._hop(agg_s, sl % LEAVES_PER_POD, 2 + s_up))
            hops.append(self._hop(spine_sw_idx(s_up), 2 * ps + a_s, 8 + ss))
            hops.append(self._hop(superspine_sw_idx(ss), s_up, s_down))
            # at the downstream spine, ingress is from the super-spine and
            # egress is toward the destination pod's agg
            hops.append(self._hop(spine_sw_idx(s_down), 8 + ss, 2 * pd + a_d))
            hops.append(self._hop(agg_d, 2 + s_down, dl % LEAVES_PER_POD))
            hops.append(self._hop(dl, 8 + a_d, dh.leaf_port))
        sig = "-".join(self.switches[h.sw].switch_id for h in hops)
        return Path(len(self._paths), sh.name, dh.name, tuple(hops), sig,
                    tuple(self.switches[h.sw].switch_id for h in hops))

    def _hop(self, sw: int, in_port: int, out_port: int) -> Hop:
        pk = self.ports[(sw, out_port)]
        link = self.port_link[pk]
        spec = self.links[link]
        return Hop(len(self._paths) * 0, sw, self.switches[sw].switch_id, in_port,
                   out_port, pk, link, spec.base_latency_us, spec.capacity_bpus,
                   self.port_cap_bytes[pk])

    # ------------------------------------------------------------------
    def topology_json(self) -> dict:
        pos = layout_positions(self.switches)
        seen = set()
        links = []
        for l in self.links:
            a, b = l.src, l.dst
            if a in self.hosts or b in self.hosts:
                continue
            ia, ib = self.sw_index(a), self.sw_index(b)
            key = (min(ia, ib), max(ia, ib))
            if key in seen:
                continue
            seen.add(key)
            links.append({"src": ia, "dst": ib,
                          "src_id": a, "dst_id": b,
                          "capacity_bps": l.capacity_bps})
        return {
            "switches": [{"id": s.switch_id, "kind": kind_of(s.switch_id),
                          "idx": i, "x": pos[s.switch_id][0], "y": pos[s.switch_id][1],
                          "n_ports": s.n_ports}
                         for i, s in enumerate(self.switches)],
            "links": links,
            "hosts": [{"name": h.name, "leaf": self.switches[h.leaf_idx].switch_id,
                       "port": h.leaf_port, "ip": h.ip}
                      for h in self.hosts.values()],
        }


def kind_of(sw_id: str) -> str:
    if sw_id.startswith("super"):
        return "super"
    if sw_id.startswith("spine"):
        return "spine"
    if sw_id.startswith("agg"):
        return "agg"
    return "leaf"


def layout_positions(switches: List[SwitchSpec]) -> Dict[str, Tuple[float, float]]:
    pos: Dict[str, Tuple[float, float]] = {}
    pod_cx = [190.0, 500.0, 810.0, 1120.0]
    for s in switches:
        sid = s.switch_id
        parts = sid.split("-")
        if parts[0] == "super":
            k = int(parts[2])
            pos[sid] = (500.0 + k * 300.0, 70.0)
        elif parts[0] == "spine":
            c = int(parts[1])
            pos[sid] = (380.0 + c * 230.0, 250.0)
        elif parts[0] == "agg":
            pod, a = int(parts[1]), int(parts[2])
            pos[sid] = (pod_cx[pod] + (a - 0.5) * 160.0, 430.0)
        else:
            pod, leaf = int(parts[1]), int(parts[2])
            pos[sid] = (pod_cx[pod] + (leaf - 0.5) * 160.0, 600.0)
    return pos


def build_fat_tree() -> Topology:
    switches: List[SwitchSpec] = []
    for pod in range(N_PODS):
        for leaf in range(LEAVES_PER_POD):
            switches.append(SwitchSpec(f"leaf-{pod}-{leaf}", HOSTS_PER_LEAF + 2))
    for pod in range(N_PODS):
        for a in range(AGGS_PER_POD):
            switches.append(SwitchSpec(f"agg-{pod}-{a}", 6))
    for s in range(N_SPINES):
        switches.append(SwitchSpec(f"spine-{s}", 8 + N_SUPERSPINES))
    for k in range(N_SUPERSPINES):
        switches.append(SwitchSpec(f"super-spine-{k}", N_SPINES))

    n_sw = len(switches)
    links: List[LinkSpec] = []
    ports: Dict[Tuple[int, int], int] = {}
    port_link: Dict[int, int] = {}
    port_rate = np.zeros(n_sw * MAX_PORTS)
    port_cap = np.zeros(n_sw * MAX_PORTS)

    def _port(sw: int, p: int, rate_bpus: float) -> int:
        assert p < MAX_PORTS, (sw, p)
        pk = sw * MAX_PORTS + p
        ports[(sw, p)] = pk
        port_rate[pk] = rate_bpus
        port_cap[pk] = max(128_000.0, rate_bpus * QUEUE_SLACK_US)
        return pk

    def add_bidir(sw_a: int, pa: int, sw_b: int, pb: int, bps: float, prop: float):
        bpus = bps_to_bpus(bps)
        fwd = LinkSpec(len(links), switches[sw_a].switch_id, switches[sw_b].switch_id, bps, prop)
        links.append(fwd)
        pk_a = _port(sw_a, pa, bpus)
        port_link[pk_a] = fwd.link_id
        rev = LinkSpec(len(links), switches[sw_b].switch_id, switches[sw_a].switch_id, bps, prop)
        links.append(rev)
        pk_b = _port(sw_b, pb, bpus)
        port_link[pk_b] = rev.link_id

    # leaf <-> agg
    for pod in range(N_PODS):
        for leaf in range(LEAVES_PER_POD):
            for a in range(AGGS_PER_POD):
                add_bidir(leaf_sw_idx(pod, leaf), 8 + a, agg_sw_idx(pod, a), leaf,
                          LEAF_UPLINK_BPS, LEAF_AGG_PROP_US)
    # agg <-> spine
    for pod in range(N_PODS):
        for a in range(AGGS_PER_POD):
            for s in range(N_SPINES):
                add_bidir(agg_sw_idx(pod, a), 2 + s, spine_sw_idx(s), 2 * pod + a,
                          AGG_UPLINK_BPS, AGG_SPINE_PROP_US)
    # spine <-> super-spine
    for s in range(N_SPINES):
        for k in range(N_SUPERSPINES):
            add_bidir(spine_sw_idx(s), 8 + k, superspine_sw_idx(k), s,
                      SPINE_UPLINK_BPS, SPINE_SUPER_PROP_US)

    hosts: Dict[str, HostInfo] = {}
    host_order: List[str] = []
    host_bpus = bps_to_bpus(HOST_BPS)
    for pod in range(N_PODS):
        for leaf in range(LEAVES_PER_POD):
            sw = leaf_sw_idx(pod, leaf)
            for h in range(HOSTS_PER_LEAF):
                name = f"host-{pod}-{leaf}-{h}"
                ip = f"10.{pod}.{leaf}.{h + 10}"
                hosts[name] = HostInfo(name, sw, h, ip)
                host_order.append(name)
                fwd = LinkSpec(len(links), switches[sw].switch_id, name, HOST_BPS, HOST_PROP_US)
                links.append(fwd)
                pk = _port(sw, h, host_bpus)
                port_link[pk] = fwd.link_id
                links.append(LinkSpec(len(links), name, switches[sw].switch_id,
                                      HOST_BPS, HOST_PROP_US))

    return Topology(switches=switches, links=links, ports=ports,
                    port_link=port_link, port_rate=port_rate,
                    port_cap_bytes=port_cap, hosts=hosts, host_order=host_order)
