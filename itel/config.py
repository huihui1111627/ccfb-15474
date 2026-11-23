"""Shared configuration objects for the INT telemetry testbed.

Everything is expressed in SI-ish units used throughout the project:

* time      -> microseconds (float64)
* bytes     -> bytes (float64)
* rates     -> bytes / microsecond  (1 B/us == 8 Mbps)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Wire constants
# --------------------------------------------------------------------------
BYTES_PER_US = 1.0                     # B/us  (8 Mbps per B/us)
NS_PER_US = 1_000.0

#: Maximum number of bytes an INT "hop record" costs on the wire.
INT_HOP_BYTES = 12

#: Default metadata budget in bytes.  A 24-byte base header (5-tuple, seq,
#: source timestamp) plus four 12-byte hop records.  A 7-hop
#: leaf-agg-spine-super-spine-agg-leaf path therefore has to drop three hops,
#: which is exactly the pressure real INT deployments face.
DEFAULT_INT_BUDGET_BYTES = 72

#: Reference service / utilization thresholds
SLOW_REQUEST_US = 5_000.0              # >= 5 ms end-to-end request is "slow"
SLOW_TXN_US = SLOW_REQUEST_US


@dataclass(frozen=True)
class LinkSpec:
    """A directed physical link.  Undirected topology edges become two links."""

    link_id: int
    src: str
    dst: str
    capacity_bps: float
    base_latency_us: float
    #: probability a packet on this link is corrupted and needs retransmit
    ber_retx_prob: float = 0.0
    #: extra latency incurred per retransmission (serialisation of the frame)
    retx_penalty_us: float = 0.0

    @property
    def capacity_bpus(self) -> float:
        return self.capacity_bps / 8.0 / 1e6


@dataclass
class SwitchSpec:
    switch_id: str
    n_ports: int
    #: per-output-port queue capacity in bytes
    queue_bytes: float = 512_000.0
    #: per-port drain rate in bytes/us (set from the incoming/outgoing link)
    #: queue propagation for INT metadata (hop records) -- metadata is not
    #: queued, it rides in the packet header.
    clock_ppb: float = 0.0             # fixed offset vs real time (ppm scale)
    clock_drift_ppb: float = 0.0       # linear drift over the run


@dataclass(frozen=True)
class FlowClass:
    """A business traffic class.

    ``weight`` is the share of *new flow* arrivals; ``rate_bpus`` and
    ``size_bytes`` define the flow itself.
    """

    name: str
    weight: float
    size_bytes: float
    rate_bpus: float                   # mean injection rate of the flow
    burstiness: float = 0.6            # 0 = CBR, 1 = heavy burst
    #: tenancy / importance used by sampling policy
    importance: float = 1.0
    #: whether the class is request/response (paired, latency measured)
    request_response: bool = False
    #: src -> dst endpoints pool
    src_pool: Tuple[str, ...] = ()
    dst_pool: Tuple[str, ...] = ()


@dataclass
class TrafficConfig:
    classes: Tuple[FlowClass, ...]
    duration_us: float = 4_000_000.0      # 4 s of traffic
    flows_per_second: float = 2_000.0
    flow_duration_us: Tuple[float, float] = (30_000.0, 400_000.0)
    seed: int = 7


@dataclass
class TelemetryConfig:
    budget_bytes: int = DEFAULT_INT_BUDGET_BYTES
    hop_bytes: int = INT_HOP_BYTES
    #: baseline random sampling probability at the source
    base_sample_prob: float = 0.02
    #: per-flow-class sampling multiplier, keyed by FlowClass.name
    class_multiplier: Dict[str, float] = field(default_factory=dict)
    #: per-path sampling multiplier (path signature key)
    path_multiplier: Dict[str, float] = field(default_factory=dict)
    #: backlog drain-time (us) at an egress port that fires a full-marking
    #: trigger.  Expressing the threshold in *drain time* rather than bytes
    #: makes it independent of link speed, which is how real hardware does it.
    #: 3 ms is roughly "a queue that will add more than one RTT of delay".
    queue_trigger_us: float = 3_000.0
    #: hop residence (us) that fires a full-marking trigger
    hop_latency_trigger_us: float = 3_000.0
    #: how long a switch keeps full-marking after a trigger.  Real hardware
    #: uses a short timer: long enough to catch the packets that will queue
    #: behind the burst (they arrive within one queue-drain time), short
    #: enough not to flood the collector with an entire incident.  100 us
    #: catches every affected request in the reference scenario.
    trigger_window_us: float = 100.0
    #: truncation policy: 'tail' | 'head' | 'key_hops'
    truncation_policy: str = "key_hops"
    #: number of steady-state (non-triggered) sampled packets headroom
    sample_burst: int = 1

    def max_hops(self) -> int:
        return max(1, self.budget_bytes // self.hop_bytes)


@dataclass
class ScenarioConfig:
    """Fault injection used for acceptance tests."""

    #: persistently rate-limited (mis-shaped) ports: (switch, out_port, factor).
    #: This is a *configuration* fault, so the drain rate is low for the whole
    #: run and the collector can learn it -- unlike episodic oversubscription,
    #: which changes the offered load, not the port's rate.
    shapers: list = field(default_factory=list)
    #: episodic *rate* changes, e.g. a port that degrades under temperature:
    #: (switch, out_port, t_start, t_end, factor)
    congestion: list = field(default_factory=list)
    #: link impairment episodes: (link_id, t_start_us, t_end_us, retx_prob_booster)
    link_impair: list = field(default_factory=list)
    #: switch clock biases in *ppb* (part per billion of real time)
    clock_bias_ppb: Dict[str, float] = field(default_factory=dict)


@dataclass
class RunConfig:
    traffic: TrafficConfig
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    topology: object = None            # resolved lazily by sim.topology
