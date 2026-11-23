"""Source-side sampling policies and the data-volume budget optimiser.

Four policies:

``FixedSampling``      every packet sampled independently with probability p.
``StratifiedSampling`` base probability multiplied per flow class and per path.
``AdaptiveSampling``   stratified + per-class/path weights chosen by the budget
                       optimiser, plus switch-triggered full marking (the
                       trigger itself lives in :mod:`itel.sim`).

All policies are **causal**: the per-packet probability uses only the flow's
5-tuple, its class and a static per-path weight that an operator would derive
from a previous calibration window.  Nothing peeks at the current run.

Volume accounting
-----------------
The wire cost of one sampled packet is ``INT_BASE_BYTES`` plus 12 bytes per
hop record, capped by the truncation budget.  The collector adds an 8-byte
index entry per sampled packet.  Both are charged, so comparing policies at
"the same data volume" compares like with like.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import INT_HOP_BYTES, TelemetryConfig
from .sim import INT_BASE_BYTES, DataPlane, MarkingResult, mark_packets
from .config import SLOW_TXN_US
from .traffic import Traffic

#: per-packet collector index overhead, amortised
COLLECTOR_RECORD_OVERHEAD = 8


# ---------------------------------------------------------------------------
def metadata_bytes(hop_counts: np.ndarray, tcfg: TelemetryConfig) -> np.ndarray:
    """Bytes a packet's INT header occupies, respecting the truncation cap."""
    max_recs = max(1, (tcfg.budget_bytes - INT_BASE_BYTES) // tcfg.hop_bytes)
    recs = np.minimum(hop_counts, max_recs)
    return INT_BASE_BYTES + recs * tcfg.hop_bytes


def estimate_volume(marked: np.ndarray, hop_counts: np.ndarray,
                    tcfg: TelemetryConfig, duration_us: float,
                    record_overhead: int = COLLECTOR_RECORD_OVERHEAD,
                    records: Optional[int] = None) -> dict:
    """Collector-side bytes and rate for a given marking mask."""
    n = int(marked.sum())
    if n == 0:
        return {"packets": 0, "metadata_bytes": 0, "collector_bytes": 0,
                "mbit": 0.0, "mbps": 0.0, "bytes_per_s": 0.0,
                "header_bytes": 0, "record_bytes": 0, "index_bytes": 0,
                "records": 0}
    max_recs = max(1, (tcfg.budget_bytes - INT_BASE_BYTES) // tcfg.hop_bytes)
    n_rec = (int(records) if records is not None
             else int(np.minimum(hop_counts[marked], max_recs).sum()))
    head = n * INT_BASE_BYTES
    rec = n_rec * tcfg.hop_bytes
    idx = n * record_overhead
    coll = head + rec + idx
    secs = max(duration_us / 1e6, 1e-9)
    return {"packets": n, "records": n_rec,
            "metadata_bytes": head + rec, "collector_bytes": coll,
            "header_bytes": head, "record_bytes": rec, "index_bytes": idx,
            "mbit": coll * 8 / 1e6, "mbps": coll * 8 / 1e6 / secs,
            "bytes_per_s": coll / secs,
            "header_bytes_per_s": head / secs,
            "record_bytes_per_s": rec / secs,
            "index_bytes_per_s": idx / secs}


# ---------------------------------------------------------------------------
@dataclass
class PolicyResult:
    name: str
    prob: np.ndarray
    reason: np.ndarray
    marking: MarkingResult
    volume: dict
    params: dict = field(default_factory=dict)


class Policy:
    name = "base"
    use_trigger = True

    def probabilities(self, dp: DataPlane, tcfg: TelemetryConfig
                      ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def apply(self, dp: DataPlane, tcfg: TelemetryConfig,
              duration_us: Optional[float] = None) -> PolicyResult:
        prob, reason = self.probabilities(dp, tcfg)
        m = mark_packets(dp, tcfg, prob, reason, use_trigger=self.use_trigger)
        dur = (duration_us if duration_us is not None
               else float(dp.traffic.t_send.max()))
        vol = estimate_volume(m.marked, dp.n_hops, tcfg, dur)
        return PolicyResult(self.name, prob, reason, m, vol, dict(self.params()))

    def params(self) -> dict:
        return {}


class FixedSampling(Policy):
    """Uniform, independent per-packet sampling; no switch triggers."""

    use_trigger = False

    def __init__(self, p: float, name: Optional[str] = None):
        self.p = float(p)
        self.name = name or f"fixed {p:.2%}"

    def probabilities(self, dp, tcfg):
        n = dp.traffic.n_packets
        return np.full(n, self.p), np.zeros(n, dtype=np.int8)

    def params(self):
        return {"base_prob": self.p}


class StratifiedSampling(Policy):
    """Class and path multipliers, no adaptive triggers."""

    use_trigger = False

    def __init__(self, base: float, class_mult: Dict[str, float],
                 path_mult: Dict[str, float], name: str = "stratified"):
        self.base = float(base)
        self.class_mult = dict(class_mult)
        self.path_mult = dict(path_mult)
        self.name = name

    def probabilities(self, dp: DataPlane, tcfg: TelemetryConfig):
        tr = dp.traffic
        n = tr.n_packets
        flow_cls = np.empty(len(tr.flows), dtype=object)
        for f in tr.flows:
            flow_cls[f.flow_id] = f.cls
        cls_names = sorted({f.cls for f in tr.flows})
        cls_idx = {c: i for i, c in enumerate(cls_names)}
        cls_mult = np.array([self.class_mult.get(c, 1.0) for c in cls_names])
        per_pkt_cls = cls_mult[np.array([cls_idx[flow_cls[f]] for f in tr.flow_id])]
        prob = self.base * per_pkt_cls
        reason = np.where(per_pkt_cls != 1.0, 1, 0).astype(np.int8)
        sigs = {p.signature: p.path_id for p in dp.topo._paths}
        pm = np.ones(len(dp.topo._paths))
        for sig, mult in self.path_mult.items():
            if sig in sigs:
                pm[sigs[sig]] = mult
        prob = prob * pm[tr.path_id]
        reason[(pm[tr.path_id] != 1.0) & (reason == 0)] = 2
        return np.clip(prob, 0.0, 1.0), reason

    def params(self):
        return {"base_prob": self.base, "class_mult": self.class_mult,
                "path_mult": self.path_mult}


class AdaptiveSampling(Policy):
    """Per-(class, path) sampling from the optimiser + switch triggers.

    The probability for a packet is ``max(base_prob, stratum[(cls, path)])``.
    The policy is expressed as a set of match-action rules a switch can hold:

        match class=web  path=leaf-0-0→agg-0-1→...  ->  sample 1/12
        match class=*    path=*                     ->  sample 1/500

    (the base floor), which is exactly the shape of a production INT
    sampling configuration.
    """

    use_trigger = True

    def __init__(self, base: float, stratum_prob: Dict[Tuple[str, str], float],
                 name: str = "adaptive"):
        self.base = float(base)
        self.stratum_prob = {(c, p): float(v)
                             for (c, p), v in stratum_prob.items()}
        self.name = name

    def probabilities(self, dp: DataPlane, tcfg: TelemetryConfig):
        tr = dp.traffic
        flow_cls = {f.flow_id: f.cls for f in tr.flows}
        flow_path = {f.flow_id: f.path_id for f in tr.flows}
        sigs = {p.path_id: p.signature for p in dp.topo._paths}
        n = tr.n_packets
        prob = np.full(n, self.base)
        reason = np.zeros(n, dtype=np.int8)
        cache: Dict[Tuple[str, str], float] = {}
        for i in range(n):
            fid = int(tr.flow_id[i])
            key = (flow_cls[fid], sigs[flow_path[fid]])
            v = cache.get(key)
            if v is None:
                v = max(self.base, self.stratum_prob.get(key, self.base))
                cache[key] = v
            if v > self.base:
                prob[i] = v
                reason[i] = 1
        return np.clip(prob, 0.0, 1.0), reason

    def params(self):
        return {"base_prob": self.base, "n_rules": len(self.stratum_prob)}


# ---------------------------------------------------------------------------
# budget optimiser
# ---------------------------------------------------------------------------
@dataclass
class StratumStats:
    """Calibration statistics for one (class, path) sampling stratum."""

    cls: str
    path_sig: str
    path_id: int
    packets_per_s: float
    slow_txns_per_s: float
    mean_hops: float


def total_packet_rate(dp: DataPlane,
                      duration_us: Optional[float] = None) -> float:
    """All packets/s the sources offer, including bulk transfers.

    The source-side floor probability applies to every one of them, so the
    budget optimiser has to price the floor against this number, not against
    the response traffic alone.
    """
    tr = dp.traffic
    dur_s = max((duration_us if duration_us is not None
                 else float(tr.t_send.max())) / 1e6, 1e-9)
    return float(tr.n_packets) / dur_s


def calibration_strata(dp: DataPlane, duration_us: Optional[float] = None
                       ) -> List[StratumStats]:
    """Per-(class, response path) packet rate and slow-transaction rate.

    This is the calibration window an operator runs before turning on the
    adaptive policy: it needs *some* telemetry (or an existing coarse
    monitor) to learn which strata carry the slow requests.
    """
    tr = dp.traffic
    dur_s = max((duration_us if duration_us is not None
                 else float(tr.t_send.max())) / 1e6, 1e-9)
    flow_cls = {f.flow_id: f.cls for f in tr.flows}
    flow_path = {f.flow_id: f.path_id for f in tr.flows}
    slow = {tx.txn_id: tx.true_latency_us >= SLOW_TXN_US
            for tx in tr.txns if not np.isnan(tx.true_latency_us)}
    resp_txn = {}
    for r in np.nonzero(tr.role == 1)[0]:
        resp_txn[int(tr.flow_id[r])] = int(tr.txn_id[r])
    # count each (flow, transaction) pair once: a slow transaction's *whole*
    # response train sits in one stratum, so the stratum's slow rate is the
    # number of slow transactions it carries, not the number it carries times
    # its packet count
    counts: Dict[Tuple[str, int], List[float]] = {}
    seen: set = set()
    for i in np.nonzero(tr.role == 1)[0]:
        fid = int(tr.flow_id[i])
        key = (flow_cls[fid], flow_path[fid])
        c = counts.setdefault(key, [0.0, 0.0])
        c[0] += 1.0
        if (fid, key) in seen:
            continue
        seen.add((fid, key))
        if slow.get(resp_txn.get(fid, -1), False):
            c[1] += 1.0
    sigs = {p.path_id: p.signature for p in dp.topo._paths}
    out = []
    for (cls, pid), (npk, nslow) in counts.items():
        out.append(StratumStats(cls, sigs[pid], pid, npk / dur_s,
                                nslow / dur_s, float(dp.topo._paths[pid].n_hops)))
    out.sort(key=lambda s: -(s.slow_txns_per_s / max(s.packets_per_s, 1e-9)))
    return out


def estimate_trigger_volume(dp: DataPlane, tcfg: TelemetryConfig) -> float:
    """Bytes/s that switch-triggered full marking will add.

    This is measured, not modelled: evaluate the trigger policy with the
    steady-state sampling switched off, and count only packets the triggers
    marked.  An operator would run exactly this during a calibration window.
    """
    n = dp.traffic.n_packets
    prob = np.zeros(n)
    reason = np.zeros(n, dtype=np.int8)
    m = mark_packets(dp, tcfg, prob, reason, use_trigger=True)
    vol = estimate_volume(m.marked, dp.n_hops, tcfg, float(dp.traffic.t_send.max()))
    return float(vol["bytes_per_s"])


def optimize_budget(strata: Sequence[StratumStats], tcfg: TelemetryConfig,
                    budget_bytes_per_s: float, base_prob: float = 0.001,
                    max_mult: float = 60.0,
                    trigger_bytes_per_s: float = 0.0,
                    strata_pkt_rate: Optional[float] = None) -> dict:
    """Greedy water-filling of the sampling volume budget.

    For each (class, path) stratum the collector knows, from a calibration
    window, its packet rate and how many slow transactions it carries.  The
    expected capture per byte spent is ``P(slow | stratum) / bytes_per_pkt``,
    so the optimum is: give every stratum a small floor, then raise the
    multiplier of whichever stratum has the best capture-per-byte until the
    budget is gone or nothing more can be bought.  Any leftover budget raises
    the floor uniformly.

    Prices are computed against the *actual* packet rates each stratum sees,
    so the realised volume matches the budget by construction.
    """
    if not strata:
        return {"base_prob": base_prob, "stratum_prob": {},
                "expected_bytes_per_s": 0.0}
    steady_budget = max(0.0, budget_bytes_per_s - trigger_bytes_per_s)
    max_recs = max(1, (tcfg.budget_bytes - INT_BASE_BYTES) // tcfg.hop_bytes)
    per_pkt = np.array([INT_BASE_BYTES
                        + min(s.mean_hops, max_recs) * tcfg.hop_bytes
                        + COLLECTOR_RECORD_OVERHEAD for s in strata])
    npk = np.array([s.packets_per_s for s in strata], dtype=np.float64)
    slow = np.array([s.slow_txns_per_s for s in strata], dtype=np.float64)
    cost = per_pkt * npk                     # bytes/s per unit probability
    total_cost = float(cost.sum())
    # ``base_prob`` is applied to *every* packet the source sees, including
    # bulk transfers that never appear in a response stratum.  Charging the
    # floor against only the stratified traffic would under-count the real
    # volume by however much bulk traffic the fabric carries, so the floor is
    # priced against the whole packet rate.
    all_pkt_per_s = (float(strata_pkt_rate) if strata_pkt_rate is not None
                     else float(npk.sum()))
    # the floor applies to every packet, and most of that traffic is not in a
    # response stratum, so use the mean record count of an unstratified
    # packet: bulk flows are long-lived and take the same paths
    max_recs_ = max(1, (tcfg.budget_bytes - INT_BASE_BYTES) // tcfg.hop_bytes)
    mean_recs = min(max_recs_, int(np.median([s.mean_hops for s in strata])))
    floor_per_pkt = INT_BASE_BYTES + max(0, mean_recs) * tcfg.hop_bytes \
        + COLLECTOR_RECORD_OVERHEAD
    floor_cost = float(floor_per_pkt) * all_pkt_per_s

    def pack(prob: np.ndarray, bp: float) -> dict:
        sp = {(s.cls, s.path_sig): float(p)
              for s, p in zip(strata, prob) if p > bp * 1.001}
        return {"base_prob": float(bp), "stratum_prob": sp,
                "steady_bytes_per_s": float((prob * cost).sum()),
                "trigger_bytes_per_s": float(trigger_bytes_per_s),
                "expected_bytes_per_s": float((prob * cost).sum()
                                              + trigger_bytes_per_s),
                "expected_slow_capture_per_s": float((prob * slow).sum()),
                "n_rules": len(sp)}

    # the cost of sampling everything is the floor cost (all packets) plus the
    # extra record bytes that longer strata add
    extra = float(((per_pkt * npk) - floor_per_pkt * npk).sum())
    if steady_budget >= floor_cost + extra:
        return pack(np.ones(len(strata)), 1.0)
    if slow.sum() <= 0 or steady_budget <= 0:
        floor = min(1.0, max(0.0, steady_budget) / max(floor_cost, 1e-9))
        return pack(np.full(len(strata), floor), floor)
    # the floor must be affordable on its own before anything is spent on hot
    # strata, otherwise the plan overshoots before the first rule is written
    base_prob = min(base_prob, steady_budget / max(floor_cost, 1e-12))

    prob = np.clip(np.full(len(strata), base_prob), 0.0, 1.0)
    # spending on the floor is priced against all traffic; the per-stratum
    # probabilities above the floor are priced against their own strata
    spent = base_prob * floor_cost + float(
        ((prob - base_prob) * cost).sum())
    if spent > steady_budget:
        floor = min(1.0, steady_budget / max(floor_cost, 1e-9))
        return pack(np.full(len(strata), floor), floor)
    # A calibration window is short, so a stratum with zero observed slow
    # transactions is not a stratum with none; give quiet strata a small
    # pseudo-count so they keep a non-zero marginal value.
    slow_p = slow + 0.02 * np.sqrt(npk / max(npk.mean(), 1e-9))
    # Water-filling by *level*, not by stratum: at each level L a stratum
    # with value density above L buys probability up to L's cap.  This is the
    # standard solution for allocating a divisible budget against diminishing
    # returns, and it converges in a handful of passes.
    total = float(slow_p.sum())
    share = slow_p / max(total, 1e-9)
    cap = np.clip(4.0 * share, 0.01, 0.35)          # per-stratum ceiling
    marg = np.where(cost > 0, slow_p / np.maximum(cost, 1e-9), 0.0)
    order = np.argsort(-marg)
    for _ in range(400):
        remaining = steady_budget - spent
        if remaining <= 0:
            break
        moved = False
        for j in order:
            if remaining <= 0:
                break
            head = np.minimum(cap[j] - prob[j], 0.98 - prob[j])
            if head <= 1e-9:
                continue
            d = min(head, 0.02, remaining / max(cost[j], 1e-12))
            if d <= 1e-12:
                continue
            prob[j] += d
            spent += d * cost[j]
            remaining -= d * cost[j]
            moved = True
        if not moved:
            break
    # Spend anything left over by raising every stratum uniformly -- the
    # operator's "we still have budget, sample more of everything" move.
    # One unit of probability across all strata costs ``total_cost``.
    for _ in range(6):
        spent_now = float((prob * cost).sum()) + base_prob * floor_cost
        remaining = steady_budget - spent_now
        if remaining <= 0:
            break
        headroom = float((np.maximum(0.0, 1.0 - prob) * cost).sum())
        if headroom <= 1e-9:
            break
        delta = remaining / max(total_cost + floor_cost, 1e-12)
        if delta <= 1e-12:
            break
        prob = np.minimum(1.0, prob + delta)
        if delta < 1e-7:
            break
    bp = base_prob + max(0.0, steady_budget
                         - float((prob * cost).sum()) - base_prob * floor_cost) \
        / max(floor_cost, 1e-12)
    return pack(prob, max(base_prob, min(bp, 1.0)))


# ---------------------------------------------------------------------------
def capture_rate(dp: DataPlane, marked: np.ndarray,
                 slow_threshold_us: float = SLOW_TXN_US) -> dict:
    """Fraction of slow transactions that reached the collector at all.

    A transaction is captured when telemetry arrived for *any* packet of its
    response leg, which is the minimum needed to attribute it.
    """
    tr = dp.traffic
    resp_rows: Dict[int, List[int]] = {}
    for r in np.nonzero(tr.role == 1)[0]:
        resp_rows.setdefault(int(tr.txn_id[r]), []).append(int(r))
    n_slow = n_cap = 0
    for tx in tr.txns:
        if np.isnan(tx.true_latency_us) or tx.true_latency_us < slow_threshold_us:
            continue
        n_slow += 1
        rows = resp_rows.get(tx.txn_id, [])
        if rows and marked[np.asarray(rows, dtype=np.int64)].any():
            n_cap += 1
    return {"slow_txns": n_slow, "captured": n_cap,
            "capture_rate": (n_cap / n_slow) if n_slow else float("nan")}
