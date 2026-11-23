"""Command-line entry points.

    python -m itel serve                 # dashboard + API
    python -m itel evaluate              # acceptance report (JSON)
    python -m itel evaluate --seeds 7 11 23
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="itel",
                                description="In-band telemetry testbed for a DC fabric")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--budget", type=float, default=0.8,
                   help="collector budget in MB/s")

    e = sub.add_parser("evaluate", help="run the acceptance comparison")
    e.add_argument("--seeds", type=int, nargs="+", default=None)
    e.add_argument("--budget", type=float, nargs="+", default=None,
                   help="budgets in MB/s")
    e.add_argument("--no-collector", action="store_true",
                   help="skip the collector phase (faster)")
    e.add_argument("--out", default=None, help="write JSON here")
    e.add_argument("--quiet", action="store_true")
    e.add_argument("--with-control", action="store_true",
                   help="also run a fault-free control for estimator bias")

    a = p.parse_args(argv)
    if a.cmd == "serve":
        from .server import serve
        serve(a.host, a.port, a.seed, a.budget * 1e6)
        return 0
    if a.cmd == "evaluate":
        from .evaluate import run_acceptance, run_scenarios
        if a.seeds:
            rep = run_scenarios(a.seeds,
                                [b * 1e6 for b in (a.budget or [1.0])])
        else:
            budgets = [b * 1e6 for b in (a.budget or [0.4, 0.8, 2.0])]
            rep = run_acceptance(budgets_bytes_per_s=budgets,
                                 include_collector=not a.no_collector)
            if a.with_control:
                from .evaluate import run_control
                rep["control"] = run_control()
        text = json.dumps(rep, indent=2, default=str)
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(text)
        if not a.quiet:
            print(text)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
