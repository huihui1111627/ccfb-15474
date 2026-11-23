"""In-band network telemetry testbed for a data-centre fabric.

Modules
-------
config      constants and configuration dataclasses
topology    3-tier fat-tree, ECMP path resolution
traffic     request/response and bulk traffic model
sim         data plane, clock skew, faults, INT marking, wire bytes
sampling    fixed / stratified / adaptive policies, budget optimiser
collector   path recovery, clock fit, per-hop decomposition, attribution
evaluate    acceptance comparison and fault-free control
server      JSON API and the dashboard host
"""

__version__ = "1.0.0"
