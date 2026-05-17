"""Force SCIP to use depth-first-search node selection.

SCIP exposes a registry of node selectors; the one with the highest
``stdpriority`` wins. We bump the DFS selector's priority well above any
default value so it is chosen for every node selection decision.
"""
from __future__ import annotations

# Any priority larger than SCIP's built-in defaults (the largest stock value
# is ``estimate`` at 200000) suffices; we pick a very large value defensively.
_DFS_PRIORITY = 10_000_000

DFS_SCIP_PARAMS: dict[str, int] = {
    "nodeselection/dfs/stdpriority": _DFS_PRIORITY,
    "nodeselection/dfs/memsavepriority": _DFS_PRIORITY,
}
