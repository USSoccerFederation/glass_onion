from collections.abc import Hashable, Iterator
from numbers import Real
from typing import Any, Literal, cast

import pandas as pd

Vertex = tuple[str, Hashable]
"""
A `(provider ID field, ID)` pair, where the field is named `{provider}_{object_type}_id`, e.g. `("provider_a_object_id", "12345")`.
"""

Record = dict[str, Any]
"""
One source row: `{"source": str, "ids": {provider ID field: ID}, "meta": {...}}`.

`ids` holds the IDs this row syncs together; `source` and `meta` are carried along untouched. IDs are normalized when read (see [normalize_id()][glass_onion.resolver.normalize_id]), so numeric IDs end up as strings and missing or blank IDs are ignored.
"""


def normalize_id(i: Hashable) -> Hashable | None:
    """
    Normalizes a provider ID into the form used for [Vertex][glass_onion.resolver.Vertex] objects.

    Missing values (`None`, NaN, `pd.NA`, `pd.NaT`), blank strings and the placeholder strings in [MISSING_ID_STRINGS][glass_onion.resolver.MISSING_ID_STRINGS] are invalid. Other strings are valid, including `"0"`. Numeric IDs are converted to integers and then to strings, so `123`, `123.0` and `"123"` are the same ID. Numbers with a fractional part are converted to strings as is.

    Args:
        i (collections.abc.Hashable): a provider ID, as found in a [Record][glass_onion.resolver.Record]'s `ids`.

    Returns:
        The normalized ID, or `None` if `i` isn't a valid ID.
    """
    if pd.api.types.is_scalar(i) and pd.isna(cast(Any, i)):
        return None
    if isinstance(i, str):
        return None if i.strip() == "" or i.strip().lower() in ["<na>", "null"] else i
    if isinstance(i, Real):
        return str(int(i)) if float(i).is_integer() else str(i)
    return i


def vertices(record: Record) -> list[Vertex]:
    """
    Builds the `(provider ID field, ID)` vertices for a record. IDs are normalized and invalid IDs dropped (see [normalize_id()][glass_onion.resolver.normalize_id]).

    Args:
        record (glass_onion.resolver.Record): the record to build vertices for.

    Returns:
        A list of [Vertex][glass_onion.resolver.Vertex] objects, one per valid ID in `record["ids"]`.
    """
    vs = [(p, normalize_id(i)) for p, i in record["ids"].items()]
    return [(p, i) for p, i in vs if i is not None]


PathVertex = tuple[Literal["n"], Vertex] | tuple[Literal["r"], int]
"""
A vertex in [resolve()][glass_onion.resolver.resolve]'s graph of candidates: `("n", node)` for a resolver root or new vertex, `("r", k)` for the `k`th candidate.
"""

BlockCutVertex = tuple[Literal["b"], int] | tuple[Literal["c"], PathVertex]
"""
A vertex in [conflict_paths()][glass_onion.resolver.conflict_paths]'s block-cut tree: `("b", i)` for the `i`th biconnected block, `("c", v)` for cut vertex `v`.
"""


def conflict_paths(
    adj: dict[PathVertex, set[PathVertex]], terminals: set[PathVertex]
) -> set[PathVertex]:
    """
    Finds every vertex of an undirected graph that lies on a simple path between two distinct terminal vertices.

    Methodology:
        1. Split the terminal vertices' connected component into biconnected blocks using an iterative version of Tarjan's algorithm.
        2. Build the block-cut tree of those blocks and their cut vertices.
        3. Prune leaves of the tree that hold no terminal vertex, until only the paths between terminal vertices remain. Every vertex of a remaining block lies on a simple path between two terminal vertices.

    Args:
        adj (dict[glass_onion.resolver.PathVertex, set[glass_onion.resolver.PathVertex]]): the graph's adjacency sets. The graph must have no parallel edges.
        terminals (set[glass_onion.resolver.PathVertex]): two or more terminal vertices of `adj`, all in one connected component.

    Returns:
        The set of [PathVertex][glass_onion.resolver.PathVertex] objects lying on a simple path between two of `terminals`.
    """
    # biconnected blocks (iterative Tarjan)
    start = next(iter(terminals))
    disc = {start: 0}
    low = {start: 0}
    stack = [start]
    blocks: list[set[PathVertex]] = []
    frames: list[tuple[PathVertex, PathVertex | None, Iterator[PathVertex]]] = [
        (start, None, iter(adj[start]))
    ]
    while frames:
        v, up, children = frames[-1]
        for w in children:
            if w not in disc:
                disc[w] = low[w] = len(disc)
                stack.append(w)
                frames.append((w, v, iter(adj[w])))
                break
            if w != up:
                low[v] = min(low[v], disc[w])
        else:
            frames.pop()
            if up is not None:
                low[up] = min(low[up], low[v])
                if low[v] >= disc[up]:
                    block = {up}
                    while True:
                        x = stack.pop()
                        block.add(x)
                        if x == v:
                            break
                    blocks.append(block)

    # block-cut tree: blocks are ("b", i), cut vertices ("c", v)
    blocks_of: dict[PathVertex, list[int]] = {}
    for i, b in enumerate(blocks):
        for v in b:
            blocks_of.setdefault(v, []).append(i)
    tree: dict[BlockCutVertex, set[BlockCutVertex]] = {
        ("b", i): set() for i in range(len(blocks))
    }
    for v, bs in blocks_of.items():
        if len(bs) > 1:
            tree[("c", v)] = {("b", i) for i in bs}
            for i in bs:
                tree[("b", i)].add(("c", v))
    pinned: set[BlockCutVertex] = {
        ("c", t) if len(blocks_of[t]) > 1 else ("b", blocks_of[t][0]) for t in terminals
    }

    # prune leaves holding no terminal vertex until only the paths between terminal vertices remain
    leaves = [t for t, ns in tree.items() if len(ns) <= 1 and t not in pinned]
    while leaves:
        t = leaves.pop()
        for u in tree.pop(t):
            tree[u].discard(t)
            if len(tree[u]) <= 1 and u not in pinned:
                leaves.append(u)
    return set().union(*[blocks[t[1]] for t in tree if t[0] == "b"])


class ObjectResolver:
    """
    A union-find over `(provider ID field, ID)` vertices, tracking each component's IDs and the records that built it.

    Every `(provider, ID)` pair is a vertex, and a sync between two IDs is an edge. Each connected component is one object, and may hold at most one ID per provider. This class does not enforce that constraint itself: callers must check for provider conflicts before calling [add_record()][glass_onion.resolver.ObjectResolver.add_record] (see [resolve()][glass_onion.resolver.resolve]).
    """

    def __init__(self) -> None:
        """
        Creates a new, empty ObjectResolver.
        """
        self.parent: dict[Vertex, Vertex] = {}
        """
        Each vertex's parent in the union-find. A component's root is its own parent.
        """
        self.ids: dict[Vertex, dict[str, Hashable]] = {}
        """
        Each component's IDs as `{provider ID field: ID}`, keyed by the component's root.
        """
        self.records: dict[Vertex, list[Record]] = {}
        """
        The records that built each component, keyed by the component's root. A record repeating an earlier record's IDs is not kept.
        """
        self.seen: set[frozenset[Vertex]] = set()
        """
        The vertex set of every record kept in `records`, used to skip repeats.
        """

    def __contains__(self, v: Vertex) -> bool:
        return v in self.parent

    def find(self, v: Vertex) -> Vertex:
        """
        Finds the root vertex of a vertex's component, compressing the path along the way.

        Args:
            v (glass_onion.resolver.Vertex): a vertex already in the resolver.

        Raises:
            KeyError: if `v` is not in the resolver.

        Returns:
            The root [Vertex][glass_onion.resolver.Vertex] of `v`'s component.
        """
        root = v
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[v] != root:
            self.parent[v], v = root, self.parent[v]
        return root

    def add_record(self, record: Record) -> None:
        """
        Adds a record's vertices to the resolver and merges them into a single component.

        A record whose vertices exactly match an earlier record's is skipped: it would add no IDs or links, and keeping it would grow `records` with every repeated sync. The earlier record is kept. Smaller components are merged into the largest one, so each merge copies only the smaller side's IDs and records.

        Callers must check for provider conflicts first: this method will happily merge two IDs from the same provider. [resolve()][glass_onion.resolver.resolve] does this check.

        Args:
            record (glass_onion.resolver.Record): the record to add. It must have at least one valid ID.

        Raises:
            AssertionError: if `record` has no valid IDs.
        """
        vs = vertices(record)
        assert len(vs) > 0, f"Record has no valid IDs: {record['ids']}"

        key = frozenset(vs)
        if key in self.seen:
            return
        for v in vs:
            if v not in self.parent:
                self.parent[v] = v
                self.ids[v] = {v[0]: v[1]}
                self.records[v] = []
        roots = list(dict.fromkeys(self.find(v) for v in vs))
        # ties keep the first vertex's root
        root = max(roots, key=lambda r: len(self.ids[r]) + len(self.records[r]))
        for other in roots:
            if other != root:
                self.parent[other] = root
                self.ids[root].update(self.ids.pop(other))
                self.records[root].extend(self.records.pop(other))
        self.records[root].append(record)
        self.seen.add(key)

    def components(self) -> list[tuple[dict[str, Hashable], list[Record]]]:
        """
        Lists the resolver's connected components (IE: its objects).

        Returns:
            One `(IDs, records)` pair per component, where `IDs` is `{provider ID field: ID}` and `records` holds the records that built the component.
        """
        return [(self.ids[root], self.records[root]) for root in self.ids]


def resolve(
    resolver: ObjectResolver, records: list[Record]
) -> tuple[list[Record], list[tuple[Record, list[Record]]]]:
    """
    Merges multi-ID records into an [ObjectResolver][glass_onion.resolver.ObjectResolver], each as a single unit.

    A record is rejected if merging it would put two IDs from one provider into the same component. The result doesn't depend on the order of `records`.

    Methodology:
        1. Drop records with no valid IDs (see [normalize_id()][glass_onion.resolver.normalize_id]). These do not appear in any downstream returned list.
        2. Reject each record that conflicts with the resolver as it stands.
        3. Of the remaining records, reject those on a path linking two IDs from one provider (see [conflict_paths()][glass_onion.resolver.conflict_paths]). Records that merely touch the same object are kept, and records that only conflict with each other are all rejected.
        4. Add the accepted records to the resolver.

    Args:
        resolver (glass_onion.resolver.ObjectResolver): the resolver to merge records into.
        records (list[glass_onion.resolver.Record]): the records to merge.

    Returns:
        A tuple of `(accepted, rejected)`, where `rejected` holds `(record, records it clashed with)` pairs. The clashing records are every resolver record involved in the conflict, followed by any other records in `records` rejected in the same conflict.
    """

    def node_of(v: Vertex) -> Vertex:
        return resolver.find(v) if v in resolver else v

    def node_ids(n: Vertex) -> dict[str, Hashable]:
        return resolver.ids[n] if n in resolver else {n[0]: n[1]}

    def conflicted(nodes: set[Vertex]) -> set[Vertex]:
        # nodes holding an ID for a provider that has more than one distinct ID across `nodes`
        by_provider: dict[str, dict[Hashable, set[Vertex]]] = {}
        for n in nodes:
            for p, i in node_ids(n).items():
                by_provider.setdefault(p, {}).setdefault(i, set()).add(n)
        return {
            n
            for by_id in by_provider.values()
            if len(by_id) > 1
            for ns in by_id.values()
            for n in ns
        }

    def clashes(nodes: set[Vertex]) -> list[Record]:
        return [r for n in nodes if n in resolver for r in resolver.records[n]]

    # conflicts with the resolver as it stands
    rejected: list[tuple[Record, list[Record]]] = []
    candidates: list[tuple[Record, set[Vertex]]] = []
    for r in [r for r in records if len(vertices(r)) > 0]:
        nodes = {node_of(v) for v in vertices(r)}
        bad = conflicted(nodes)
        if len(bad) > 0:
            rejected.append((r, clashes(bad)))
        else:
            candidates.append((r, nodes))

    # conflicts between candidates: in a bipartite graph of candidates and the nodes they link, reject only the candidates
    # on a path between two IDs from one provider. Removing those leaves no such path, so one pass is enough.
    adj: dict[PathVertex, set[PathVertex]] = {}
    for k, (_, nodes) in enumerate(candidates):
        adj[("r", k)] = {("n", n) for n in nodes}
        for n in nodes:
            adj.setdefault(("n", n), set()).add(("r", k))

    seen: set[PathVertex] = set()
    # (candidates, resolver nodes) per conflicted component
    dropped: list[tuple[set[int], set[Vertex]]] = []
    for g in adj:
        if g in seen:
            continue
        component = {g}
        queue = [g]
        while queue:
            for h in adj[queue.pop()] - component:
                component.add(h)
                queue.append(h)
        seen |= component

        by_provider: dict[str, set[PathVertex]] = {}
        for x in component:
            if x[0] == "n":
                for p in node_ids(x[1]):
                    by_provider.setdefault(p, set()).add(x)
        on_paths = set().union(
            *[conflict_paths(adj, ts) for ts in by_provider.values() if len(ts) > 1]
        )
        if len(on_paths) > 0:
            dropped.append(
                (
                    {g[1] for g in on_paths if g[0] == "r"},
                    {g[1] for g in on_paths if g[0] == "n" and g[1] in resolver},
                )
            )

    # candidates rejected in one component share one conflict: each clashes with every resolver record on its paths, then
    # with the other candidates rejected alongside it
    for ks, resolver_nodes in dropped:
        involved = clashes(resolver_nodes)
        for k in sorted(ks):
            peers = [candidates[j][0] for j in sorted(ks) if j != k]
            rejected.append((candidates[k][0], involved + peers))
    rejected_ks = set().union(*[ks for ks, _ in dropped])
    candidates = [c for k, c in enumerate(candidates) if k not in rejected_ks]

    for r, _ in candidates:
        resolver.add_record(r)
    return [r for r, _ in candidates], rejected
