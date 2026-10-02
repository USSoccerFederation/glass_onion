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

PathVertex = tuple[Literal["n"], Vertex] | tuple[Literal["r"], int]
"""
A vertex in [resolve()][glass_onion.resolver.ObjectResolver.resolve]'s graph of candidates: `("n", node)` for a resolver root or new vertex, `("r", k)` for the `k`th candidate.
"""

BlockCutVertex = tuple[Literal["b"], int] | tuple[Literal["c"], PathVertex]
"""
A vertex in [conflict_paths()][glass_onion.resolver.ObjectResolver.conflict_paths]'s block-cut tree: `("b", i)` for the `i`th biconnected block, `("c", v)` for cut vertex `v`.
"""


def normalize_id(i: Hashable) -> Hashable | None:
    """
    Normalizes a provider ID into the form used for [Vertex][glass_onion.resolver.Vertex] objects.

    Missing values (`None`, NaN, `pd.NA`, `pd.NaT`), blank strings, and the strings "<na>" and "null" (in any casing) are invalid. Other strings are valid, including `"0"`. Numeric IDs are converted to integers and then to strings, so `123`, `123.0` and `"123"` are the same ID. Numbers with a fractional part are converted to strings as is.

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


def to_frame(
    records: list[Record], conflicts: list[list[Record]] | None = None
) -> pd.DataFrame:
    """
    Flattens records into a DataFrame shaped like `SyncEngine.synchronize()` output.

    Args:
        records (list[glass_onion.resolver.Record]): the records to flatten.
        conflicts (list[list[glass_onion.resolver.Record]] | None): if given, the records each of `records` conflicted with, added as a `conflicts` column.

    Returns:
        A DataFrame with one row per record: a `source` column, one column per provider ID field in any record (IDs normalized, invalid or missing IDs `None`; see [normalize_id()][glass_onion.resolver.normalize_id]), then the record's `meta` keys, then `conflicts` if given.
    """
    fields = list(dict.fromkeys(p for r in records for p in r["ids"]))
    frame = pd.DataFrame(
        [
            {
                "source": r["source"],
                **{p: normalize_id(r["ids"].get(p)) for p in fields},
                **r["meta"],
            }
            for r in records
        ],
        columns=None if len(records) > 0 else ["source"],
    )
    if conflicts is not None:
        frame["conflicts"] = pd.Series(conflicts, index=frame.index, dtype=object)
    return frame


class ObjectResolver:
    """
    Applies a graph-based approach to resolving object identifiers from multiple runs of [SyncEngine.synchronize()][glass_onion.engine.SyncEngine.synchronize] to properly identify related identifiers and discard conflicts.

    Within the graph, every `(provider, ID)` pair is a vertex, and a sync between two IDs is an edge. `ObjectResolver` builds graph "components" out of these vertices to represent objects. By design, these components may hold AT MOST one ID per provider.

    More methodology details are available in [ObjectResolver.resolve()][glass_onion.resolver.ObjectResolver.resolve].
    """

    def __init__(self):
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
        The records that built each component, keyed by the component's root. A record whose IDs are all in another kept record is not kept.
        """
        self.kept: dict[frozenset[Vertex], Record] = {}
        """
        Every record in `records`, keyed by its vertex set.
        """
        self.covers: dict[Vertex, set[frozenset[Vertex]]] = {}
        """
        For each vertex, the vertex sets of the records in `kept` that hold it, used to find records a new record is inside.
        """
        self.anchored: dict[Vertex, set[frozenset[Vertex]]] = {}
        """
        The vertex sets in `kept`, each under the one of its vertices held by the fewest kept records when it was added. Used to find records inside a new record without scanning every record that shares a common vertex.
        """
        self.added: pd.DataFrame = to_frame([])
        """
        The records the latest [resolve()][glass_onion.resolver.ObjectResolver.resolve] call added to the resolver, flattened by [to_frame()][glass_onion.resolver.to_frame].
        """
        self.rejected: pd.DataFrame = to_frame([], [])
        """
        The records the latest [resolve()][glass_onion.resolver.ObjectResolver.resolve] call rejected, flattened by [to_frame()][glass_onion.resolver.to_frame]. Each row's `conflicts` holds the records it conflicted with: every resolver record involved in the conflict, followed by any other records rejected in the same conflict.
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

    def add_record(self, record: Record) -> bool:
        """
        Adds a record's vertices to the resolver and merges them into a single component.

        Only the largest of a set of nested records is kept, whatever order they arrive in: a record whose vertices are all in one kept record (an exact repeat, or a 2-ID record inside a 3-ID record) is skipped, and kept records whose vertices are all in the new record are removed. A smaller record adds no IDs or links the larger one doesn't, and keeping it would grow `records` with every repeated sync. Of two exact repeats, the earlier is kept. A record linking IDs from several kept records is not a repeat, even if they're all in one component. Smaller components are merged into the largest one, so each merge copies only the smaller side's IDs and records.

        Callers must check for provider conflicts first: this method will happily merge two IDs from the same provider. [resolve()][glass_onion.resolver.ObjectResolver.resolve] does this check.

        Args:
            record (glass_onion.resolver.Record): the record to add. It must have at least one valid ID.

        Raises:
            AssertionError: if `record` has no valid IDs.

        Returns:
            `True` if the record was added, `False` if it was skipped as a repeat.
        """
        vs = vertices(record)
        assert len(vs) > 0, f"Record has no valid IDs: {record['ids']}"

        key = frozenset(vs)
        # any record covering `key` holds every vertex in it, so checking the vertex in the fewest records is enough
        fewest = min((self.covers.get(v, set()) for v in key), key=len)
        if any(key <= k for k in fewest):
            return False
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
        # a record inside this one is anchored at one of its vertices, and holds only its vertices, so it's in `root` by now
        inside = {k for v in key for k in self.anchored.get(v, set()) if k < key}
        if len(inside) > 0:
            removed = {id(self.kept.pop(k)) for k in inside}
            for k in inside:
                for v in k:
                    self.covers[v].discard(k)
                    self.anchored.get(v, set()).discard(k)
            self.records[root] = [r for r in self.records[root] if id(r) not in removed]
        self.records[root].append(record)
        self.kept[key] = record
        anchor = min(key, key=lambda v: len(self.covers.get(v, set())))
        self.anchored.setdefault(anchor, set()).add(key)
        for v in key:
            self.covers.setdefault(v, set()).add(key)
        return True

    def components(self) -> list[tuple[dict[str, Hashable], list[Record]]]:
        """
        Lists the resolver's connected components (IE: its objects).

        Returns:
            One `(IDs, records)` pair per component, where `IDs` is `{provider ID field: ID}` and `records` holds the records that built the component.
        """
        return [(self.ids[root], self.records[root]) for root in self.ids]

    def conflict_paths(
        self, adj: dict[PathVertex, set[PathVertex]], terminals: set[PathVertex]
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
            ("c", t) if len(blocks_of[t]) > 1 else ("b", blocks_of[t][0])
            for t in terminals
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

    def resolve(self, records: list[Record]):
        """
        Merges multi-ID records into the resolver, each as a single unit, and sets [added][glass_onion.resolver.ObjectResolver.added] and [rejected][glass_onion.resolver.ObjectResolver.rejected] to this call's results.

        A record is rejected if merging it would put two IDs from one provider into the same component. The result doesn't depend on the order of `records`.

        Methodology:
            1. Drop records with no valid IDs (see [normalize_id()][glass_onion.resolver.normalize_id]). These appear in neither `added` nor `rejected`.
            2. Reject each record that conflicts with the resolver as it stands.
            3. Of the remaining records, reject those on a path linking two IDs from one provider (see [conflict_paths()][glass_onion.resolver.ObjectResolver.conflict_paths]). Records that merely touch the same object are kept, and records that only conflict with each other are all rejected.
            4. Add the remaining records to the resolver, largest first. Records whose IDs are all in one record from the resolver or this batch are skipped and appear in neither `added` nor `rejected`; resolver records whose IDs are all in an added record are removed from the resolver (see [add_record()][glass_onion.resolver.ObjectResolver.add_record]).

        Args:
            records (list[glass_onion.resolver.Record]): the records to merge.
        """

        def node_of(v: Vertex) -> Vertex:
            return self.find(v) if v in self else v

        def node_ids(n: Vertex) -> dict[str, Hashable]:
            return self.ids[n] if n in self else {n[0]: n[1]}

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

        def conflicts(nodes: set[Vertex]) -> list[Record]:
            return [r for n in nodes if n in self for r in self.records[n]]

        # conflicts with the resolver as it stands
        rejected: list[tuple[Record, list[Record]]] = []
        candidates: list[tuple[Record, set[Vertex]]] = []
        for r in [r for r in records if len(vertices(r)) > 0]:
            nodes = {node_of(v) for v in vertices(r)}
            bad = conflicted(nodes)
            if len(bad) > 0:
                rejected.append((r, conflicts(bad)))
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
                *[
                    self.conflict_paths(adj, ts)
                    for ts in by_provider.values()
                    if len(ts) > 1
                ]
            )
            if len(on_paths) > 0:
                dropped.append(
                    (
                        {g[1] for g in on_paths if g[0] == "r"},
                        {g[1] for g in on_paths if g[0] == "n" and g[1] in self},
                    )
                )

        # candidates rejected in one component share one conflict: each conflicts with every resolver record on its paths, then
        # with the other candidates rejected alongside it
        for ks, resolver_nodes in dropped:
            involved = conflicts(resolver_nodes)
            for k in sorted(ks):
                peers = [candidates[j][0] for j in sorted(ks) if j != k]
                rejected.append((candidates[k][0], involved + peers))
        rejected_ks = set().union(*[ks for ks, _ in dropped])
        candidates = [c for k, c in enumerate(candidates) if k not in rejected_ks]

        # largest first, so a record inside a larger one in this batch is skipped whatever the order
        stored: set[int] = set()
        for k in sorted(
            range(len(candidates)), key=lambda k: -len(vertices(candidates[k][0]))
        ):
            if self.add_record(candidates[k][0]):
                stored.add(k)
        self.added = to_frame([r for k, (r, _) in enumerate(candidates) if k in stored])
        self.rejected = to_frame([r for r, _ in rejected], [c for _, c in rejected])
