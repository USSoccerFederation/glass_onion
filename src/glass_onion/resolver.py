from collections.abc import Hashable
from typing import Any

Vertex = tuple[str, Hashable]
"""
A `(provider ID field, ID)` pair, where the field is named `{provider}_{object_type}_id`,
e.g. `("provider_a_object_id", "12345")`.
"""

Record = dict[str, Any]
"""
One source row: `{"source": str, "ids": {provider ID field: ID}, "meta": {...}}`.
`ids` holds the IDs this row syncs together; `source` and `meta` are carried along untouched.
"""


def vertices(record: Record) -> list[Vertex]:
    """
    Returns the `(provider ID field, ID)` vertices for a record.
    """
    return list(record["ids"].items())


class ObjectResolver:
    """
    Union-find over `(provider ID field, ID)` vertices, tracking each component's IDs and the records that built it.

    Every `(provider, ID)` pair is a vertex, and a sync between two IDs is an edge. Each connected component is one object,
    and may hold at most one ID per provider. This class does not enforce that constraint itself: callers must check for
    provider conflicts before calling `add_record` (see `resolve`).
    """

    def __init__(self) -> None:
        self.parent: dict[Vertex, Vertex] = {}
        self.ids: dict[Vertex, dict[str, Hashable]] = {}  # root -> {provider: ID}
        self.records: dict[Vertex, list[Record]] = {}  # root -> [record]

    def __contains__(self, v: Vertex) -> bool:
        return v in self.parent

    def find(self, v: Vertex) -> Vertex:
        """
        Returns the root vertex of `v`'s component, compressing the path along the way.
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

        Callers must check for provider conflicts first: this method will happily merge two IDs from the same provider.
        """
        vs = vertices(record)
        for v in vs:
            if v not in self.parent:
                self.parent[v] = v
                self.ids[v] = {v[0]: v[1]}
                self.records[v] = []
        root = self.find(vs[0])
        for v in vs[1:]:
            other = self.find(v)
            if other != root:
                self.parent[other] = root
                self.ids[root].update(self.ids.pop(other))
                self.records[root].extend(self.records.pop(other))
        self.records[root].append(record)

    def components(self) -> list[tuple[dict[str, Hashable], list[Record]]]:
        """
        Returns one `(IDs, records)` pair per connected component (i.e. per object).
        """
        return [(self.ids[root], self.records[root]) for root in self.ids]


def resolve(
    resolver: ObjectResolver, records: list[Record]
) -> tuple[list[Record], list[tuple[Record, list[Record]]]]:
    """
    Merges multi-ID records into `resolver`, each as a single unit.

    A record is rejected if merging it would put two IDs from one provider into the same component, either with the
    resolver as it stands or together with the other records. Records that only conflict with each other are all rejected,
    so the result doesn't depend on their order.

    Returns `(accepted, rejected)`, where `rejected` holds `(record, resolver records it clashed with)` pairs.
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
    for r in records:
        nodes = {node_of(v) for v in vertices(r)}
        bad = conflicted(nodes)
        if len(bad) > 0:
            rejected.append((r, clashes(bad)))
        else:
            candidates.append((r, nodes))

    # conflicts between candidates: drop every candidate touching a conflicted node, then re-check what's left
    while True:
        parent: dict[Vertex, Vertex] = {}

        def find(n: Vertex, parent: dict[Vertex, Vertex] = parent) -> Vertex:
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        for _, nodes in candidates:
            ordered = list(nodes)
            for n in ordered:
                parent.setdefault(n, n)
            for n in ordered[1:]:
                parent[find(n)] = find(ordered[0])

        members: dict[Vertex, set[Vertex]] = {}
        for n in parent:
            members.setdefault(find(n), set()).add(n)
        bad = set().union(*[conflicted(ms) for ms in members.values()])
        if len(bad) == 0:
            break

        remaining = []
        for r, nodes in candidates:
            if len(nodes & bad) > 0:
                rejected.append((r, clashes(nodes & bad)))
            else:
                remaining.append((r, nodes))
        candidates = remaining

    for r, _ in candidates:
        resolver.add_record(r)
    return [r for r, _ in candidates], rejected
