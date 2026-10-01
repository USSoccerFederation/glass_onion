from glass_onion.resolver import ObjectResolver, resolve


def record(source: str, name: str, object_type: str = "player", **ids) -> dict:
    return {
        "source": source,
        "ids": {f"{provider}_{object_type}_id": i for provider, i in ids.items()},
        "meta": {f"{object_type}_name": name},
    }


def component_ids(resolver: ObjectResolver) -> set[frozenset]:
    # order-insensitive view of each object's IDs
    return {frozenset(ids.items()) for ids, _ in resolver.components()}


def objects(*ids: dict) -> set[frozenset]:
    return {frozenset(i.items()) for i in ids}


def test_resolve_happy_path():
    # synced players, plus proposals that extend them or describe new players
    resolver = ObjectResolver()
    truth = [
        record("player", "Alex Morgan", provider_a="101", provider_b="201"),
        record("player", "Megan Rapinoe", provider_a="102", provider_b="202"),
    ]
    proposals = [
        # extends Alex Morgan with a provider_c ID
        record("raw", "A. Morgan", provider_b="201", provider_c="301"),
        # extends Megan Rapinoe with a provider_d ID
        record("raw", "M. Rapinoe", provider_a="102", provider_d="402"),
        # a new player linked across two providers
        record("raw", "Rose Lavelle", provider_a="103", provider_c="303"),
    ]

    accepted, rejected = resolve(resolver, truth)
    assert accepted == truth
    assert rejected == []

    accepted, rejected = resolve(resolver, proposals)
    assert accepted == proposals
    assert rejected == []

    assert component_ids(resolver) == objects(
        {
            "provider_a_player_id": "101",
            "provider_b_player_id": "201",
            "provider_c_player_id": "301",
        },
        {
            "provider_a_player_id": "102",
            "provider_b_player_id": "202",
            "provider_d_player_id": "402",
        },
        {"provider_a_player_id": "103", "provider_c_player_id": "303"},
    )


def test_resolve_rejects_conflict_with_existing_object():
    # a proposal that would give an existing player a second provider_b ID
    resolver = ObjectResolver()
    truth = record("player", "Alex Morgan", provider_a="101", provider_b="201")
    resolve(resolver, [truth])

    conflicting = record("raw", "Alex Morgan", provider_a="101", provider_b="999")
    clean = record("raw", "Alex Morgan", provider_b="201", provider_c="301")

    accepted, rejected = resolve(resolver, [conflicting, clean])

    assert accepted == [clean]
    assert rejected == [(conflicting, [truth])]
    assert component_ids(resolver) == objects(
        {
            "provider_a_player_id": "101",
            "provider_b_player_id": "201",
            "provider_c_player_id": "301",
        }
    )
    assert ("provider_b_player_id", "999") not in resolver


def test_resolve_rejects_proposal_bridging_two_players():
    # a proposal that would merge two existing players, each with its own provider_a ID
    resolver = ObjectResolver()
    morgan = record("player", "Alex Morgan", provider_a="101", provider_b="201")
    rapinoe = record("player", "Megan Rapinoe", provider_a="102", provider_c="302")
    resolve(resolver, [morgan, rapinoe])

    bridge = record("raw", "Alex Morgan", provider_b="201", provider_c="302")

    accepted, rejected = resolve(resolver, [bridge])

    assert accepted == []
    assert len(rejected) == 1
    rejected_record, clashes = rejected[0]
    assert rejected_record == bridge
    assert sorted(clashes, key=lambda r: r["meta"]["player_name"]) == [
        morgan,
        rapinoe,
    ]
    assert len(resolver.components()) == 2


def test_resolve_rejects_all_mutually_conflicting_proposals():
    # two proposals that disagree on the same player's provider_a ID: neither wins, regardless of order
    proposals = [
        record("raw", "Rose Lavelle", provider_b="203", provider_a="103"),
        record("raw", "Rose Lavelle", provider_b="203", provider_a="999"),
    ]
    unrelated = record("raw", "Lindsey Horan", provider_a="104", provider_c="304")

    for ordering in (proposals, proposals[::-1]):
        resolver = ObjectResolver()
        accepted, rejected = resolve(resolver, ordering + [unrelated])

        assert accepted == [unrelated]
        assert [r for r, _ in rejected] == ordering
        # nothing in the resolver to clash with: the conflict is purely between proposals
        assert all(clashes == [] for _, clashes in rejected)
        assert component_ids(resolver) == objects(
            {"provider_a_player_id": "104", "provider_c_player_id": "304"}
        )


def test_resolve_rejects_transitive_conflict_between_proposals():
    # no pair of proposals conflicts, but chained together they give one player two provider_a IDs:
    # provider_a 103 - provider_b 203 - provider_c 303 - provider_a 999
    resolver = ObjectResolver()
    chain = [
        record("raw", "Rose Lavelle", provider_a="103", provider_b="203"),
        record("raw", "Rose Lavelle", provider_b="203", provider_c="303"),
        record("raw", "Rose Lavelle", provider_c="303", provider_a="999"),
    ]

    accepted, rejected = resolve(resolver, chain)

    # only the links touching a conflicting provider_a ID are dropped; the middle link survives
    assert accepted == [chain[1]]
    assert [r for r, _ in rejected] == [chain[0], chain[2]]
    assert component_ids(resolver) == objects(
        {"provider_b_player_id": "203", "provider_c_player_id": "303"}
    )


def test_find_compresses_path():
    # merging into an existing root leaves a two-step path: provider_b 201 -> provider_a 101 -> provider_c 301
    resolver = ObjectResolver()
    resolver.add_record(
        record("player", "Alex Morgan", provider_a="101", provider_b="201")
    )
    resolver.add_record(
        record("player", "Alex Morgan", provider_c="301", provider_d="401")
    )
    resolver.add_record(
        record("raw", "Alex Morgan", provider_c="301", provider_a="101")
    )

    a = ("provider_a_player_id", "101")
    b = ("provider_b_player_id", "201")
    c = ("provider_c_player_id", "301")
    assert resolver.parent[b] == a
    assert resolver.parent[a] == c

    assert resolver.find(b) == c
    # every vertex on the path now points straight at the root
    assert resolver.parent[b] == c
    assert resolver.parent[a] == c
    assert len(resolver.components()) == 1


def test_add_record_within_existing_player():
    # a record whose IDs already belong to the same player adds no merges, only the record itself
    resolver = ObjectResolver()
    truth = [
        record("player", "Alex Morgan", provider_a="101", provider_b="201"),
        record("player", "Alex Morgan", provider_b="201", provider_c="301"),
    ]
    for r in truth:
        resolver.add_record(r)
    parents_before = dict(resolver.parent)

    repeat = record("raw", "A. Morgan", provider_a="101", provider_c="301")
    resolver.add_record(repeat)

    assert resolver.parent == parents_before
    assert resolver.components() == [
        (
            {
                "provider_a_player_id": "101",
                "provider_b_player_id": "201",
                "provider_c_player_id": "301",
            },
            truth + [repeat],
        )
    ]
