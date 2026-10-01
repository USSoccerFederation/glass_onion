import re
import pytest
import numpy as np
import pandas as pd

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
        # nothing in the resolver to clash with: each proposal clashes only with the other
        assert rejected == [(ordering[0], [ordering[1]]), (ordering[1], [ordering[0]])]
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

    # every link is on the path between the two provider_a IDs, so all three are dropped
    assert accepted == []
    assert rejected == [
        (chain[0], [chain[1], chain[2]]),
        (chain[1], [chain[0], chain[2]]),
        (chain[2], [chain[0], chain[1]]),
    ]
    assert resolver.components() == []


def test_resolve_rejects_transitive_conflict_through_existing_object():
    # provider_a 101 - provider_b 201 is synced; two proposals each pass the check against the resolver alone, but
    # together they give one player two provider_b IDs: provider_b 201 - provider_a 101 - provider_c 301 - provider_b 999
    resolver = ObjectResolver()
    truth = record("player", "Alex Morgan", provider_a="101", provider_b="201")
    resolve(resolver, [truth])

    proposals = [
        record("raw", "Alex Morgan", provider_c="301", provider_a="101"),
        record("raw", "Alex Morgan", provider_c="301", provider_b="999"),
    ]

    for ordering in (proposals, proposals[::-1]):
        resolver = ObjectResolver()
        resolve(resolver, [truth])
        accepted, rejected = resolve(resolver, ordering)

        assert accepted == []
        assert [r for r, _ in rejected] == ordering
        assert all(clashes[0] == truth for _, clashes in rejected)
        assert component_ids(resolver) == objects(
            {"provider_a_player_id": "101", "provider_b_player_id": "201"}
        )


def test_resolve_keeps_proposals_off_the_conflict_path():
    # provider_a 101 (synced) - provider_b 201 - provider_c 301 - provider_a 999 is a conflict, but proposals that
    # only hang off it link no provider_a IDs, so they're kept
    resolver = ObjectResolver()
    truth = record("player", "Alex Morgan", provider_a="101", provider_b="201")
    resolve(resolver, [truth])

    extend = record("raw", "Alex Morgan", provider_b="201", provider_c="301")
    bridge = record("raw", "Alex Morgan", provider_c="301", provider_a="999")
    off_existing = record("raw", "Alex Morgan", provider_b="201", provider_d="401")
    off_new = record("raw", "Alex Morgan", provider_c="301", provider_e="501")
    # duplicates form a cycle off the path, which mustn't pull them onto it
    duplicate = record("raw", "A. Morgan", provider_b="201", provider_d="401")

    accepted, rejected = resolve(
        resolver, [extend, bridge, off_existing, off_new, duplicate]
    )

    assert accepted == [off_existing, off_new, duplicate]
    assert rejected == [(extend, [truth, bridge]), (bridge, [truth, extend])]
    assert component_ids(resolver) == objects(
        {
            "provider_a_player_id": "101",
            "provider_b_player_id": "201",
            "provider_d_player_id": "401",
        },
        {"provider_c_player_id": "301", "provider_e_player_id": "501"},
    )


def test_resolve_rejects_conflict_between_existing_objects():
    # two synced players with different provider_a IDs, bridged by two proposals that hold no provider_a ID themselves
    resolver = ObjectResolver()
    morgan = record("player", "Alex Morgan", provider_a="101", provider_b="201")
    rapinoe = record("player", "Megan Rapinoe", provider_a="102", provider_c="302")
    resolve(resolver, [morgan, rapinoe])

    proposals = [
        record("raw", "Alex Morgan", provider_b="201", provider_d="401"),
        record("raw", "Alex Morgan", provider_d="401", provider_c="302"),
    ]

    accepted, rejected = resolve(resolver, proposals)

    assert accepted == []
    assert [r for r, _ in rejected] == proposals
    assert all(
        sorted(clashes[:2], key=lambda r: r["meta"]["player_name"]) == [morgan, rapinoe]
        for _, clashes in rejected
    )
    assert len(resolver.components()) == 2


def test_resolve_rejects_every_path_of_a_cyclic_conflict():
    # two independent routes from provider_a 101 to provider_a 999: removing either alone leaves the conflict
    resolver = ObjectResolver()
    truth = record("player", "Alex Morgan", provider_a="101", provider_b="201")
    resolve(resolver, [truth])

    proposals = [
        record("raw", "Alex Morgan", provider_b="201", provider_c="301"),
        record("raw", "Alex Morgan", provider_c="301", provider_a="999"),
        record("raw", "Alex Morgan", provider_b="201", provider_d="401"),
        record("raw", "Alex Morgan", provider_d="401", provider_a="999"),
    ]

    accepted, rejected = resolve(resolver, proposals)

    assert accepted == []
    assert [r for r, _ in rejected] == proposals
    assert component_ids(resolver) == objects(
        {"provider_a_player_id": "101", "provider_b_player_id": "201"}
    )


def test_resolve_reports_resolver_and_proposal_clashes_together():
    # a conflict running through an existing player: provider_a 101 - provider_b 201 - provider_c 301 - provider_a 999
    resolver = ObjectResolver()
    truth = record("player", "Alex Morgan", provider_a="101", provider_b="201")
    resolve(resolver, [truth])

    extend = record("raw", "Alex Morgan", provider_b="201", provider_c="301")
    bridge = record("raw", "Alex Morgan", provider_c="301", provider_a="999")

    accepted, rejected = resolve(resolver, [extend, bridge])

    assert accepted == []
    # every resolver record involved comes first, even for bridge, which never touches truth's IDs directly;
    # then the other proposals rejected in the same conflict
    assert rejected == [(extend, [truth, bridge]), (bridge, [truth, extend])]


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


def test_resolve_ignores_missing_ids():
    # missing provider_b IDs must not become a shared vertex linking unrelated players
    resolver = ObjectResolver()
    proposals = [
        record("raw", "Alex Morgan", provider_a="101", provider_b=None),
        record("raw", "Megan Rapinoe", provider_a="102", provider_b=float("nan")),
        record("raw", "Rose Lavelle", provider_a="103", provider_b=np.nan),
        record("raw", "Lindsey Horan", provider_a="104", provider_b=pd.NA),
        record("raw", "Sophia Smith", provider_a="105", provider_b=""),
        record("raw", "Trinity Rodman", provider_a="106", provider_b="  "),
        record("raw", "Mallory Swanson", provider_a="107", provider_b=np.nan),
        record("raw", "Crystal Dunn", provider_a="108", provider_b="<NA>"),
        record("raw", "Naomi Girma", provider_a="109", provider_b="null"),
        record("raw", "Tierna Davidson", provider_a="110", provider_b="null"),
        record("raw", "Lindsey Heaps", provider_a="111", provider_b="NULL"),
        record("raw", "Jaedyn Shaw", provider_a="112", provider_b="Null"),
        record("raw", "Catarina Macario", provider_a="113", provider_b="<na>"),
    ]

    accepted, rejected = resolve(resolver, proposals)

    assert accepted == proposals
    assert rejected == []
    assert component_ids(resolver) == objects(
        *({"provider_a_player_id": str(i)} for i in range(101, 114))
    )
    # the records themselves are carried along untouched
    assert resolver.components()[0][1][0]["ids"]["provider_b_player_id"] is None


def test_resolve_keeps_zero_id():
    # "0" is a valid ID, not a placeholder: it links records like any other ID, and 0 is the same ID
    resolver = ObjectResolver()
    proposals = [
        record("raw", "Alex Morgan", provider_a="0", provider_b="201"),
        record("raw", "Alex Morgan", provider_a=0, provider_c="301"),
    ]

    accepted, rejected = resolve(resolver, proposals)

    assert accepted == proposals
    assert rejected == []
    assert component_ids(resolver) == objects(
        {
            "provider_a_player_id": "0",
            "provider_b_player_id": "201",
            "provider_c_player_id": "301",
        }
    )


def test_resolve_normalizes_numeric_ids():
    # 101, 101.0, numpy ints/floats and "101" are all the same provider_a ID
    resolver = ObjectResolver()
    proposals = [
        record("raw", "Alex Morgan", provider_a=101, provider_b="201"),
        record("raw", "Alex Morgan", provider_a=101.0, provider_c=301),
        record(
            "raw", "Alex Morgan", provider_a=np.int64(101), provider_d=np.float64(401.0)
        ),
        record("raw", "Alex Morgan", provider_a="101", provider_e="501"),
    ]

    accepted, rejected = resolve(resolver, proposals)

    assert accepted == proposals
    assert rejected == []
    assert component_ids(resolver) == objects(
        {
            "provider_a_player_id": "101",
            "provider_b_player_id": "201",
            "provider_c_player_id": "301",
            "provider_d_player_id": "401",
            "provider_e_player_id": "501",
        }
    )


def test_resolve_numeric_and_string_ids_conflict():
    # once normalized, 101.0 and "102" are two distinct provider_a IDs for the same provider_b ID
    resolver = ObjectResolver()
    truth = record("player", "Alex Morgan", provider_a=101.0, provider_b="201")
    resolve(resolver, [truth])

    conflicting = record("raw", "Alex Morgan", provider_a="102", provider_b=201)

    accepted, rejected = resolve(resolver, [conflicting])

    assert accepted == []
    assert rejected == [(conflicting, [truth])]


def test_resolve_drops_records_without_ids():
    # records with no IDs, or only missing ones, are dropped instead of crashing
    resolver = ObjectResolver()
    valid = record("raw", "Alex Morgan", provider_a="101", provider_b="201")
    empty = record("raw", "Megan Rapinoe")
    all_missing = record("raw", "Rose Lavelle", provider_a=None, provider_b=np.nan)

    accepted, rejected = resolve(resolver, [empty, valid, all_missing])

    assert accepted == [valid]
    assert rejected == []
    assert component_ids(resolver) == objects(
        {"provider_a_player_id": "101", "provider_b_player_id": "201"}
    )


def test_add_record_skips_repeated_ids():
    # re-syncing the same player keeps the first record only, even if the repeat's IDs are written differently
    resolver = ObjectResolver()
    first = record("matchday_1", "Alex Morgan", provider_a="101", provider_b="201")
    resolver.add_record(first)
    for md in range(2, 6):
        resolver.add_record(
            record(f"matchday_{md}", "A. Morgan", provider_a=101, provider_b=201.0)
        )

    assert resolver.components() == [
        (
            {"provider_a_player_id": "101", "provider_b_player_id": "201"},
            [first],
        )
    ]


def test_add_record_keeps_new_link_between_known_ids():
    # a record linking a new combination of IDs already in one object adds no merges, but isn't a repeat
    resolver = ObjectResolver()
    truth = [
        record("player", "Alex Morgan", provider_a="101", provider_b="201"),
        record("player", "Alex Morgan", provider_b="201", provider_c="301"),
    ]
    for r in truth:
        resolver.add_record(r)

    link = record("raw", "Alex Morgan", provider_a="101", provider_c="301")
    resolver.add_record(link)
    resolver.add_record(link)

    assert resolver.components()[0][1] == truth + [link]


def test_resolve_repeats_do_not_grow_clashes():
    # a player re-synced every matchday is still one record, so a later conflict clashes with just that record
    resolver = ObjectResolver()
    truth = record("matchday_1", "Alex Morgan", provider_a="101", provider_b="201")
    resolve(resolver, [truth])
    for md in range(2, 51):
        repeat = record(
            f"matchday_{md}", "Alex Morgan", provider_a="101", provider_b="201"
        )
        accepted, rejected = resolve(resolver, [repeat])
        # repeats are still accepted: they agree with the resolver
        assert accepted == [repeat]
        assert rejected == []

    conflicting = record("raw", "Alex Morgan", provider_a="101", provider_b="999")
    accepted, rejected = resolve(resolver, [conflicting])

    assert rejected == [(conflicting, [truth])]


def test_add_record_merges_into_largest_component():
    # the record's first ID is new, but the existing three-ID object is larger, so it keeps its root
    resolver = ObjectResolver()
    resolver.add_record(
        record("player", "Alex Morgan", provider_a="101", provider_b="201")
    )
    resolver.add_record(
        record("player", "Alex Morgan", provider_b="201", provider_c="301")
    )
    root = resolver.find(("provider_a_player_id", "101"))

    resolver.add_record(
        record("raw", "Alex Morgan", provider_d="401", provider_a="101")
    )

    assert resolver.parent[("provider_d_player_id", "401")] == root
    assert resolver.find(("provider_d_player_id", "401")) == root
    assert len(resolver.components()) == 1


def test_add_record_rejects_record_without_ids():
    resolver = ObjectResolver()

    with pytest.raises(AssertionError, match=re.escape("Record has no valid IDs:")):
        resolver.add_record(record("raw", "Megan Rapinoe", provider_a=None))
