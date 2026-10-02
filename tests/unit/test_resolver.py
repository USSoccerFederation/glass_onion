import re
import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from glass_onion.resolver import ObjectResolver, to_frame


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


def assert_added(resolver: ObjectResolver, records: list[dict]):
    assert_frame_equal(resolver.added, to_frame(records))


def assert_rejected(resolver: ObjectResolver, rejected: list[tuple[dict, list[dict]]]):
    # `rejected` as `(record, records it conflicted with)` pairs
    assert_frame_equal(
        resolver.rejected,
        to_frame([r for r, _ in rejected], [c for _, c in rejected]),
    )


def assert_rejected_records(resolver: ObjectResolver, records: list[dict]):
    # the rejected records, ignoring what they conflicted with
    assert_frame_equal(resolver.rejected.drop(columns="conflicts"), to_frame(records))


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

    resolver.resolve(truth)
    assert_added(resolver, truth)
    assert_rejected(resolver, [])

    resolver.resolve(proposals)
    assert_added(resolver, proposals)
    assert_rejected(resolver, [])

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
    resolver.resolve([truth])

    conflicting = record("raw", "Alex Morgan", provider_a="101", provider_b="999")
    clean = record("raw", "Alex Morgan", provider_b="201", provider_c="301")

    resolver.resolve([conflicting, clean])

    assert_added(resolver, [clean])
    assert_rejected(resolver, [(conflicting, [truth])])
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
    resolver.resolve([morgan, rapinoe])

    bridge = record("raw", "Alex Morgan", provider_b="201", provider_c="302")

    resolver.resolve([bridge])

    assert_added(resolver, [])
    assert_rejected_records(resolver, [bridge])
    assert sorted(
        resolver.rejected["conflicts"][0], key=lambda r: r["meta"]["player_name"]
    ) == [
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
        resolver.resolve(ordering + [unrelated])

        assert_added(resolver, [unrelated])
        # nothing in the resolver to conflict with: each proposal conflicts only with the other
        assert_rejected(
            resolver, [(ordering[0], [ordering[1]]), (ordering[1], [ordering[0]])]
        )
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

    resolver.resolve(chain)

    # every link is on the path between the two provider_a IDs, so all three are dropped
    assert_added(resolver, [])
    assert_rejected(
        resolver,
        [
            (chain[0], [chain[1], chain[2]]),
            (chain[1], [chain[0], chain[2]]),
            (chain[2], [chain[0], chain[1]]),
        ],
    )
    assert resolver.components() == []


def test_resolve_rejects_transitive_conflict_through_existing_object():
    # provider_a 101 - provider_b 201 is synced; two proposals each pass the check against the resolver alone, but
    # together they give one player two provider_b IDs: provider_b 201 - provider_a 101 - provider_c 301 - provider_b 999
    resolver = ObjectResolver()
    truth = record("player", "Alex Morgan", provider_a="101", provider_b="201")
    resolver.resolve([truth])

    proposals = [
        record("raw", "Alex Morgan", provider_c="301", provider_a="101"),
        record("raw", "Alex Morgan", provider_c="301", provider_b="999"),
    ]

    for ordering in (proposals, proposals[::-1]):
        resolver = ObjectResolver()
        resolver.resolve([truth])
        resolver.resolve(ordering)

        assert_added(resolver, [])
        assert_rejected_records(resolver, ordering)
        assert all(
            conflicts[0] == truth for conflicts in resolver.rejected["conflicts"]
        )
        assert component_ids(resolver) == objects(
            {"provider_a_player_id": "101", "provider_b_player_id": "201"}
        )


def test_resolve_keeps_proposals_off_the_conflict_path():
    # provider_a 101 (synced) - provider_b 201 - provider_c 301 - provider_a 999 is a conflict, but proposals that
    # only hang off it link no provider_a IDs, so they're kept
    resolver = ObjectResolver()
    truth = record("player", "Alex Morgan", provider_a="101", provider_b="201")
    resolver.resolve([truth])

    extend = record("raw", "Alex Morgan", provider_b="201", provider_c="301")
    bridge = record("raw", "Alex Morgan", provider_c="301", provider_a="999")
    off_existing = record("raw", "Alex Morgan", provider_b="201", provider_d="401")
    off_new = record("raw", "Alex Morgan", provider_c="301", provider_e="501")
    # repeats off_existing's IDs: skipped, and the cycle it forms mustn't pull either onto the path
    duplicate = record("raw", "A. Morgan", provider_b="201", provider_d="401")

    resolver.resolve([extend, bridge, off_existing, off_new, duplicate])

    assert_added(resolver, [off_existing, off_new])
    assert_rejected(resolver, [(extend, [truth, bridge]), (bridge, [truth, extend])])
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
    resolver.resolve([morgan, rapinoe])

    proposals = [
        record("raw", "Alex Morgan", provider_b="201", provider_d="401"),
        record("raw", "Alex Morgan", provider_d="401", provider_c="302"),
    ]

    resolver.resolve(proposals)

    assert_added(resolver, [])
    assert_rejected_records(resolver, proposals)
    assert all(
        sorted(conflicts[:2], key=lambda r: r["meta"]["player_name"])
        == [morgan, rapinoe]
        for conflicts in resolver.rejected["conflicts"]
    )
    assert len(resolver.components()) == 2


def test_resolve_rejects_every_path_of_a_cyclic_conflict():
    # two independent routes from provider_a 101 to provider_a 999: removing either alone leaves the conflict
    resolver = ObjectResolver()
    truth = record("player", "Alex Morgan", provider_a="101", provider_b="201")
    resolver.resolve([truth])

    proposals = [
        record("raw", "Alex Morgan", provider_b="201", provider_c="301"),
        record("raw", "Alex Morgan", provider_c="301", provider_a="999"),
        record("raw", "Alex Morgan", provider_b="201", provider_d="401"),
        record("raw", "Alex Morgan", provider_d="401", provider_a="999"),
    ]

    resolver.resolve(proposals)

    assert_added(resolver, [])
    assert_rejected_records(resolver, proposals)
    assert component_ids(resolver) == objects(
        {"provider_a_player_id": "101", "provider_b_player_id": "201"}
    )


def test_resolve_reports_resolver_and_proposal_conflicts_together():
    # a conflict running through an existing player: provider_a 101 - provider_b 201 - provider_c 301 - provider_a 999
    resolver = ObjectResolver()
    truth = record("player", "Alex Morgan", provider_a="101", provider_b="201")
    resolver.resolve([truth])

    extend = record("raw", "Alex Morgan", provider_b="201", provider_c="301")
    bridge = record("raw", "Alex Morgan", provider_c="301", provider_a="999")

    resolver.resolve([extend, bridge])

    assert_added(resolver, [])
    # every resolver record involved comes first, even for bridge, which never touches truth's IDs directly;
    # then the other proposals rejected in the same conflict
    assert_rejected(resolver, [(extend, [truth, bridge]), (bridge, [truth, extend])])


def test_resolve_sets_flattened_frames():
    # one row per record: source, every provider ID field (normalized, None if missing), then meta
    resolver = ObjectResolver()
    truth = record("player", "Alex Morgan", provider_a=101.0, provider_b="201")
    resolver.resolve([truth])

    conflicting = record("raw", "A. Morgan", provider_a="101", provider_b=999)
    new = record("raw", "Rose Lavelle", provider_c="303", provider_a=None)
    resolver.resolve([conflicting, new])

    assert_frame_equal(
        resolver.added,
        pd.DataFrame(
            [
                {
                    "source": "raw",
                    "provider_c_player_id": "303",
                    "provider_a_player_id": None,
                    "player_name": "Rose Lavelle",
                }
            ]
        ),
    )
    assert_frame_equal(
        resolver.rejected,
        pd.DataFrame(
            [
                {
                    "source": "raw",
                    "provider_a_player_id": "101",
                    "provider_b_player_id": "999",
                    "player_name": "A. Morgan",
                    "conflicts": [truth],
                }
            ]
        ),
    )


def test_resolve_frames_hold_latest_call_only():
    resolver = ObjectResolver()
    assert list(resolver.added.columns) == ["source"]
    assert list(resolver.rejected.columns) == ["source", "conflicts"]

    truth = record("player", "Alex Morgan", provider_a="101", provider_b="201")
    conflicting = record("raw", "Alex Morgan", provider_a="101", provider_b="999")
    resolver.resolve([truth])
    resolver.resolve([conflicting])
    assert_added(resolver, [])
    assert_rejected(resolver, [(conflicting, [truth])])

    # an empty call clears both, keeping their columns
    resolver.resolve([])
    assert resolver.added.empty and list(resolver.added.columns) == ["source"]
    assert resolver.rejected.empty
    assert list(resolver.rejected.columns) == ["source", "conflicts"]
    assert resolver.rejected["conflicts"].dtype == object


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

    resolver.resolve(proposals)

    assert_added(resolver, proposals)
    assert_rejected(resolver, [])
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

    resolver.resolve(proposals)

    assert_added(resolver, proposals)
    assert_rejected(resolver, [])
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

    resolver.resolve(proposals)

    assert_added(resolver, proposals)
    assert_rejected(resolver, [])
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
    resolver.resolve([truth])

    conflicting = record("raw", "Alex Morgan", provider_a="102", provider_b=201)

    resolver.resolve([conflicting])

    assert_added(resolver, [])
    assert_rejected(resolver, [(conflicting, [truth])])


def test_resolve_drops_records_without_ids():
    # records with no IDs, or only missing ones, are dropped instead of crashing
    resolver = ObjectResolver()
    valid = record("raw", "Alex Morgan", provider_a="101", provider_b="201")
    empty = record("raw", "Megan Rapinoe")
    all_missing = record("raw", "Rose Lavelle", provider_a=None, provider_b=np.nan)

    resolver.resolve([empty, valid, all_missing])

    assert_added(resolver, [valid])
    assert_rejected(resolver, [])
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


def test_add_record_skips_ids_within_one_record():
    # a 2-ID record whose IDs are both in an earlier 3-ID record adds nothing, so it's skipped like an exact repeat
    resolver = ObjectResolver()
    full = record(
        "player", "Alex Morgan", provider_a="101", provider_b="201", provider_c="301"
    )
    assert resolver.add_record(full)

    assert not resolver.add_record(
        record("raw", "A. Morgan", provider_a="101", provider_c=301)
    )
    assert not resolver.add_record(record("raw", "A. Morgan", provider_b="201"))
    assert resolver.components()[0][1] == [full]


def test_add_record_replaces_smaller_record_with_larger():
    # the other way round, the larger record replaces the smaller one, so the result is the same either way
    resolver = ObjectResolver()
    partial = record("raw", "A. Morgan", provider_a="101", provider_b="201")
    full = record(
        "player", "Alex Morgan", provider_a="101", provider_b="201", provider_c="301"
    )

    assert resolver.add_record(partial)
    assert resolver.add_record(full)
    assert resolver.components()[0][1] == [full]
    # the replaced record no longer counts as kept
    assert not resolver.add_record(partial)
    assert resolver.components()[0][1] == [full]


def test_add_record_replaces_every_record_inside_it():
    # two 2-ID records linking one player are both inside the later 3-ID record; an unrelated one is untouched
    resolver = ObjectResolver()
    links = [
        record("raw", "A. Morgan", provider_a="101", provider_b="201"),
        record("raw", "A. Morgan", provider_b="201", provider_c="301"),
    ]
    extend = record("raw", "A. Morgan", provider_c="301", provider_d="401")
    for r in links + [extend]:
        resolver.add_record(r)

    full = record(
        "player", "Alex Morgan", provider_a="101", provider_b="201", provider_c="301"
    )
    assert resolver.add_record(full)
    assert resolver.components()[0][1] == [extend, full]


def test_resolve_skips_partial_repeats_in_one_batch():
    # within one batch, a record inside a larger one is skipped whichever comes first
    partial = record("raw", "A. Morgan", provider_a="101", provider_b="201")
    full = record(
        "player", "Alex Morgan", provider_a="101", provider_b="201", provider_c="301"
    )

    for ordering in ([partial, full], [full, partial]):
        resolver = ObjectResolver()
        resolver.resolve(ordering)

        assert_added(resolver, [full])
        assert_rejected(resolver, [])
        assert resolver.components()[0][1] == [full]


def test_resolve_replaces_smaller_records_across_calls():
    # a smaller record accepted in an earlier call is replaced by the larger one, so later conflicting records list it alone
    partial = record("raw", "A. Morgan", provider_a="101", provider_b="201")
    full = record(
        "player", "Alex Morgan", provider_a="101", provider_b="201", provider_c="301"
    )
    conflicting = record("raw", "Alex Morgan", provider_a="101", provider_c="999")

    for first, second in ([partial, full], [full, partial]):
        resolver = ObjectResolver()
        resolver.resolve([first])
        resolver.resolve([second])

        assert resolver.components()[0][1] == [full]
        resolver.resolve([conflicting])
        assert_added(resolver, [])
        assert_rejected(resolver, [(conflicting, [full])])


def test_resolve_skips_partial_repeats_of_existing_records():
    resolver = ObjectResolver()
    full = record(
        "player", "Alex Morgan", provider_a="101", provider_b="201", provider_c="301"
    )
    resolver.resolve([full])

    partial = record("raw", "A. Morgan", provider_b="201", provider_c="301")
    extend = record("raw", "A. Morgan", provider_c="301", provider_d="401")
    resolver.resolve([partial, extend])

    assert_added(resolver, [extend])
    assert_rejected(resolver, [])


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


def test_resolve_repeats_do_not_grow_conflicts():
    # a player re-synced every matchday is still one record, so a later conflicting record lists just that record
    resolver = ObjectResolver()
    truth = record("matchday_1", "Alex Morgan", provider_a="101", provider_b="201")
    resolver.resolve([truth])
    for md in range(2, 51):
        repeat = record(
            f"matchday_{md}", "Alex Morgan", provider_a="101", provider_b="201"
        )
        resolver.resolve([repeat])
        # repeats aren't rejected, but they aren't added either
        assert_added(resolver, [])
        assert_rejected(resolver, [])

    conflicting = record("raw", "Alex Morgan", provider_a="101", provider_b="999")
    resolver.resolve([conflicting])

    assert_rejected(resolver, [(conflicting, [truth])])


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
