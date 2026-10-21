"""
End-to-end tests for player synchronization and resolution: every player fixture is synchronized with PlayerSyncEngine
match by match, and a name-only provider (provider_d) is synchronized against the existing players and those match
proposals. Every proposal is then resolved into one ObjectResolver at the same time.

The resolver starts with a set of existing players (mimicking earlier runs): these player records should persist through a test run, with any conflicts should be rejected by the resolver.

Known gaps are strict xfails.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from glass_onion.player import PlayerSyncableContent, PlayerSyncEngine
from glass_onion.resolver import ObjectResolver, Record, Vertex, vertices
from tests.utils import FIXTURE_DATA_PATH, utils_create_syncables

MATCH_FILES = [
    "2023-09-13-oma-usa.csv",
    "2024-09-22-chi-sd-w.csv",
    "2025-07-05-usa-mex.csv",
    "2025-07-15-kor-jpn.csv",
]


def read_fixture(file: str) -> pd.DataFrame:
    return pd.read_csv(
        FIXTURE_DATA_PATH / "player" / file,
        dtype={"jersey_number": str, "birth_date": str},
    )


FIXTURES = {f: read_fixture(f) for f in MATCH_FILES}


@dataclass(frozen=True)
class Person:
    # one real-world player, as each provider names them in the fixtures. `jersey` tells apart same-named teammates.
    files: tuple[str, ...]
    team: str
    names: dict[str, str] = field(hash=False)
    jersey: str | None = None


def fixture_id(person: Person, provider: str) -> str:
    rows = pd.concat([FIXTURES[f] for f in person.files])
    rows = rows[
        (rows["data_provider"] == provider)
        & (rows["player_name"] == person.names[provider])
        & (rows["team_id"] == person.team)
    ]
    if person.jersey is not None:
        rows = rows[rows["jersey_number"] == person.jersey]
    ids = set(rows["provider_player_id"].astype(str))
    assert len(ids) == 1, f"{person.names[provider]} ({provider}) has {len(ids)} IDs in {person.files}"
    return ids.pop()


def expected_ids(person: Person) -> set[Vertex]:
    return {(f"{p}_player_id", fixture_id(person, p)) for p in person.names}


OMA_USA, CHI_SD, USA_MEX, KOR_JPN = MATCH_FILES

PEOPLE = {
    # --- Korean names, romanized three ways (kor-jpn) ------------------------------------------------------------------
    "sang_yun_kang": Person((KOR_JPN,), "KOR", {"provider_a": "Sang-yun Kang", "provider_b": "Sang-Yoon Kang", "provider_c": "Sang-Yun Kang"}),
    "myeong_kwan_seo": Person((KOR_JPN,), "KOR", {"provider_a": "Myeong-kwan Seo", "provider_b": "Myeong-Kwan Seo", "provider_c": "Myung-Guan Seo"}),
    # provider_c's name has a leading space in the fixture
    "jin_seob_park": Person((KOR_JPN,), "KOR", {"provider_a": "Jin-seob Park", "provider_b": "Jin-Sub Park", "provider_c": " Jin-Seop Park"}),
    "ju_sung_kim": Person((KOR_JPN,), "KOR", {"provider_a": "Ju-sung Kim", "provider_b": "Joo-Sung Kim", "provider_c": "Ju-Sung Kim"}),
    "jin_gyu_kim": Person((KOR_JPN,), "KOR", {"provider_a": "Jin-gyu Kim", "provider_b": "Jin-Kyu Kim", "provider_c": "Jin-gyu Kim"}),
    # two different teammates with the same name
    "tae_hyeon_kim_22": Person((KOR_JPN,), "KOR", {"provider_a": "Tae-hyeon Kim", "provider_b": "Tae-Hyun Kim", "provider_c": "Tae-Hyeon Kim"}, "22"),
    "tae_hyeon_kim_26": Person((KOR_JPN,), "KOR", {"provider_a": "Tae-hyeon Kim", "provider_b": "Tae-Hyeon Kim", "provider_c": "Tae-Hyeon Kim"}, "26"),
    # --- Arabic names, anglicized differently, often with extra given names (oma-usa) ----------------------------------
    "essam_al_subhi": Person((OMA_USA,), "OMA", {"provider_a": "Essam Al-Subhi", "provider_b": "Issam Al Sabhi"}),
    "tamim_al_balushi": Person((OMA_USA,), "OMA", {"provider_a": "Tamim Al-Balushi", "provider_b": "Tameem Haitham Shambeh Al Balushi"}),
    "faiyz_al_rusheidi": Person((OMA_USA,), "OMA", {"provider_a": "Faiyz Al-Rusheidi", "provider_b": "Faiyz Issa Khadoom Al Rashidi"}),
    "fahmi_durbein": Person((OMA_USA,), "OMA", {"provider_a": "Fahmi Durbein", "provider_b": "Fahmi Said Rajab Nasib Bait Durbin"}),
    # in the resolver before the run with only provider_a's ID (see EXISTING_PLAYERS)
    "juma_al_habsi": Person((OMA_USA,), "OMA", {"provider_a": "Juma Al-Habsi", "provider_b": "Juma Marhoon Juma Al Habsi"}),
    # only one provider lists them
    "mohammed_al_amri": Person((OMA_USA,), "OMA", {"provider_b": "Mohammed Al Amri"}),
    "omar_al_malki": Person((OMA_USA,), "OMA", {"provider_a": "Omar Al-Malki"}),
    # --- full legal names vs common names (usa-mex) --------------------------------------------------------------------
    "guillermo_ochoa": Person((USA_MEX,), "MEX", {"provider_a": "Guillermo Ochoa", "provider_b": "Francisco Guillermo Ochoa Magaña", "provider_c": "Francisco Guillermo Ochoa Magaña"}),
    "luis_malagon": Person((USA_MEX,), "MEX", {"provider_a": "Luis Malagón", "provider_b": "Luis Ángel Malagón Velázquez", "provider_c": "Luis Ángel Malagón Velázquez"}),
    "matt_turner": Person((USA_MEX,), "USA", {"provider_a": "Matt Turner", "provider_b": "Matthew Charles Turner", "provider_c": "Matt Turner"}),
    # --- accents and spelling variants ---------------------------------------------------------------------------------
    "erik_lira": Person((USA_MEX,), "MEX", {"provider_a": "Érik Lira", "provider_b": "Erik Antonio Lira Méndez", "provider_c": "Erick Antonio Lira Méndez"}),
    "santiago_gimenez": Person((USA_MEX,), "MEX", {"provider_a": "Santiago Gimenez", "provider_b": "Santiago Tomás Giménez", "provider_c": "Santiago Tomás Giménez"}),
    "sergino_dest": Person((OMA_USA,), "USA", {"provider_a": "Sergiño Dest", "provider_b": "Sergino Dest"}),
    # --- NWSL, where provider_a has no jersey numbers (chi-sd-w) -------------------------------------------------------
    # married name vs maiden name
    "makenzy_robbe": Person((CHI_SD,), "SD", {"provider_a": "Makenzy Robbe", "provider_b": "Makenzy Doniak"}),
    # short names vs full names
    "mel_barcenas": Person((CHI_SD,), "SD", {"provider_a": "Mel Barcenas", "provider_b": "Melanie Barcenas"}),
    "ally_schlegel": Person((CHI_SD,), "CHI", {"provider_a": "Ally Schlegel", "provider_b": "Allyson Schlegel"}),
    "maria_sanchez": Person((CHI_SD,), "SD", {"provider_a": "María Sánchez", "provider_b": "Maria Guadalupe Sanchez Morales"}),
    "chardonnay_curran": Person((CHI_SD,), "CHI", {"provider_b": "Chardonnay Curran"}),
    # a name-only profile links to her (see NAME_ONLY_PROFILES)
    "naomi_girma": Person((CHI_SD,), "SD", {"provider_a": "Naomi Girma", "provider_b": "Naomi Haile Girma"}),
    # --- USA players in both USA matches: oma-usa has provider_a and provider_b, usa-mex adds provider_c -------------
    "tim_ream": Person((OMA_USA, USA_MEX), "USA", {"provider_a": "Tim Ream", "provider_b": "Tim Ream", "provider_c": "Timothy Michael Ream"}),
    "miles_robinson": Person((OMA_USA, USA_MEX), "USA", {"provider_a": "Miles Robinson", "provider_b": "Miles Gordon Robinson", "provider_c": "Miles Robinson"}),
    "brenden_aaronson": Person((OMA_USA, USA_MEX), "USA", {"provider_a": "Brenden Aaronson", "provider_b": "Brenden Aaronson", "provider_c": "Brenden  Aaronson"}),
    # in usa-mex, provider_b lists him under a second, duplicate profile (see DUPLICATE_PROFILE_ID)
    "chris_richards": Person((OMA_USA,), "USA", {"provider_a": "Chris Richards", "provider_b": "Chris Richards"}),
    # an override in the resolver links his provider_a ID to a different provider_b ID (see EXISTING_PLAYERS)
    "bong_soo_kim": Person((KOR_JPN,), "KOR", {"provider_a": "Bong-soo Kim", "provider_b": "Bong-Soo Kim", "provider_c": "Bong-Soo Kim"}),
}

DUPLICATE_PROFILE_ID = 9941266
OVERRIDE_ID = "9900001"


def existing(person_key: str, providers: list[str], name: str, gender: str = "male", **ids: str) -> Record:
    person = PEOPLE[person_key]
    return {
        "source": "existing",
        "ids": {**{f"{p}_player_id": fixture_id(person, p) for p in providers}, **{f"{p}_player_id": i for p, i in ids.items()}},
        "meta": {"player_name": name, "player_gender": gender},
    }


# the resolver before the run
EXISTING_PLAYERS = [
    # already fully synced from an earlier oma-usa run
    existing("tim_ream", ["provider_a", "provider_b"], "Tim Ream"),
    # only provider_a's ID is known so far
    existing("juma_al_habsi", ["provider_a"], "Juma Al-Habsi"),
    # a manual override: provider_a's ID belongs with this provider_b ID, not the one sync will propose
    existing("bong_soo_kim", ["provider_a"], "Bong-soo Kim", provider_b=OVERRIDE_ID),
]


# ---------------------------------------------------------------------------------------------------------------------
# Name-only provider: profiles with no jersey number, team, or birth date, matched by name within each gender against the
# existing players and the match proposals
# ---------------------------------------------------------------------------------------------------------------------

# (profile ID, player_name, player_gender)
NAME_ONLY_PROFILES = [
    # resolved players, spelled without an accent / under a maiden name
    ("401", "Sergino Dest", "male"),
    ("402", "Makenzy Doniak", "female"),
    ("403", "Naomi Girma", "female"),
    # decoys: former US players who aren't in any fixture. Tim Howard shares only a first name with Tim Ream
    ("411", "Tim Howard", "male"),
    # share nothing with anyone
    ("412", "Landon Donovan", "male"),
    ("413", "Mia Hamm", "female"),
]


def name_only_sync(known_records: list[Record], profiles: list[tuple[str, str, str]]) -> list[Record]:
    # one row per known record, under a stand-in provider whose ID is the record's index in `known_records`
    known = pd.DataFrame(
        [
            {
                "known_player_id": str(k),
                "player_name": r["meta"]["player_name"],
                "team_id": r["meta"]["player_gender"],
            }
            for k, r in enumerate(known_records)
        ]
    )
    new = pd.DataFrame(profiles, columns=["provider_d_player_id", "player_name", "team_id"])
    for frame in (known, new):
        frame["player_nickname"] = frame["player_name"]

    # team_id holds the gender, so a team-only match would pair any two remaining players of one gender
    engine = PlayerSyncEngine(
        [PlayerSyncableContent("known", known), PlayerSyncableContent("provider_d", new)],
        allow_team_only_layer=False,
    )
    synced = engine.synchronize().data
    records = []
    for _, row in synced.iterrows():
        if pd.isna(row["provider_d_player_id"]):
            continue
        # a linked profile is proposed together with every ID of the record it matched
        ids = {"provider_d_player_id": row["provider_d_player_id"]}
        if pd.notna(row["known_player_id"]):
            ids.update(known_records[int(row["known_player_id"])]["ids"])
        records.append(
            {"source": "name_only", "ids": ids, "meta": {"player_name": row["player_name"], "player_gender": row["team_id"]}}
        )
    return records


# ---------------------------------------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------------------------------------


def with_duplicate_profile(file: str, data: pd.DataFrame) -> pd.DataFrame:
    # in usa-mex, provider_b lists Chris Richards under a second profile
    if file == USA_MEX:
        richards = (data["data_provider"] == "provider_b") & (data["player_name"] == "Chris Richards")
        assert richards.sum() == 1
        data.loc[richards, "provider_player_id"] = DUPLICATE_PROFILE_ID
    return data


def match_sync(file: str, data: pd.DataFrame) -> list[Record]:
    # synchronize() doesn't carry player_gender through, but every match is one gender
    genders = data["player_gender"].unique()
    assert len(genders) == 1
    synced = PlayerSyncEngine(utils_create_syncables(data, "player")).synchronize().data
    id_fields = [c for c in synced.columns if c.endswith("_player_id")]
    return [
        {
            "source": file,
            "ids": {f: row[f] for f in id_fields},
            "meta": {"player_name": row["player_name"], "player_gender": genders[0]},
        }
        for _, row in synced.iterrows()
    ]


def run_pipeline(edit: Callable[[str, pd.DataFrame], pd.DataFrame] = with_duplicate_profile) -> dict[str, Any]:
    resolver = ObjectResolver()
    resolver.resolve(EXISTING_PLAYERS)
    assert len(resolver.added) == len(EXISTING_PLAYERS)

    proposals = [r for file in MATCH_FILES for r in match_sync(file, edit(file, FIXTURES[file].copy()))]
    proposals += name_only_sync(EXISTING_PLAYERS + proposals, NAME_ONLY_PROFILES)
    resolver.resolve(proposals)
    return {"resolver": resolver, "rejected": resolver.rejected}


@pytest.fixture(scope="module")
def pipeline() -> dict[str, Any]:
    return run_pipeline()


def object_of(resolver: ObjectResolver, v: Vertex) -> set[Vertex] | None:
    # the IDs of the object holding `v`, if any
    if v not in resolver:
        return None
    return set(resolver.ids[resolver.find(v)].items())


def assert_one_player(pipeline: dict[str, Any], ids: set[Vertex]):
    # every ID is in the resolver, all in one object that holds no other IDs
    resolver = pipeline["resolver"]
    missing = {v for v in ids if v not in resolver}
    assert not missing, f"IDs missing from the resolver: {sorted(missing)}"
    objects = {frozenset(object_of(resolver, v)) for v in ids}
    assert len(objects) == 1, f"IDs split across objects: {[sorted(o) for o in objects]}"
    assert objects.pop() == ids


def rejected_ids(pipeline: dict[str, Any]) -> list[tuple[set[Vertex], list[Record]]]:
    frame = pipeline["rejected"]
    fields = [c for c in frame.columns if c.endswith("_player_id")]
    return [
        ({(f, row[f]) for f in fields if pd.notna(row[f])}, row["conflicts"])
        for _, row in frame.iterrows()
    ]


# ---------------------------------------------------------------------------------------------------------------------
# Synchronization across providers
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("person_key", ["sang_yun_kang", "myeong_kwan_seo", "jin_seob_park", "ju_sung_kim", "jin_gyu_kim"])
def test_korean_romanizations_across_three_providers(pipeline, person_key):
    assert_one_player(pipeline, expected_ids(PEOPLE[person_key]))


def test_same_named_teammates_are_not_merged(pipeline):
    assert_one_player(pipeline, expected_ids(PEOPLE["tae_hyeon_kim_22"]))
    assert_one_player(pipeline, expected_ids(PEOPLE["tae_hyeon_kim_26"]))


@pytest.mark.parametrize("person_key", ["essam_al_subhi", "tamim_al_balushi", "faiyz_al_rusheidi", "fahmi_durbein"])
def test_arabic_anglicizations_across_two_providers(pipeline, person_key):
    assert_one_player(pipeline, expected_ids(PEOPLE[person_key]))


def test_anglicization_adds_provider_id_to_existing_player(pipeline):
    assert_one_player(pipeline, expected_ids(PEOPLE["juma_al_habsi"]))


@pytest.mark.parametrize("person_key", ["guillermo_ochoa", "luis_malagon", "matt_turner"])
def test_full_legal_names_vs_common_names(pipeline, person_key):
    assert_one_player(pipeline, expected_ids(PEOPLE[person_key]))


@pytest.mark.parametrize("person_key", ["erik_lira", "santiago_gimenez"])
def test_accents_and_spelling_variants(pipeline, person_key):
    assert_one_player(pipeline, expected_ids(PEOPLE[person_key]))


def test_married_name_vs_maiden_name_without_jersey_numbers(pipeline):
    # provider_a has no jersey numbers in chi-sd-w, and no name test passes, so this relies on the team-only layer
    assert_one_player(pipeline, expected_ids(PEOPLE["makenzy_robbe"]))


@pytest.mark.parametrize("person_key", ["mel_barcenas", "ally_schlegel", "maria_sanchez"])
def test_short_names_vs_full_names_without_jersey_numbers(pipeline, person_key):
    assert_one_player(pipeline, expected_ids(PEOPLE[person_key]))


@pytest.mark.parametrize("person_key", ["tim_ream", "miles_robinson", "brenden_aaronson"])
def test_ids_from_two_matches_merge_into_one_player(pipeline, person_key):
    assert_one_player(pipeline, expected_ids(PEOPLE[person_key]))


@pytest.mark.parametrize("person_key", ["mohammed_al_amri", "omar_al_malki", "chardonnay_curran"])
def test_single_provider_player_in_multi_provider_match(pipeline, person_key):
    assert_one_player(pipeline, expected_ids(PEOPLE[person_key]))


# ---------------------------------------------------------------------------------------------------------------------
# Resolution against existing players
# ---------------------------------------------------------------------------------------------------------------------


def test_existing_players_keep_their_ids(pipeline):
    resolver = pipeline["resolver"]
    for record in EXISTING_PLAYERS:
        vs = set(vertices(record))
        for v in vs:
            assert vs <= (object_of(resolver, v) or set()), f"{record['meta']['player_name']}: {v} lost its existing links"


def test_conflicting_proposals_are_rejected_not_applied(pipeline):
    # oma-usa pairs Chris Richards' provider_a ID with his provider_b profile, and usa-mex with a duplicate provider_b
    # profile. Both proposals are resolved at the same time and conflict only with each other, so the resolver can't tell
    # which is right and rejects both.
    resolver = pipeline["resolver"]
    original = expected_ids(PEOPLE["chris_richards"])
    duplicate = ("provider_b_player_id", str(DUPLICATE_PROFILE_ID))
    for v in original | {duplicate}:
        assert v not in resolver, f"{v} was applied"

    provider_a = ("provider_a_player_id", fixture_id(PEOPLE["chris_richards"], "provider_a"))
    rejections = [(ids, conflicts) for ids, conflicts in rejected_ids(pipeline) if provider_a in ids]
    assert len(rejections) == 2
    rejected = [ids for ids, _ in rejections]
    assert original in rejected
    assert any(duplicate in ids for ids in rejected)
    # each lists the other as its conflict
    for ids, conflicts in rejections:
        assert [set(vertices(c)) for c in conflicts] == [other for other in rejected if other != ids]


def test_proposal_contradicting_an_override_does_not_replace_the_override(pipeline):
    resolver = pipeline["resolver"]
    person = PEOPLE["bong_soo_kim"]
    override = {("provider_a_player_id", fixture_id(person, "provider_a")), ("provider_b_player_id", OVERRIDE_ID)}
    assert_one_player(pipeline, override)

    proposed = expected_ids(person)
    assert any(ids == proposed for ids, _ in rejected_ids(pipeline))
    assert ("provider_b_player_id", fixture_id(person, "provider_b")) not in resolver


def test_no_provider_id_is_on_two_players(pipeline):
    for ids, _ in pipeline["resolver"].components():
        assert len(ids) == len(set(ids)), f"two IDs from one provider: {ids}"
    # and no ID is in two objects
    seen: dict[Vertex, Any] = {}
    resolver = pipeline["resolver"]
    for v in resolver.parent:
        root = resolver.find(v)
        assert seen.setdefault(v, root) == root


def test_every_fixture_id_is_resolved_or_rejected(pipeline):
    resolver = pipeline["resolver"]
    rejected = set().union(*[ids for ids, _ in rejected_ids(pipeline)])
    lost = []
    for file in MATCH_FILES:
        data = with_duplicate_profile(file, FIXTURES[file].copy())
        for row in data.itertuples():
            v = (f"{row.data_provider}_player_id", str(row.provider_player_id))
            if v not in resolver and v not in rejected:
                lost.append((file, row.player_name, v))
    assert not lost, f"neither resolved nor rejected: {lost}"


# ---------------------------------------------------------------------------------------------------------------------
# Name-only provider
# ---------------------------------------------------------------------------------------------------------------------


def test_name_only_profile_with_accents_links_to_existing_player(pipeline):
    assert_one_player(pipeline, expected_ids(PEOPLE["sergino_dest"]) | {("provider_d_player_id", "401")})


def test_name_only_profile_links_to_existing_player(pipeline):
    assert_one_player(pipeline, expected_ids(PEOPLE["naomi_girma"]) | {("provider_d_player_id", "403")})


@pytest.mark.xfail(
    strict=True,
    reason="a name-only profile has no birth date or jersey number, and neither name's parts are a subset of the "
    "other's (Layer 4), so a maiden-name profile stays a separate player",
)
def test_name_only_profile_under_maiden_name_links_to_existing_player(pipeline):
    # "Makenzy Doniak" vs "Makenzy Robbe", the name the resolved player carries
    assert_one_player(pipeline, expected_ids(PEOPLE["makenzy_robbe"]) | {("provider_d_player_id", "402")})


@pytest.mark.parametrize("profile", ["411", "412", "413"])
def test_unmatched_name_only_profile_is_added_alone(pipeline, profile):
    # 411 shares only a first name with Tim Ream; the team-only layer is off, so nothing else can link them
    assert_one_player(pipeline, {("provider_d_player_id", profile)})


# ---------------------------------------------------------------------------------------------------------------------
# Known gaps
#
# These run the pipeline a second time on the fixtures plus the rows below, so the tests above keep their data.
# ---------------------------------------------------------------------------------------------------------------------

# two different players that only provider_b lists, with the same name, on the same team, and no birth date. The name
# is a former US player's who isn't in any fixture, and the match ID isn't a fixture's, so the rows can't be mistaken
# for fixture data.
SAME_NAME_ROWS = [
    {"match_id": "decoy-same-name", "provider_player_id": 9900801, "jersey_number": "98", "player_name": "Carli Lloyd",
     "player_nickname": "Carli Lloyd", "birth_date": None, "team_id": "SD", "player_gender": "female", "data_provider": "provider_b"},
    {"match_id": "decoy-same-name", "provider_player_id": 9900802, "jersey_number": "99", "player_name": "Carli Lloyd",
     "player_nickname": "Carli Lloyd", "birth_date": None, "team_id": "SD", "player_gender": "female", "data_provider": "provider_b"},
]


def with_same_name_players(file: str, data: pd.DataFrame) -> pd.DataFrame:
    data = with_duplicate_profile(file, data)
    if file == CHI_SD:
        data = pd.concat([data, pd.DataFrame(SAME_NAME_ROWS)], ignore_index=True)
    return data


@pytest.fixture(scope="module")
def gap_pipeline() -> dict[str, Any]:
    return run_pipeline(with_same_name_players)


@pytest.mark.xfail(
    strict=True,
    reason="SyncEngine.synchronize groups the result by join_columns (team_id, player_name) and keeps the first ID per "
    "provider, so a second player with the same name loses their ID",
)
def test_same_name_new_players_are_resolved_or_rejected(gap_pipeline):
    resolver = gap_pipeline["resolver"]
    rejected = set().union(*[ids for ids, _ in rejected_ids(gap_pipeline)])
    lost = [
        r["provider_player_id"]
        for r in SAME_NAME_ROWS
        if (v := ("provider_b_player_id", str(r["provider_player_id"]))) not in resolver and v not in rejected
    ]
    assert not lost, f"neither resolved nor rejected: {lost}"
