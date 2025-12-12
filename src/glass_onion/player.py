from enum import Enum
from functools import reduce
from itertools import product
import pandas as pd
from glass_onion.engine import SyncableContent, SyncEngine
from typing import Optional, Tuple


class PlayerSyncableContent(SyncableContent):
    """
    A subclass of SyncableContent to use for player objects.
    """

    def __init__(self, provider: str, data: pd.DataFrame):
        super().__init__("player", provider, data)


class PlayerSyncSimilarityMethod(Enum):
    """
    Used to calculate similarity between player name (or nickname) strings.
    """

    COSINE = "cosine_similarity"
    """
    Use cosine similarity via [synchronize_with_cosine_similarity][glass_onion.engine.SyncEngine.synchronize_with_cosine_similarity]. The methodology is explained at [https://unravelsports.com/post.html?id=2022-07-11-player-id-matching-system](https://unravelsports.com/post.html?id=2022-07-11-player-id-matching-system).
    """
    NAIVE = "naive"
    """
    Use 'naive' similarity via [synchronize_with_naive_match][glass_onion.engine.SyncEngine.synchronize_with_naive_match]. TL;DR: split the two player name strings on spaces into sets and consider the intersection of the two sets.
    """
    FUZZY = "fuzzy"
    """
    Use fuzzy similarity via [synchronize_with_fuzzy_match][glass_onion.engine.SyncEngine.synchronize_with_fuzzy_match], which is a wrapper around [`thefuzz.process()`](https://github.com/seatgeek/thefuzz/blob/master/thefuzz/process.py).
    """


class PlayerSyncLayer:
    """
    A helper class that encapsulates a set of synchronization options for player objects.
    """

    def __init__(
        self,
        title: str,
        match_methodology: PlayerSyncSimilarityMethod = PlayerSyncSimilarityMethod.COSINE,
        date_adjustment: Optional[pd.Timedelta] = pd.Timedelta(0),
        swap_birth_month_day: bool = False,
        input_fields: Tuple[str] = ("player_name", "player_name"),
        other_equal_fields: list[str] = ["birth_date", "team_id"],
        threshold: float = 0.75,
    ):
        """
        Creates a new synchronization layer object.

        Args:
            match_methodology (glass_onion.player.PlayerSyncSimilarityMethod): see [PlayerSyncSimilarityMethod][glass_onion.player.PlayerSyncSimilarityMethod] for options.
            date_adjustment (pandas.Timedelta): a time period to adjust `birth_date` by for this layer.
            swap_birth_month_day (bool): a flag for if this layer should swap birth day and month
            input_fields (Tuple[str]): a two-tuple containing the column names to use for player name similarity. Possible options for tuple values: `player_name`, `player_nickname`
            other_equal_fields (list[str]): a list of columns that must be equal between the two PlayerSyncableContent datasets in order for an identifier to be synchronized validly.
            threshold (float): the threshold to use for string similarity when match_methodology is PlayerSyncSimilarityMethod.COSINE` or PlayerSyncSimilarityMethod.FUZZY`.
        """
        self.title = title
        self.date_adjustment = date_adjustment
        self.swap_birth_month_day = swap_birth_month_day
        self.input_fields = input_fields
        self.other_equal_fields = other_equal_fields
        self.similarity_threshold = threshold
        self.match_methodology = match_methodology


class PlayerSyncEngine(SyncEngine):
    """
    A subclass of SyncEngine to use for player objects.

    See `synchronize_pair()`[glass_onion.player.PlayerSyncEngine.synchronize_pair] for methodology details.
    """

    def __init__(self, content: list[PlayerSyncableContent], verbose: bool = False):
        """
        Creates a new PlayerSyncEngine object.

        Unlike other SyncEngine subclasses, this subclass does not accept any explicit `join_columns`. Player synchronization is a more finicky task than team or match synchronization, and as a result, the columns used in synchronization have to be checked for null values and validity before synchronization can be attempted.
        In this case, the base set of columns is ["jersey_number", "team_id", "player_name"], but since some providers may not provide a player's jersey number, we remove this column so that the logic in `SyncEngine.synchronize()` does not use it to group, aggregate, and deduplicate results.

        Args:
            content (list[str]): a list of PlayerSyncableContent objects.
            verbose (bool): a flag to verbose logging. This will be `extremely` verbose, allowing new SyncEngine developers and those integrating SyncEngine into their workflows to see the interactions between different logical layers during synchronization.
        """
        join_cols = ["jersey_number", "team_id", "player_name"]
        super().__init__("player", content, join_cols, verbose)
        # check if jersey number is empty / unreliable
        for c in content:
            for j in join_cols:
                if j not in c.data.columns:
                    join_cols.remove(j)
                    self.verbose_log(
                        f"Removing column `{j}` from join logic because of issues with content from data provider {c.provider} does not include it"
                    )
                    break

                if len(c.data[c.data[j].notna()]) != len(c.data):
                    join_cols.remove(j)
                    self.verbose_log(
                        f"Removing column `{j}` from join logic because content from data provider {c.provider} does not have complete coverage"
                    )
                    break

        assert len(join_cols) > 0, "No join columns remaining to use for aggregation."
        self.join_columns = join_cols

    def synchronize_using_layer(
        self,
        input1: SyncableContent,
        input2: SyncableContent,
        layer: PlayerSyncLayer,
    ) -> pd.DataFrame:
        """
        Synchronizes two PlayerSyncableContent objects using options from `layer` and returns a `pandas.DataFrame` of synchronized identifiers.

        This method also prevents the propagation of duplicate synchronized identifiers by ensuring that the resulting dataframe only contains identifiers that are only used once per `input`.

        Args:
            input1 (glass_onion.player.PlayerSyncableContent): a PlayerSyncableContent object.
            input2 (glass_onion.player.PlayerSyncableContent): a PlayerSyncableContent object.
            layer (glass_onion.player.PlayerSyncLayer`): a PlayerSyncLayer` object.

        Returns:
            pandas.DataFrame: a `pandas.DataFrame` object that contains unique synchronized identifier pairs from `input1` and `input2`. The available columns are the `id_field` values of `input1` and `input2`.
        """
        self.verbose_log(
            f"Attempting strategy-based cosine-similarity pair synchronization for inputs {input1.provider} (length {len(input1.data)}) and {input2.provider} (length {len(input2.data)})"
        )
        self.verbose_log(
            f"Strategy: {layer.title}\n- match_methodology: {layer.match_methodology}\n- birth date adjustment: {layer.date_adjustment}\n- swapped birth month/day: {layer.swap_birth_month_day}\n- cosine-sim fields: {layer.input_fields}\n- other equal fields: {layer.other_equal_fields}"
        )
        self.verbose_log(f"Input 1 Columns: {input1.data.columns.to_list()}")
        self.verbose_log(f"Input 2 Columns: {input2.data.columns.to_list()}")

        if (
            layer.date_adjustment is not None
            and "birth_date" in input1.data.columns
            and "birth_date" in input2.data.columns
        ):
            date_format = "%Y-%d-%m" if layer.swap_birth_month_day else "%Y-%m-%d"
            input1.data.birth_date = (
                pd.to_datetime(input1.data.birth_date) + layer.date_adjustment
            ).dt.strftime(date_format)

        if layer.match_methodology == PlayerSyncSimilarityMethod.NAIVE:
            match_result = self.synchronize_with_naive_match(
                input1, input2, fields=layer.input_fields
            )
        elif layer.match_methodology == PlayerSyncSimilarityMethod.FUZZY:
            match_result = self.synchronize_with_fuzzy_match(
                input1,
                input2,
                fields=layer.input_fields,
                threshold=layer.similarity_threshold,
            )
        else:
            match_result = self.synchronize_with_cosine_similarity(
                input1,
                input2,
                input1_field=layer.input_fields[0],
                input2_field=layer.input_fields[1],
                threshold=layer.similarity_threshold,
            )

        actual_available_fields = (
            input1.data.columns[
                input1.data.columns.isin(layer.other_equal_fields)
            ].to_list()
            if len(layer.other_equal_fields) > 0
            else []
        )
        if len(actual_available_fields) > 0:
            match_result = pd.merge(
                match_result,
                input1.data[[input1.id_field] + actual_available_fields],
                how="inner",
                on=input1.id_field,
            ).rename({k: f"input1_{k}" for k in actual_available_fields}, axis=1)

            match_result = pd.merge(
                match_result,
                input2.data[[input2.id_field] + actual_available_fields],
                how="inner",
                on=input2.id_field,
            ).rename({k: f"input2_{k}" for k in actual_available_fields}, axis=1)

            for f in actual_available_fields:
                match_result = match_result[
                    match_result[f"input1_{f}"] == match_result[f"input2_{f}"]
                ]

        # edge case: detect whether one player has been matched to multiple other players (somehow)
        input1_summary = (
            match_result.groupby(input1.id_field)[input2.id_field]
            .nunique()
            .reset_index()
        )
        input2_summary = (
            match_result.groupby(input2.id_field)[input1.id_field]
            .nunique()
            .reset_index()
        )
        match_result = pd.merge(
            match_result,
            input1_summary.rename({input2.id_field: "input2_num"}, axis=1),
            how="left",
            on=input1.id_field,
        )

        match_result = pd.merge(
            match_result,
            input2_summary.rename({input1.id_field: "input1_num"}, axis=1),
            how="left",
            on=input2.id_field,
        )

        synced = match_result.loc[
            (match_result.input1_num == 1) & (match_result.input2_num == 1),
            [input1.id_field, input2.id_field],
        ]

        self.verbose_log(
            f"Using strategy-based cosine-similarity pair synchronization, found {len(synced)} new rows"
        )
        return synced

    def synchronize_pair(
        self, input1: SyncableContent, input2: SyncableContent
    ) -> SyncableContent:
        """
        Synchronizes two PlayerSyncableContent objects.

        Methodology:
            1. Attempt to join pair using `player_name` with a minimum 75% cosine similarity threshold for player name. Additionally, require that `jersey_number` and `team_id` are equal for matches that meet the similarity threshold.
            2. Account for players with different birth dates across data providers (timezones, human error, etc) by adjusting `birth_date` in one dataset in the pair by -1 to 1 days and/or swapping the month and day, then attempting synchronization using `birth_date`, `team_id`, and a combination of `player_name` and `player_nickname`. This process is then repeated for the other dataset in the pair.
            3. Attempt to join remaining records using combinations of `player_name` and `player_nickname` with a minimum 75% cosine similarity threshold for player name. Additionally, require that `team_id` is equal for matches that meet the similarity threshold.
            4. Attempt to join remaining records using "naive similarity": looking for normalized parts of one record's `player name` (or `player_nickname`) that exist in another's. Additionally, require that `team_id` is equal for matches found via this method.
            5. Attempt to join remaining records using combinations of `player_name` and `player_nickname` with no minimum cosine similarity threshold. Additionally, require that `team_id` is equal.

        Args:
            input1 (glass_onion.SyncableContent): a PlayerSyncableContent object from `PlayerSyncEngine.content`
            input2 (glass_onion.SyncableContent): a PlayerSyncableContent object from `PlayerSyncEngine.content`

        Returns:
            If `input1`'s underlying `data` dataframe is empty, returns `input2` with a column in `input2.data` for `input1.id_field`.
            If `input2`'s underlying `data` dataframe is empty, returns `input1` with a column in `input1.data` for `input2.id_field`.
            If both dataframes are non-empty, returns a new PlayerSyncableContent object with synchronized identifiers from `input1` and `input2`.
        """
        if len(input1.data) == 0 and len(input2.data) > 0:
            input2.data[input1.id_field] = pd.NA
            return input2

        if len(input1.data) > 0 and len(input2.data) == 0:
            input1.data[input2.id_field] = pd.NA
            return input1

        # first layer: cosine similarity x jersey number x team
        self.verbose_log(
            f"Attempting simple match on jersey/team for inputs {input1.provider} (length {len(input1.data)}) and {input2.provider} (length {len(input2.data)})"
        )
        sync_result = self.synchronize_using_layer(
            input1,
            input2,
            PlayerSyncLayer(
                title="Layer 1: cosine similarity x jersey number x team",
                date_adjustment=None,
                other_equal_fields=["jersey_number", "team_id"],
            ),
        )
        id_mask = input1.data.columns[input1.data.columns.str.endswith("_player_id")]
        sync_result = pd.merge(
            input1.data[
                ["player_name", "player_nickname", "jersey_number", "team_id"]
                + id_mask.to_list()
            ],
            sync_result,
            on=input1.id_field,
            how="left",
        )
        synced = sync_result.dropna(subset=[input1.id_field, input2.id_field])
        self.verbose_log(f"Using simple match on jersey/team, found {len(synced)} rows")

        # `itertools.product` can only be iterated once (https://stackoverflow.com/a/17557923), so turn this into a list
        input_field_options = list(
            product(
                ["player_name", "player_nickname"], ["player_name", "player_nickname"]
            )
        )

        # second layer: cosine similarity x birth date x team
        sync_strategies = []
        if (
            "birth_date" in input1.data.columns
            and len(input1.data[input1.data["birth_date"].notna()]) == len(input1.data)
            and "birth_date" in input2.data.columns
            and len(input2.data[input2.data["birth_date"].notna()]) == len(input2.data)
        ):
            birth_date_layers = []
            for p in input_field_options:
                birth_date_layers.append(
                    [
                        PlayerSyncLayer(
                            title="Layer 2: cosine similarity x birth date x team",
                            date_adjustment=pd.Timedelta(d),
                            input_fields=p,
                        )
                        for d in range(-1, 1)
                    ]
                )
                birth_date_layers.append(
                    [
                        PlayerSyncLayer(
                            title="Layer 2: cosine similarity x birth date x team",
                            date_adjustment=pd.Timedelta(d),
                            swap_birth_month_day=True,
                            input_fields=p,
                        )
                        for d in range(-1, 1)
                    ]
                )
            sync_strategies += reduce(lambda x, y: (x + y), birth_date_layers, [])
        else:
            self.verbose_log(
                "Skipping birth date matching strategies because `birth_date` field is not reliable"
            )

        # third layer: cosine similarity + team
        sync_strategies += [
            PlayerSyncLayer(
                title="Layer 3: cosine similarity x team",
                date_adjustment=None,
                input_fields=p,
                other_equal_fields=["team_id"],
            )
            for p in input_field_options
        ]

        # fourth layer: take remainders and split normalized strings into sets on whitespace. and check what strings fit into what sets
        sync_strategies += [
            PlayerSyncLayer(
                title="Layer 4: naive similarity x team",
                match_methodology="naive",
                date_adjustment=None,
                input_fields=p,
                other_equal_fields=["team_id"],
            )
            for p in input_field_options
        ]

        # fifth layer: simple jersey number/team match
        layer5_fields = ["jersey_number", "team_id"]
        layer5_title = "Layer 5: jersey number x team"
        if "jersey_number" not in self.join_columns:
            # jersey number is not reliable
            self.verbose_log(
                "Removing `jersey_number` from Layer 5 processing because it's been marked unreliable"
            )
            layer5_fields.remove("jersey_number")
            layer5_title = "Layer 5: team"

        sync_strategies += [
            PlayerSyncLayer(
                title=layer5_title,
                date_adjustment=None,
                other_equal_fields=layer5_fields,
                threshold=0,
            )
        ]

        self.verbose_log(
            f"Collected {len(sync_strategies)} possible sync strategies. Applying one by one until we run out of rows..."
        )
        for i, strat in enumerate(sync_strategies):
            self.verbose_log(
                f"Applying pair synchronization strategy {i}: {strat.title}"
            )
            remaining_1 = PlayerSyncableContent(
                input1.provider,
                input1.data[
                    ~(input1.data[input1.id_field].isin(synced[input1.id_field]))
                ],
            )
            remaining_2 = PlayerSyncableContent(
                input2.provider,
                input2.data[
                    ~(input2.data[input2.id_field].isin(synced[input2.id_field]))
                ],
            )
            if len(remaining_1.data) == 0 or len(remaining_2.data) == 0:
                self.verbose_log(f"No more data to synchronize -- bailing out.")
                continue

            attempt_syncs = self.synchronize_using_layer(
                remaining_1, remaining_2, strat
            )
            if len(attempt_syncs) > 0:
                sync_result = pd.merge(
                    sync_result, attempt_syncs, on=input1.id_field, how="left"
                )
                sync_result.loc[
                    sync_result[input1.id_field].isin(attempt_syncs[input1.id_field]),
                    f"{input2.id_field}_x",
                ] = sync_result.loc[
                    sync_result[input1.id_field].isin(attempt_syncs[input1.id_field]),
                    f"{input2.id_field}_y",
                ]
                sync_result.drop([f"{input2.id_field}_y"], axis=1, inplace=True)

                sync_result.rename(
                    {
                        f"{input2.id_field}_x": input2.id_field,
                    },
                    axis=1,
                    inplace=True,
                )
                synced = sync_result.dropna(subset=[input1.id_field, input2.id_field])

        final_result = PlayerSyncableContent(
            provider=input1.provider,
            data=sync_result.dropna(subset=[input1.id_field, input2.id_field]),
        )
        self.verbose_log(
            f"After all pair sync strategies, found {len(final_result.data)} unique synced rows"
        )
        return final_result
