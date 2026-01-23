import pandas as pd
import pandera.pandas as pa
from pandera import Field, Column
from pandera.typing import Series
from typing import Optional

from glass_onion.engine import SyncableContent, SyncEngine
from glass_onion.utils import dataframe_coalesce, dataframe_clean_merged_fields


class MatchDataSchema(pa.DataFrameModel):
    """
    A panderas.DataFrameModel for team information. 
    
    Provider-specific match identifier fields are added before validation during [MatchSyncableContent.validate_data_schema()][glass_onion.match.MatchSyncableContent.validate_data_schema].

    `competition_id` and `season_id` must be provided when using `MatchSyncEngine.use_competition_context`.
    """
    match_date: Series[str] = Field(nullable=False)
    """
    The date of the match.
    """
    home_team_id: Series[str] = Field(nullable=False)
    """
    The team identifier of the home team. This is assumed to be universally unique across the [MatchSyncableContent][glass_onion.match.MatchSyncableContent] objects provided to [MatchSyncEngine][glass_onion.match.MatchSyncEngine].
    """
    away_team_id: Series[str] = Field(nullable=False)
    """
    The team identifier of the away team. This is assumed to be universally unique across the [MatchSyncableContent][glass_onion.match.MatchSyncableContent] objects provided to [MatchSyncEngine][glass_onion.match.MatchSyncEngine].
    """
    competition_id: Optional[Series[str]] = Field(nullable=False)
    """
    The competition of the match. This is assumed to be universally unique across the [MatchSyncableContent][glass_onion.match.MatchSyncableContent] objects provided to [TeamSyncEngine][glass_onion.team.TeamSyncEngine].
    """
    season_id: Optional[Series[str]] = Field(nullable=False)
    """
    The season of the match. This is assumed to be universally unique across the [MatchSyncableContent][glass_onion.match.MatchSyncableContent] objects provided to [TeamSyncEngine][glass_onion.team.TeamSyncEngine].
    """

    @pa.check("match_date")
    def is_valid_yyyy_mm_dd_date(self, series: Series[str]) -> bool:
        return series.dropna().apply(lambda x: pd.Timestamp(x)).apply(lambda x: (x != pd.Timestamp(0))).all()


class MatchSyncableContent(SyncableContent):
    """
    A subclass of SyncableContent to use for match objects.
    """

    def __init__(self, provider: str, data: pd.DataFrame):
        super().__init__("match", provider, data)

    def validate_data_schema(self) -> bool:
        (
            MatchDataSchema.to_schema()
            .add_columns({f"{self.id_field}": Column(str, required=True, nullable=False)})
            .validate(self.data)
        )
        return super().validate_data_schema()


class MatchSyncEngine(SyncEngine):
    """
    A subclass of SyncEngine to use for match objects.

    See `synchronize_pair()`[glass_onion.player.MatchSyncEngine.synchronize_pair] for methodology details.
    """

    def __init__(
        self,
        content: list[SyncableContent],
        use_competition_context: bool = False,
        verbose: bool = False,
    ):
        """
        Creates a new MatchSyncEngine object. Setting `use_competition_context` adds `competition_id` and `season_id` (assumed to be universal across all data providers) to `join_columns`.

        Args:
            content (list[str]): a list of SyncableContent objects.
            use_competition_context (bool): should the competition context (IE: columns `competition_id` and `season_id`) be used to synchronize/aggregate match identifiers?
            verbose (bool): a flag to verbose logging. This will be `extremely` verbose, allowing new SyncEngine developers and those integrating SyncEngine into their workflows to see the interactions between different logical layers during synchronization.
        """
        self.object_type = "match"
        self.content = content
        self.join_columns = (
            [
                "match_date",
                "competition_id",
                "season_id",
                "home_team_id",
                "away_team_id",
            ]
            if use_competition_context
            else ["match_date", "home_team_id", "away_team_id"]
        )
        self.verbose = verbose

        if use_competition_context:
            comp_schema = (
                MatchDataSchema.to_schema()
                    .update_columns(
                        {
                            "competition_id": {"required": True, "nullable": False},
                            "season_id": {"required": True, "nullable": False}
                        }
                    )
            )
            assert [comp_schema.validate(d.data) for d in self.content]

    def synchronize_on_adjusted_dates(
        self,
        input1: SyncableContent,
        input2: SyncableContent,
        date_adjustment: pd.Timedelta,
    ) -> pd.DataFrame:
        """
        Synchronizes two MatchSyncableContent objects after adjusting the `match_date` field of `input1` by `date_adjustment`.

        Args:
            input1 (glass_onion.engine.SyncableContent): a SyncableContent object.
            input2 (glass_onion.engine.SyncableContent): a SyncableContent object.
            date_adjustment (`pandas.Timedelta`): a time period to adjust `match_date` by for this layer.
        Returns:
            a `pandas.DataFrame` object that contains synchronized identifier pairs from `input1` and `input2`. The available columns are the `id_field` values of `input1` and `input2`.
        """
        self.verbose_log(
            f"Triggering date adjustment ({date_adjustment}) and sync for inputs {input1.provider} (length {len(input1.data)}) and {input2.provider} (length {len(input2.data)})"
        )
        input1_remaining_adj = input1.data.copy()
        input1_remaining_adj.match_date = (
            pd.to_datetime(input1_remaining_adj.match_date) + date_adjustment
        ).dt.strftime("%Y-%m-%d")

        stage_2 = pd.merge(
            left=input2.data,
            right=input1_remaining_adj,
            on=self.join_columns,
            how="inner",
        ).dropna(subset=[input1.id_field, input2.id_field])
        self.verbose_log(
            f"via date adjustment ({date_adjustment}), found {len(stage_2)} more synced rows"
        )

        return stage_2[[input1.id_field, input2.id_field]]

    def synchronize_on_matchday(
        self, input1: SyncableContent, input2: SyncableContent
    ) -> pd.DataFrame:
        """
        Synchronizes two MatchSyncableContent objects using `matchday` instead of `match_date`.

        Args:
            input1 (glass_onion.engine.SyncableContent): a SyncableContent object.
            input2 (glass_onion.engine.SyncableContent): a SyncableContent object.

        Returns:
            a `pandas.DataFrame` object that contains synchronized identifier pairs from `input1` and `input2`. The available columns are the `id_field` values of `input1` and `input2`.
        """
        temp_join_columns = [
            k.replace("match_date", "matchday") for k in self.join_columns
        ]
        result = pd.merge(
            left=input1.data,
            right=input2.data,
            on=temp_join_columns,
            how="inner",
        ).dropna(subset=[input1.id_field, input2.id_field])

        self.verbose_log(f"via matchday, found {len(result)} more synced rows")
        return result[[input1.id_field, input2.id_field]]

    def synchronize_pair(
        self, input1: SyncableContent, input2: SyncableContent
    ) -> SyncableContent:
        """
        Synchronizes two MatchSyncableContent objects.

        Methodology:
            1. Attempt to join pair using `match_date`, `home_team_id`, and `away_team_id`.
            2. Account for matches with different dates across data providers (timezones, TV scheduling, etc) by adjusting `match_date` in one dataset in the pair by -3 to 3 days, then attempting synchronization using `match_date`, `home_team_id`, and `away_team_id` again. This process is then repeated for the other dataset in the pair.
            3. Account for matches postponed to a different date outside the [-3, 3] day range by attempting synchronization using `matchday`, `home_team_id`, and `away_team_id` (if `matchday` is available).

        Args:
            input1 (glass_onion.SyncableContent): a MatchSyncableContent object from `MatchSyncEngine.content`
            input2 (glass_onion.SyncableContent): a MatchSyncableContent object from `MatchSyncEngine.content`

        Returns:
            If `input1`'s underlying `data` dataframe is empty, returns `input2` with a column in `input2.data` for `input1.id_field`.
            If `input2`'s underlying `data` dataframe is empty, returns `input1` with a column in `input1.data` for `input2.id_field`.
            If both dataframes are non-empty, returns a new MatchSyncableContent object with synchronized identifiers from `input1` and `input2`.
        """
        if len(input1.data) == 0 and len(input2.data) > 0:
            input2.data[input1.id_field] = pd.NA
            return input2

        if len(input1.data) > 0 and len(input2.data) == 0:
            input1.data[input2.id_field] = pd.NA
            return input1

        # first pass: dates are equal
        self.verbose_log(
            f"Attempting pair synchronization for inputs {input1.provider} (length {len(input1.data)}) and {input2.provider} (length {len(input2.data)})"
        )

        sync_result = pd.merge(
            input1.data, input2.data, on=self.join_columns, how="left"
        )
        dataframe_coalesce(sync_result, [input1.id_field, input2.id_field])
        dataframe_clean_merged_fields(sync_result, self.join_columns + ["matchday"])

        synced = sync_result.dropna(subset=[input1.id_field, input2.id_field])

        # second pass: dates are off by [-3, 3]
        remaining_1 = MatchSyncableContent(
            input1.provider,
            input1.data[~(input1.data[input1.id_field].isin(synced[input1.id_field]))],
        )
        remaining_2 = MatchSyncableContent(
            input2.provider,
            input2.data[~(input2.data[input2.id_field].isin(synced[input2.id_field]))],
        )

        result = []
        if len(remaining_1.data) > 0 and len(remaining_2.data) > 0:
            self.verbose_log(
                f"Attempting date-adjusted pair synchronization for inputs {remaining_1.provider} (length {len(remaining_1.data)}) and {remaining_2.provider} (length {len(remaining_2.data)})"
            )
            for d in range(-3, 3):
                r = self.synchronize_on_adjusted_dates(
                    remaining_1, remaining_2, pd.Timedelta(d)
                )
                result.append(r)

            for d in range(-3, 3):
                r = self.synchronize_on_adjusted_dates(
                    remaining_2, remaining_1, pd.Timedelta(d)
                )
                result.append(r)

        if len(result) > 0:
            attempt_syncs = pd.concat(result, axis=0).dropna().drop_duplicates()
            self.verbose_log(
                f"Via date-adjusted pair synchronization for inputs, found {len(attempt_syncs)} new rows"
            )

            if len(attempt_syncs) > 0:
                sync_result = pd.merge(
                    sync_result, attempt_syncs, on=input1.id_field, how="left"
                )

                dataframe_coalesce(sync_result, [input2.id_field])

        # third pass: use matchday (if available) instead of match_date (for situations where the game was postponed or delayed to another date outside of the [-3, 3] range)
        synced = sync_result.dropna(subset=[input1.id_field, input2.id_field])
        remaining_1 = MatchSyncableContent(
            input1.provider,
            input1.data[~(input1.data[input1.id_field].isin(synced[input1.id_field]))],
        )
        remaining_2 = MatchSyncableContent(
            input2.provider,
            input2.data[~(input2.data[input2.id_field].isin(synced[input2.id_field]))],
        )
        if (
            len(remaining_1.data) > 0
            and len(remaining_2.data) > 0
            and "matchday" in remaining_1.data.columns
            and "matchday" in remaining_2.data.columns
        ):
            self.verbose_log(
                f"Attempting matchday pair synchronization for inputs {remaining_1.provider} (length {len(remaining_1.data)}) and {remaining_2.provider} (length {len(remaining_2.data)})"
            )
            result_df = self.synchronize_on_matchday(remaining_1, remaining_2)
            self.verbose_log(
                f"Via matchday pair synchronization for inputs, found {len(result_df)} new rows"
            )

            if len(result_df) > 0:
                sync_result = pd.merge(
                    sync_result, result_df, on=input1.id_field, how="left"
                )

                dataframe_coalesce(sync_result, [input2.id_field])

        return MatchSyncableContent(input1.provider, data=sync_result)
