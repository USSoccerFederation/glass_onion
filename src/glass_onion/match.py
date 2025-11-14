import pandas as pd
from glass_onion.engine import SyncableContent, SyncEngine


class MatchSyncableContent(SyncableContent):
    def __init__(self, provider: str, data: pd.DataFrame):
        super().__init__("match", provider, data)


class MatchSyncEngine(SyncEngine):
    def __init__(
        self,
        content: list[SyncableContent],
        use_competition_context: bool = False,
        verbose: bool = False,
    ):
        self.data_type = "match"
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
        self.use_competition_context = use_competition_context

    def synchronize_on_adjusted_dates(
        self,
        input1_remaining: SyncableContent,
        input2_remaining: SyncableContent,
        date_adjustment: pd.Timedelta,
    ) -> pd.DataFrame:
        self.verbose_log(
            f"Triggering date adjustment ({date_adjustment}) and sync for inputs {input1_remaining.provider} (length {len(input1_remaining.data)}) and {input2_remaining.provider} (length {len(input2_remaining.data)})"
        )
        input1_remaining_adj = input1_remaining.data.copy()
        input1_remaining_adj.match_date = (
            pd.to_datetime(input1_remaining_adj.match_date) + date_adjustment
        ).dt.strftime("%Y-%m-%d")

        stage_2 = pd.merge(
            left=input2_remaining.data,
            right=input1_remaining_adj,
            on=self.join_columns,
            how="inner",
        ).dropna(subset=[input1_remaining.id_field, input2_remaining.id_field])
        self.verbose_log(
            f"via date adjustment ({date_adjustment}), found {len(stage_2)} more synced rows"
        )

        return stage_2[[input1_remaining.id_field, input2_remaining.id_field]]

    def synchronize_on_matchday(
        self, input1_remaining: SyncableContent, input2_remaining: SyncableContent
    ) -> pd.DataFrame:
        temp_join_columns = [
            k.replace("match_date", "matchday") for k in self.join_columns
        ]
        stage_3 = pd.merge(
            left=input1_remaining.data,
            right=input2_remaining.data,
            on=temp_join_columns,
            how="inner",
        ).dropna(subset=[input1_remaining.id_field, input2_remaining.id_field])

        self.verbose_log(f"via matchday, found {len(stage_3)} more synced rows")
        return stage_3[[input1_remaining.id_field, input2_remaining.id_field]]

    def synchronize_pair(
        self, input1: SyncableContent, input2: SyncableContent
    ) -> SyncableContent:
        # first pass: dates are equal
        self.verbose_log(
            f"Attempting pair synchronization for inputs {input1.provider} (length {len(input1.data)}) and {input2.provider} (length {len(input2.data)})"
        )
        # self.verbose_log(input1.data.columns)
        # self.verbose_log(input2.data.columns)
        sync_result = pd.merge(
            input1.data, input2.data, on=self.join_columns, how="left"
        )
        synced = sync_result.dropna(subset=[input1.id_field, input2.id_field])

        # second pass: dates are off by [-2, 2]
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

                sync_result.loc[
                    (sync_result[f"{input2.id_field}_x"].isna()), f"{input2.id_field}_x"
                ] = sync_result.loc[
                    (sync_result[f"{input2.id_field}_x"].isna()), f"{input2.id_field}_y"
                ]
                sync_result.drop([f"{input2.id_field}_y"], axis=1, inplace=True)

                sync_result.rename(
                    {
                        f"{input2.id_field}_x": input2.id_field,
                    },
                    axis=1,
                    inplace=True,
                )

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
            result = self.synchronize_on_matchday(remaining_1, remaining_2)
            self.verbose_log(
                f"Via matchday pair synchronization for inputs, found {len(result)} new rows"
            )

            if len(result) > 0:
                sync_result = pd.merge(
                    sync_result, result, on=input1.id_field, how="left"
                )

                sync_result.loc[
                    (sync_result[f"{input2.id_field}_x"].isna()), f"{input2.id_field}_x"
                ] = sync_result.loc[
                    (sync_result[f"{input2.id_field}_x"].isna()), f"{input2.id_field}_y"
                ]
                sync_result.drop([f"{input2.id_field}_y"], axis=1, inplace=True)

                sync_result.rename(
                    {
                        f"{input2.id_field}_x": input2.id_field,
                    },
                    axis=1,
                    inplace=True,
                )

        return MatchSyncableContent(input1.provider, data=sync_result)
