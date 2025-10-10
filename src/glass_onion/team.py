import pandas as pd
from glass_onion.engine import SyncableContent, SyncEngine
from glass_onion.utils import (
    apply_cosine_similarity,
    series_normalize,
    series_remove_common_suffixes,
    series_remove_common_prefixes,
)


class TeamSyncableContent(SyncableContent):
    def __init__(self, provider: str, data: pd.DataFrame):
        super().__init__("team", provider, data)


class TeamSyncEngine(SyncEngine):
    def __init__(
        self,
        content: list[SyncableContent],
        use_competition_context: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            "team",
            content,
            ["team_name", "competition_id", "season_id"]
            if use_competition_context
            else ["team_name"],
            verbose,
        )

    def normalize_team_names(self, input: pd.Series) -> pd.Series:
        result = series_remove_common_suffixes(result)
        result = series_remove_common_prefixes(result)
        result = series_normalize(result)
        result = result.str.lower().str.strip()
        return result

    def synchronize_on_cosine_similarity(
        self,
        input1_remaining: SyncableContent,
        input2_remaining: SyncableContent,
        threshold: float = 0.75,
    ) -> pd.DataFrame:
        self.verbose_log(
            f"Attempting cosine-similarity pair synchronization for inputs {input1_remaining.provider} (length {len(input1_remaining.data)}) and {input2_remaining.provider} (length {len(input2_remaining.data)})"
        )

        input1_teams = input1_remaining.data["team_name"].reset_index(drop=True)
        input2_teams = input2_remaining.data["team_name"].reset_index(drop=True)
        self.verbose_log(input1_teams)
        self.verbose_log(input2_teams)

        match_results = apply_cosine_similarity(input1_teams, input2_teams)

        result = match_results.sort_values(by="similarity", ascending=False)
        result = result[result.similarity >= threshold]
        result["similarity_rank"] = result.groupby(["input1", "input2"])[
            "similarity"
        ].rank(method="dense", ascending=False)
        result = result[result.similarity_rank <= 1]

        composite = pd.merge(
            input1_remaining.data[["team_name", input1_remaining.id_field]],
            result,
            left_on="team_name",
            right_on="input1",
        )

        composite = pd.merge(
            composite,
            input2_remaining.data[["team_name", input2_remaining.id_field]],
            left_on="input2",
            right_on="team_name",
        )
        self.verbose_log(composite)

        return composite[[input1_remaining.id_field, input2_remaining.id_field]]

    def synchronize_pair(
        self, input1: SyncableContent, input2: SyncableContent
    ) -> SyncableContent:
        # first pass: names are equal
        self.verbose_log(
            f"Attempting pair synchronization for inputs {input1.provider} (length {len(input1.data)}) and {input2.provider} (length {len(input2.data)})"
        )
        sync_result = pd.merge(
            input1.data, input2.data, on=self.join_columns, how="left"
        )
        synced = sync_result.dropna(subset=[input1.id_field, input2.id_field])

        # second pass: cosine similarity
        remaining_1 = TeamSyncableContent(
            input1.provider,
            input1.data[~(input1.data[input1.id_field].isin(synced[input1.id_field]))],
        )
        remaining_2 = TeamSyncableContent(
            input2.provider,
            input2.data[~(input2.data[input2.id_field].isin(synced[input2.id_field]))],
        )

        if len(remaining_1.data) > 0 and len(remaining_2.data) > 0:
            cosine_results = self.synchronize_on_cosine_similarity(
                remaining_1, remaining_2
            )

            if len(cosine_results) > 0:
                attempt_syncs = cosine_results.dropna().drop_duplicates()
                self.verbose_log(
                    f"Via cosine-similarity pair synchronization for inputs, found {len(attempt_syncs)} new rows"
                )

                if len(attempt_syncs) > 0:
                    sync_result = pd.merge(
                        sync_result, attempt_syncs, on=input1.id_field, how="left"
                    )

                    sync_result.loc[
                        (sync_result[f"{input2.id_field}_x"].isna()),
                        f"{input2.id_field}_x",
                    ] = sync_result.loc[
                        (sync_result[f"{input2.id_field}_x"].isna()),
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

        # third pass: cosine similarity but we don't care about the threshold
        remaining_1 = TeamSyncableContent(
            input1.provider,
            input1.data[~(input1.data[input1.id_field].isin(synced[input1.id_field]))],
        )
        remaining_2 = TeamSyncableContent(
            input2.provider,
            input2.data[~(input2.data[input2.id_field].isin(synced[input2.id_field]))],
        )
        if len(remaining_1.data) > 0 and len(remaining_2.data) > 0:
            self.verbose_log(
                f"Attempting less-stringent cosine-similarity pair synchronization for inputs {remaining_1.provider} (length {len(remaining_1.data)}) and {remaining_2.provider} (length {len(remaining_2.data)})"
            )
            cosine_results = self.synchronize_on_cosine_similarity(
                remaining_1, remaining_2, threshold=0.0
            )

            if len(cosine_results) > 0:
                attempt_syncs = cosine_results.dropna().drop_duplicates()
                attempt_syncs = attempt_syncs[
                    [remaining_1.id_field, remaining_2.id_field]
                ]
                self.verbose_log(
                    f"Via less-stringent cosine-similarity pair synchronization for inputs, found {len(attempt_syncs)} new rows"
                )

                if len(attempt_syncs) > 0:
                    sync_result = pd.merge(
                        sync_result, attempt_syncs, on=input1.id_field, how="left"
                    )

                    sync_result.loc[
                        (sync_result[f"{input2.id_field}_x"].isna()),
                        f"{input2.id_field}_x",
                    ] = sync_result.loc[
                        (sync_result[f"{input2.id_field}_x"].isna()),
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

        return TeamSyncableContent(input1.provider, data=sync_result)
