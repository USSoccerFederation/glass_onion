from __future__ import annotations
import pandas as pd
from glass_onion.engine import SyncableContent, SyncEngine


class TeamSyncableContent(SyncableContent):
    """
    A subclass of SyncableContent to use for team objects.
    """

    def __init__(self, provider: str, data: pd.DataFrame):
        super().__init__("team", provider, data)


class TeamSyncEngine(SyncEngine):
    """
    A subclass of SyncEngine to use for team objects.

    See `synchronize_pair()`[glass_onion.team.TeamSyncEngine.synchronize_pair] for methodology details.
    """

    def __init__(
        self,
        content: list[SyncableContent],
        use_competition_context: bool = False,
        verbose: bool = False,
    ):
        """
        Creates a new TeamSyncEngine object. Setting `use_competition_context` adds `competition_id` and `season_id` (assumed to be universal across all data providers) to `join_columns`.

        Args:
            content (list[str]): a list of SyncableContent objects.
            use_competition_context (bool): should the competition context (IE: columns `competition_id` and `season_id`) be used to synchronize team names?
            verbose (bool): a flag to verbose logging. This will be `extremely` verbose, allowing new SyncEngine developers and those integrating SyncEngine into their workflows to see the interactions between different logical layers during synchronization.
        """
        super().__init__(
            "team",
            content,
            ["team_name", "competition_id", "season_id"]
            if use_competition_context
            else ["team_name"],
            verbose,
        )

    def synchronize_pair(
        self, input1: SyncableContent, input2: SyncableContent
    ) -> SyncableContent:
        """
        Synchronizes two TeamSyncableContent objects.

        Methodology:
            1. Attempt to join pair simply on `team_name`.
            2. With remaining records, attempt to match via cosine similarity using a minimum threshold of 75% similarity.
            3. For any remaining records, attempt to match via cosine similarity using no minimum similarity threshold.

        Args:
            input1 (glass_onion.SyncableContent): a TeamSyncableContent object from `TeamSyncEngine.content`
            input2 (glass_onion.SyncableContent): a TeamSyncableContent object from `TeamSyncEngine.content`

        Returns:
            If `input1`'s underlying `data` dataframe is empty, returns `input2` with a column in `input2.data` for `input1.id_field`.
            If `input2`'s underlying `data` dataframe is empty, returns `input1` with a column in `input1.data` for `input2.id_field`.
            If both dataframes are non-empty, returns a new TeamSyncableContent object with synchronized identifiers from `input1` and `input2`.
        """
        if len(input1.data) == 0 and len(input2.data) > 0:
            input2.data[input1.id_field] = pd.NA
            return input2

        if len(input1.data) > 0 and len(input2.data) == 0:
            input1.data[input2.id_field] = pd.NA
            return input1

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
            cosine_results = self.synchronize_with_cosine_similarity(
                remaining_1,
                remaining_2,
                fields=("team_name", "team_name"),
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
                    # update remainders check
                    synced = sync_result.dropna(
                        subset=[input1.id_field, input2.id_field]
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
            cosine_results = self.synchronize_with_cosine_similarity(
                remaining_1,
                remaining_2,
                fields=("team_name", "team_name"),
                threshold=0.0,
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
