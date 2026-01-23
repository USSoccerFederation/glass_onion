from __future__ import annotations
import pandas as pd
import pandera.pandas as pa
from pandera import Field, Column
from pandera.typing import Series
from typing import Optional

from glass_onion.engine import SyncableContent, SyncEngine
from glass_onion.utils import dataframe_coalesce


class TeamDataSchema(pa.DataFrameModel):
    """
    A panderas.DataFrameModel for team information.

    Provider-specific team identifier fields are added before validation during [TeamSyncableContent.validate_data_schema()][glass_onion.team.TeamSyncableContent.validate_data_schema].

    `competition_id` and `season_id` must be provided when using `TeamSyncEngine.use_competition_context`.
    """

    team_name: Series[str] = Field(nullable=False)
    """
    The name of the team.
    """
    competition_id: Optional[Series[str]] = Field(nullable=False)
    """
    The competition the team is competing in. This is assumed to be universally unique across the [TeamSyncableContent][glass_onion.team.TeamSyncableContent] objects provided to [TeamSyncEngine][glass_onion.team.TeamSyncEngine].
    """
    season_id: Optional[Series[str]] = Field(nullable=False)
    """
    The season of the competition that the team is competing in. This is assumed to be universally unique across the [TeamSyncableContent][glass_onion.team.TeamSyncableContent] objects provided to [TeamSyncEngine][glass_onion.team.TeamSyncEngine].
    """


class TeamSyncableContent(SyncableContent):
    """
    A subclass of SyncableContent to use for team objects.
    """

    def __init__(self, provider: str, data: pd.DataFrame):
        super().__init__("team", provider, data)

    def validate_data_schema(self) -> bool:
        """
        Checks if this object's `data` meets the schema requirements for this object type. See [TeamDataSchema][glass_onion.team.TeamDataSchema] for more details.

        Raises:
            pandera.errors.SchemaError: if `data` does not conform to the schema.

        Returns:
            True, if `data` is formatted properly.
        """
        (
            TeamDataSchema.to_schema()
            .add_columns(
                {f"{self.id_field}": Column(str, required=True, nullable=False)}
            )
            .validate(self.data)
        )
        return super().validate_data_schema()


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
            use_competition_context (bool): should the competition context (IE: columns `competition_id` and `season_id`) be used to synchronize/aggregate team identifiers?
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

        if use_competition_context:
            comp_schema = TeamDataSchema.to_schema().update_columns(
                {
                    "competition_id": {"required": True, "nullable": False},
                    "season_id": {"required": True, "nullable": False},
                }
            )
            assert [comp_schema.validate(d.data) for d in self.content]

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
                    dataframe_coalesce(sync_result, [input2.id_field])

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

                    dataframe_coalesce(sync_result, [input2.id_field])

        return TeamSyncableContent(input1.provider, data=sync_result)
