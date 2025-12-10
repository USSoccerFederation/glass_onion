from functools import reduce
from datetime import datetime
from typing import Union
import pandas as pd


class SyncableContent:
    """
    The underlying unit of the synchronization logic. This class is just a wrapper for the dataframe being synchronized, providing some context on the object type (`data_type`) being synchronized and the provider from which the data is sourced.

    This class should be subclassed for each new object type: see [`PlayerSyncableContent`][`glass_onion.player.PlayerSyncableContent`] for an example.
    """
    def __init__(self, data_type: str, provider: str, data: pd.DataFrame):
        self.data_type = data_type
        self.provider = provider
        self.id_field = f"{provider}_{data_type}_id"
        self.data = data

    def merge(left: "SyncableContent", right: "SyncableContent") -> "SyncableContent":
        """
        Combine two SyncableContent objects into one by conducting a left-join on the underlying dataframes.

        Notes:
        - This operation is not in-place and produces a new SyncableContent object. 
        - This operation is not permitted on SyncableContent objects that do not use the same `data_type`.
        - This operation only moves fields from `right` that are identifiers of the same `data_type` as `left`. Example: if `left` has `data_type` player, the only fields merged from right will be those that contain `_player_id`. 
        - This operation's left-join is done using the `id_field` of `right`, which MUST exist in `left` for the operation to work. 

        Args:
            left (glass_onion.SyncableContent, required): a SyncableContent object.
            right (glass_onion.SyncableContent, required): a SyncableContent object.
        
        Returns:
            a new SyncableContent object that uses
            - the shared `data_type` of both parent objects
            - the `provider` of `left`
            - a combined `data` pandas.DataFrame that contains all columns from `right` that are identifiers of the same `data_type` as `left` + all columns from `left`.
        """
        assert left.data_type == right.data_type, f"Left `data_type` ({left.data_type}) does not match Right `data_type` ({right.data_type})."
        assert right.id_field in left.data.columns, f"Right `id_field` ({right.id_field}) not in Left `data` columns."

        id_mask = right.data.columns[
            right.data.columns.str.contains(f"_{left.data_type}_id")
        ]
        assert len(id_mask) > 0, f"Identifiers for left `data_type` ({left.data_type}) are not in any columns in Right."

        merged = pd.merge(left.data, right.data[id_mask], how="left", on=right.id_field)

        return SyncableContent(
            data_type=left.data_type, provider=left.provider, data=merged
        )

    def merge(self, right: "SyncableContent") -> "SyncableContent":
        """
        See [`SyncableContent.merge`][`glass_onion.engine.SyncableContent.merge`] for more details. 
        """
        return SyncableContent.merge(self, right)

    def transform_provider_fields(self):
        """
        In certain workflows, it may be useful to have dataframes that use a unified schema to store identifiers from different data providers.

        For `data_type` player, this schema might look something like
        - data_provider
        - provider_player_id
        - player_name
        - <other fields>

        This method cleans up dataframes shaped this way for use in other `SyncableContent` operations by
        - converting `provider_*_id` fields in the dataframe to use `SyncableContent.provider`.
        - removing the `data_provider` (or `provider`) column.

        This method is an in-place operation. If a `provider_*_id` field and a `data_provider`/`provider` method are found, the above cleaning steps will be applied. 
        If only one or neither are found, then no cleaning will be applied.

        Args:
            None

        Returns:
            None
        """
        provider_data_field = f"provider_{self.data_type}_id"
        if (
            "data_provider" in self.data.columns or "provider" in self.data.columns
        ) and (provider_data_field in self.data.columns):
            self.data.rename({provider_data_field: self.id_field}, axis=1, inplace=True)
            provider_columns = self.data.columns[
                self.data.columns.isin(["data_provider", "provider"])
            ]
            self.data.drop(provider_columns, axis=1, inplace=True)

    def append(left: "SyncableContent", right: Union["SyncableContent", pd.DataFrame]):
        """
        Combine two SyncableContent objects into one by appending all rows from `right` to the end of `left`.

        Notes:
        - This operation is in-place and does NOT produce a new SyncableContent object. This method simply returns the adjusted `left` object. 
        - If `right` is a `SyncableContent` object, the rows from its `data` dataframe are appended to the end of `left`'s `data` dataframe. If `right` is a `pandas.DataFrame` object, its own rows are appended to the end of `left`'s `data` dataframe.
        - This operation is not permitted on SyncableContent objects that do not use the same `data_type`.
        - If `right` is None, this method is a no-op.

        Args:
            left (glass_onion.SyncableContent, required): a SyncableContent object.
            right (glass_onion.SyncableContent OR pandas.DataFrame, required): a SyncableContent object or a pandas.DataFrame object.
        
        Returns:
            `left` but with a `data` pandas.DataFrame that contains all rows from `right` and `left`.
        """
        new_data = None
        if right is not None:
            if isinstance(right, SyncableContent):
                if left.data_type != right.data_type:
                    return

                new_data = right.data

            elif isinstance(right, pd.DataFrame):
                new_data = right

        if new_data is not None:
            left.data = pd.concat([left.data, new_data], axis=0, ignore_index=True)

        return left


    def append(self, right: Union["SyncableContent", pd.DataFrame]):
        """
        See [`SyncableContent.append`][`glass_onion.engine.SyncableContent.append`] for more details. 
        """
        return SyncableContent.append(self, right)


class SyncEngine:
    def __init__(
        self,
        data_type: str,
        content: list[SyncableContent],
        join_columns: list[str] = [],
        verbose: bool = False,
    ):
        self.content = content
        self.data_type = data_type
        self.verbose = verbose
        self.join_columns = join_columns

    def verbose_log(self, msg: str):
        if self.verbose:
            print(f"{datetime.now()}: {msg}")

    def synchronize_pair(
        self, input1: SyncableContent, input2: SyncableContent
    ) -> SyncableContent:
        raise NotImplementedError()

    def synchronize(self) -> SyncableContent:
        if len(self.content) == 0:
            return SyncableContent(self.data_type, "unknown", pd.DataFrame())

        if len(self.content) == 1:
            return self.content[0]

        # first pass: straight matching - approach: agglomerative
        self.verbose_log(
            f"Starting {self.data_type} synchronization across {len(self.content)} datasets"
        )

        self.verbose_log(f"Pass 1: agglomeration")
        results = []
        for i in range(0, len(self.content) - 1):
            x = self.content[i]
            y = self.content[i + 1]
            z = self.synchronize_pair(x, y)
            results.append(z)

        sync_result = reduce(lambda x, y: x.merge(y), results[1:], results[0])
        id_mask = list(map(lambda x: x.id_field, self.content))

        synced = SyncableContent(
            self.data_type,
            self.content[0].provider,
            sync_result.data.dropna(subset=id_mask),
        )

        self.verbose_log(
            f"Pass 1: Using {self.content[0].provider} as sync basis, found {len(sync_result.data)} total rows and {len(synced.data)} fully synced rows."
        )

        ## second pass: relate remainders to each other
        remainders = []
        for c in self.content:
            missing = c.data[~(c.data[c.id_field].isin(synced.data[c.id_field]))]

            if len(missing) > 0:
                self.verbose_log(
                    f"Pass 2: Aggregating {len(missing)} identified unsynced rows for {c.provider}"
                )
                remainders.append(SyncableContent(self.data_type, c.provider, missing))

        if len(remainders) > 1:
            self.verbose_log(
                f"Pass 2: Agglomeration on remaining unsynced rows across {len(remainders)} datasets"
            )
            rem_results = []
            for i in range(0, len(remainders) - 1):
                x = remainders[i]
                y = remainders[i + 1]
                z = self.synchronize_pair(x, y)
                rem_results.append(z)

            self.verbose_log([d.data for d in rem_results])
            remainders_result = reduce(
                lambda x, y: x.merge(y), rem_results[1:], rem_results[0]
            )
            rem_id_mask = list(map(lambda x: x.id_field, self.content))
            rem_id_mask = [
                x for x in rem_id_mask if x in remainders_result.data.columns
            ]
            remainders_result.data.dropna(subset=rem_id_mask, inplace=True)

            self.verbose_log(
                f"Pass 2: Using remainders as sync basis, found {len(remainders_result.data)} new fully synced rows."
            )
            if len(remainders_result.data) > 0:
                synced.append(remainders_result)

        ## third pass: add remainders to end
        remainders = []
        for c in self.content:
            r = c.data[~(c.data[c.id_field].isin(synced.data[c.id_field]))]
            if len(r) > 0:
                self.verbose_log(
                    f"Pass 3: Aggregating {len(r)} identified unsynced rows for {c.provider}"
                )
                applicable_columns = synced.data.columns[
                    synced.data.columns.isin(c.data.columns)
                ]
                missing_columns = synced.data.columns[
                    ~(synced.data.columns.isin(c.data.columns))
                ]
                r_appendable = r[applicable_columns]
                r_appendable[missing_columns] = pd.NA
                r_appendable["provider"] = c.provider
                print(r_appendable)
                remainders.append(r_appendable)

        if len(remainders) > 0:
            remainder_df = pd.concat(remainders, axis=0)
            self.verbose_log(f"Pass 3: Including {len(remainder_df)} unsynced rows")
            synced.append(remainder_df)

        self.verbose_log(
            f"Pre-deduplication: Found {len(synced.data)} total rows: {synced.data}"
        )

        # validation / dedupe step: final table should summarize on key matching columns to eliminate any duplicate rows
        dedupe_aggregation = {
            k: (lambda x: x[x.notna()].iloc[0] if len(x[x.notna()]) > 0 else pd.NA)
            for k in id_mask
        }
        synced.data = (
            synced.data.groupby(self.join_columns).agg(dedupe_aggregation).reset_index()
        )
        synced.data["provider"] = self.content[0].provider
        self.verbose_log(
            f"After deduplication: Found {len(synced.data)} total unique rows based on join_columns: {self.join_columns}"
        )
        return synced
