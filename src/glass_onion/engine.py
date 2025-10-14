from functools import reduce
from datetime import datetime
import pandas as pd


class SyncableContent:
    def __init__(self, data_type: str, provider: str, data: pd.DataFrame):
        self.data_type = data_type
        self.provider = provider
        self.id_field = f"{provider}_{data_type}_id"
        self.data = data

    def merge(self, right: "SyncableContent") -> "SyncableContent":
        id_mask = right.data.columns[
            right.data.columns.str.contains(f"_{self.data_type}_id")
        ]
        merged = pd.merge(self.data, right.data[id_mask], how="left", on=right.id_field)

        return SyncableContent(
            data_type=self.data_type, provider=self.provider, data=merged
        )

    def transform_provider_fields(self) -> pd.DataFrame:
        provider_data_field = f"provider_{self.data_type}_id"
        if (
            "data_provider" in self.data.columns or "provider" in self.data.columns
        ) and (provider_data_field in self.data.columns):
            self.data.rename({provider_data_field: self.id_field}, axis=1, inplace=True)
            provider_columns = self.data.columns[
                self.data.columns.isin(["data_provider", "provider"])
            ]
            self.data.drop(provider_columns, axis=1, inplace=True)


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

    def calculate_sync_scores(self, content: SyncableContent) -> pd.DataFrame:
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
                synced.data = pd.concat(
                    [synced.data, remainders_result.data], axis=0, ignore_index=True
                )

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
            synced.data = pd.concat(
                [synced.data, remainder_df], axis=0, ignore_index=True
            )

        self.verbose_log(f"Pre-deduplication: Found {len(synced.data)} total rows")

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
