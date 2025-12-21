from functools import reduce
from datetime import datetime
import re
from typing import Any, Tuple, Union
import pandas as pd
from thefuzz import process
from glass_onion.utils import apply_cosine_similarity, series_normalize


class SyncableContent:
    """
    The underlying unit of the synchronization logic. This class is just a wrapper for the dataframe being synchronized, providing some context on the object type (`object_type`) being synchronized and the provider from which the data is sourced.

    This class should be subclassed for each new object type: see [PlayerSyncableContent][glass_onion.player.PlayerSyncableContent] for an example.
    """

    def __init__(self, object_type: str, provider: str, data: pd.DataFrame):
        self.object_type = object_type
        self.provider = provider
        self.id_field = f"{provider}_{object_type}_id"
        self.data = data

        assert data is not None, f"Field `data` can not be null"
        assert self.id_field in self.data.columns, (
            f"Field `{self.id_field}` must be available as a column in `data`"
        )

    def merge(self, right: "SyncableContent") -> "SyncableContent":
        """
        Combine two SyncableContent objects into one by conducting a left-join on the underlying dataframes.

        Notes:

        * This operation is not in-place and produces a new SyncableContent object.
        * This operation is not permitted on SyncableContent objects that do not use the same `object_type`.
        * This operation only moves fields from `right` that are identifiers of the same `object_type` as `self`. Example: if `self` has `object_type` player, the only fields merged from right will be those that contain `_player_id`.
        * This operation's left-join is done using the `id_field` of `right`, which MUST exist in `self` for the operation to work.

        Args:
            right (glass_onion.SyncableContent): a SyncableContent object.

        Returns:
            a new object that contains the shared `object_type` of both parent objects, `self.provider`, and a combined `data` `pandas.DataFrame` that contains all columns from `right` that are identifiers of the same `object_type` as `self` + all columns from `self.data`.
        """
        if right is None:
            return self

        assert self.object_type == right.object_type, (
            f"Left `object_type` ({self.object_type}) does not match Right `object_type` ({right.object_type})."
        )
        assert right.id_field in self.data.columns, (
            f"Right `id_field` ({right.id_field}) not in Left `data` columns."
        )

        id_mask = right.data.columns[
            right.data.columns.str.contains(f"_{self.object_type}_id")
        ]

        merged = pd.merge(self.data, right.data[id_mask], how="left", on=right.id_field)

        return SyncableContent(
            object_type=self.object_type, provider=self.provider, data=merged
        )

    def append(
        self, right: Union["SyncableContent", pd.DataFrame]
    ) -> "SyncableContent":
        """
        Combine two SyncableContent objects into one by appending all rows from `right` to the end of `left`.

        Notes:

        * This operation is in-place and does NOT produce a new SyncableContent object. This method simply returns the adjusted `left` object.
        * If `right` is a SyncableContent object, the rows from its `data` dataframe are appended to the end of `left`'s `data` dataframe. If `right` is a `pandas.DataFrame` object, its own rows are appended to the end of `left`'s `data` dataframe.
        * This operation is not permitted on SyncableContent objects that do not use the same `object_type`.
        * If `right` is None, this method is a no-op.

        Args:
            right (glass_onion.SyncableContent OR `pandas.DataFrame`): a SyncableContent object or a pandas.DataFrame object.

        Returns:
            `self` but with a `data` pandas.DataFrame that contains all rows from `right` and `left`.
        """
        new_data = None
        if right is not None:
            if isinstance(right, SyncableContent):
                assert self.object_type == right.object_type, (
                    f"Left `object_type` ({self.object_type}) does not match Right `object_type` ({right.object_type})."
                )

                new_data = right.data

            elif isinstance(right, pd.DataFrame):
                new_data = right

        if new_data is not None:
            self.data = pd.concat([self.data, new_data], axis=0, ignore_index=True)

        return self


class SyncEngine:
    """
    A wrapper around an object type's synchronization process.

    Given a list of SyncableContent, a SyncEngine synchronizes one pair of objects at a time (via `SyncEngine.synchronize_pair()`). The results of all pairs are then merged together and deduplicated.
    Each object type corresponds to a subclass of SyncEngine that overrides `synchronize_pair()` to define how pairs are synchronized in `synchronize()`, which contains wrapper logic for the entire process.

    There are three distinct layers within `SyncEngine.synchronize()`'s wrapper logic:

    1. The aforementioned sync process that results in a data frame of synced identifiers.
    2. Collect remaining unsynced rows and run the sync process on those. Append any newly synced rows to the result dataframe from Layer 1.
    3. Append any remaining unsynced rows to the bottom of the result data frame.

    This result dataframe is then deduplicated: by default, the result dataframe is grouped by the specific columns defined in SyncEngine and the first non-null result is selected for each data provider's identifier field.

    This class should be subclassed for each new object type: see [PlayerSyncEngine][glass_onion.player.PlayerSyncEngine] for an example.
    """

    def __init__(
        self,
        object_type: str,
        content: list[SyncableContent],
        join_columns: list[str],
        verbose: bool = False,
    ):
        """
        Create a new SyncEngine object.

        Args:
            object_type (str): the object type this SyncEngine is working with.
            content (list[SyncableContent]): a list of SyncableContent objects that correspond to `object_type`.
            join_columns (list[str]): a list of columns used to aggregate and deduplicate identifiers. In some subclasses, these columns are used to do an initial, naive synchronization pass before moving on to more complex checks.
            verbose (bool, optional): a flag to verbose logging. This will be `extremely` verbose, allowing new SyncEngine developers and those integrating SyncEngine into their workflows to see the interactions between different logical layers during synchronization.
        """
        assert isinstance(content, list), (
            "`content` must be a list of SyncableContent objects."
        )
        assert len(content) > 0, "`content` can not be empty"
        assert all([isinstance(c, SyncableContent) for c in content]), (
            "One or more objects in `content` are not `SyncableContent` objects."
        )

        assert object_type is not None, "`object_type` can not be NULL"
        assert len(object_type.strip()) > 0, (
            "`object_type` can not be empty or just whitespace"
        )

        assert all([c.object_type == object_type for c in content]), (
            "One or more `SyncableContent` objects in `content` do not match `SyncEngine.object_type`."
        )

        self.content = content
        self.object_type = object_type
        self.verbose = verbose
        self.join_columns = join_columns

    def verbose_log(self, msg: Any):
        """
        Helper method to enable verbose logging via `print()`. These logs are sent to `stdout` (the default output location of `print()`). Logs are also prefixed with a timestamp for easy sorting.

        Args:
            msg (Any): any string-serializable object
        """
        if self.verbose:
            print(f"{datetime.now()}: {msg}")

    def synchronize_with_fuzzy_match(
        self,
        input1: SyncableContent,
        input2: SyncableContent,
        fields: Tuple[str, str],
        threshold: float = 0.90,
    ):
        """
        Synchronizes two SyncableContent objects using fuzzy matching similarity using the columns provided by the two-tuple `fields`.

        Index 0 of `fields` is the column to use for similarity in `input1`, while index 1 is the column to use in `input2`.

        NOTE: this approach uses a dictionary/map, so a string from `input1` can only be mapped to one string in `input2`.
        If there are duplicate instances of a string in `input1` under a different ID, the 2nd...Nth instances of that string will not get matched.

        See [thefuzz.process()](https://github.com/seatgeek/thefuzz/blob/master/thefuzz/process.py) for more details.

        Args:
            input1 (glass_onion.engine.SyncableContent): a SyncableContent object.
            input2 (glass_onion.engine.SyncableContent): a SyncableContent object.
            fields (Tuple[str, str]): a two-tuple containing the column names to use for player name similarity.
            threshold (float): the minimum similarity threshold that a match must be in order to be considered valid. Options: any float value from 0.0 to 1.0. Options outside this range will be clamped to the min or max value.

        Returns:
            (pandas.DataFrame): a pandas.DataFrame object that contains synchronized identifier pairs from `input1` and `input2`. The available columns are the `id_field` values of `input1` and `input2`.
        """

        assert len(fields) == 2, (
            "Must provide two columns (one from `input1` and one from `input2`) as `fields`."
        )
        assert fields[0] in input1.data.columns, (
            "First element of `fields` must exist in `input1.data`."
        )
        assert fields[1] in input2.data.columns, (
            "Second element of `fields` must exist in `input2.data`."
        )
        assert len(input1.data) > 0 and len(input2.data) > 0, (
            "Both SyncableContent objects must be non-empty."
        )

        name_population = input1.data[fields[0]]
        name_sample = input2.data[fields[1]]

        name_population = input1.data.loc[
            input1.data[fields[0]].notna(), [fields[0], input1.id_field]
        ]
        name_sample = input2.data.loc[
            input2.data[fields[1]].notna(), [fields[1], input2.id_field]
        ]

        assert len(name_population) > 0 and len(name_sample) > 0, (
            "Both SyncableContent objects must have > 0 non-null elements in `data`."
        )

        normalized_name_population = series_normalize(name_population[fields[0]])
        normalized_name_sample = series_normalize(name_sample[fields[1]])

        adjusted_threshold = max(min(threshold, 1.0), 0.0) * 100

        results = []
        name_map: dict[str, str] = {}
        for j in range(0, len(normalized_name_sample)):
            i2_raw = normalized_name_sample.loc[normalized_name_sample.index[j]]
            if i2_raw in name_map.values():
                continue

            result = process.extractOne(i2_raw, normalized_name_population)
            if result and result[1] >= adjusted_threshold:
                if result[0] in name_map.keys():
                    continue

                self.verbose_log(f"Logging match: {result[0]} -> {i2_raw}")
                name_map[result[0]] = i2_raw
                i = result[2]
                results.append(
                    {
                        f"{input1.id_field}": name_population.loc[i, input1.id_field],
                        f"{input2.id_field}": name_sample.loc[
                            name_sample.index[j], input2.id_field
                        ],
                    }
                )
            elif result and result[1] < adjusted_threshold:
                self.verbose_log(
                    f"not match: {result[0]} -/-> {i2_raw} (similarity: {result[1]} < ({adjusted_threshold}))"
                )

        if len(results) == 0:
            return pd.DataFrame(data=[], columns=[input1.id_field, input2.id_field])

        return pd.DataFrame(results)

    def synchronize_with_cosine_similarity(
        self,
        input1: SyncableContent,
        input2: SyncableContent,
        fields: Tuple[str, str],
        threshold: float = 0.75,
    ) -> pd.DataFrame:
        """
        Synchronizes two SyncableContent objects using cosine similarity and the columns provided by the two-tuple `fields`.

        Index 0 of `fields` is the column to use for similarity in `input1`, while index 1 is the column to use in `input2`.

        See [apply_cosine_similarity][glass_onion.utils.apply_cosine_similarity] for more details on implementation.

        Args:
            input1 (glass_onion.engine.SyncableContent): a SyncableContent object.
            input2 (glass_onion.engine.SyncableContent): a SyncableContent object.
            fields (Tuple[str, str]): a two-tuple containing the column names to use for player name similarity.
            threshold (float): the minimum similarity threshold that a match must be in order to be considered valid. Options: any float value from 0.0 to 1.0. Options outside this range will be clamped to the min or max value.

        Returns:
            pandas.DataFrame: contains unique synchronized identifier pairs from `input1` and `input2`. The available columns are the `id_field` values of `input1` and `input2`.
        """

        assert len(fields) == 2, (
            "Must provide two columns (one from `input1` and one from `input2`) as `fields`."
        )
        assert fields[0] in input1.data.columns, (
            "First element of `fields` must exist in `input1.data`."
        )
        assert fields[1] in input2.data.columns, (
            "Second element of `fields` must exist in `input2.data`."
        )
        assert len(input1.data) > 0 and len(input2.data) > 0, (
            "Both SyncableContent objects must be non-empty."
        )

        input1_fields = input1.data.loc[
            input1.data[fields[0]].notna(), fields[0]
        ].reset_index(drop=True)
        input2_fields = input2.data.loc[
            input2.data[fields[1]].notna(), fields[1]
        ].reset_index(drop=True)

        assert len(input1_fields) > 0 and len(input2_fields) > 0, (
            "Both SyncableContent objects must have > 0 non-null elements in `data`."
        )

        adjusted_threshold = max(min(threshold, 1.0), 0.0)

        match_results = apply_cosine_similarity(input1_fields, input2_fields)

        result = match_results.sort_values(by="similarity", ascending=False)
        result = result[result.similarity >= adjusted_threshold]
        result["similarity_rank"] = result.groupby(["input1", "input2"])[
            "similarity"
        ].rank(method="dense", ascending=False)
        result = result[result.similarity_rank <= 1]

        composite = pd.merge(
            input1.data[[fields[0], input1.id_field]],
            result,
            left_on=fields[0],
            right_on="input1",
            how="inner",
        )

        composite = pd.merge(
            composite,
            input2.data[[fields[1], input2.id_field]],
            left_on="input2",
            right_on=fields[1],
            how="inner",
        )

        # deduplicate if there are duplicate name matches.
        # Must use rank(method="first") in case `similarity` is the same.
        composite["field_rank"] = composite.groupby(input1.id_field)["similarity"].rank(
            method="first", ascending=False
        )
        return composite.loc[
            composite["field_rank"] == 1, [input1.id_field, input2.id_field]
        ]

    def synchronize_with_naive_match(
        self, input1: SyncableContent, input2: SyncableContent, fields: Tuple[str, str]
    ) -> pd.DataFrame:
        """
        Synchronizes two SyncableContent objects using the `naive` similarity using the columns provided by the two-tuple `fields`.

        Index 0 of `fields` is the column to use for similarity in `input1`, while index 1 is the column to use in `input2`.

        NOTE: this is a _naive_ approach that uses a dictionary/map, so a string from `input1` can only be mapped to one string in `input2`.
        If there are duplicate instances of a string in `input1` under a different ID, the 2nd...Nth instances of that string will not get matched.

        Args:
            input1 (glass_onion.engine.SyncableContent): a SyncableContent object.
            input2 (glass_onion.engine.SyncableContent): a SyncableContent object.
            fields (Tuple[str, str]): a two-tuple containing the column names to use for player name similarity.

        Returns:
            pandas.DataFrame: contains synchronized identifier pairs from `input1` and `input2`. The available columns are the `id_field` values of `input1` and `input2`.
        """

        assert len(fields) == 2, (
            "Must provide two columns (one from `input1` and one from `input2`) as `fields`."
        )
        assert fields[0] in input1.data.columns, (
            "First element of `fields` must exist in `input1.data`."
        )
        assert fields[1] in input2.data.columns, (
            "Second element of `fields` must exist in `input2.data`."
        )

        assert len(input1.data) > 0 and len(input2.data) > 0, (
            "Both SyncableContent objects must be non-empty."
        )

        name_population = input1.data.loc[
            input1.data[fields[0]].notna(), [fields[0], input1.id_field]
        ]
        name_sample = input2.data.loc[
            input2.data[fields[1]].notna(), [fields[1], input2.id_field]
        ]

        assert len(name_population) > 0 and len(name_sample) > 0, (
            "Both SyncableContent objects must have > 0 non-null elements in `data`."
        )

        normalized_name_population = series_normalize(name_population[fields[0]])
        normalized_name_sample = series_normalize(name_sample[fields[1]])

        results = []
        name_map: dict[str, str] = {}

        # first pass to encapsulate exact matches
        for i in range(0, len(normalized_name_population)):
            i1_raw = normalized_name_population.loc[normalized_name_population.index[i]]
            if i1_raw in name_map.keys():
                continue

            for j in range(0, len(normalized_name_sample)):
                i2_raw = normalized_name_sample.loc[normalized_name_sample.index[j]]
                if i2_raw in name_map.values():
                    continue

                if i1_raw == i2_raw:
                    # this is a match
                    self.verbose_log(f"Logging match: {i1_raw} -> {i2_raw}")
                    name_map[i1_raw] = i2_raw
                    results.append(
                        {
                            f"{input1.id_field}": name_population.loc[
                                name_population.index[i], input1.id_field
                            ],
                            f"{input2.id_field}": name_sample.loc[
                                name_sample.index[j], input2.id_field
                            ],
                        }
                    )

        for i in range(0, len(normalized_name_population)):
            i1_raw = normalized_name_population.loc[normalized_name_population.index[i]]
            if i1_raw in name_map.keys():
                continue

            for j in range(0, len(normalized_name_sample)):
                i2_raw = normalized_name_sample.loc[normalized_name_sample.index[j]]
                if i2_raw in name_map.values():
                    continue

                i1_set = set(re.split(r"\s+", i1_raw))
                self.verbose_log(f"Input1 {i}: {i1_raw} -> {i1_set}")
                i2_set = set(re.split(r"\s+", i2_raw))
                self.verbose_log(f"Input2 {j}: {i2_raw} -> {i2_set}")

                if len(i1_set.intersection(i2_set)) == len(i2_set) or len(
                    i2_set.intersection(i1_set)
                ) == len(i1_set):
                    # this is a match
                    self.verbose_log(f"Logging match: {i1_raw} -> {i2_raw}")
                    name_map[i1_raw] = i2_raw
                    results.append(
                        {
                            f"{input1.id_field}": name_population.loc[
                                name_population.index[i], input1.id_field
                            ],
                            f"{input2.id_field}": name_sample.loc[
                                name_sample.index[j], input2.id_field
                            ],
                        }
                    )
                    break

        if len(results) == 0:
            return pd.DataFrame(data=[], columns=[input1.id_field, input2.id_field])

        return pd.DataFrame(results)

    def synchronize_pair(
        self, input1: SyncableContent, input2: SyncableContent
    ) -> SyncableContent:
        """
        Synchronize two SyncableContent objects.

        This method should be overridden for each new object type: see [PlayerSyncEngine][glass_onion.player.PlayerSyncEngine] for an example.

        Args:
            input1 (glass_onion.SyncableContent): a SyncableContent object from `SyncEngine.content`
            input2 (glass_onion.SyncableContent): a SyncableContent object from `SyncEngine.content`

        Returns:
            If `input1`'s underlying `data` dataframe is empty, returns `input2` with a column in `input2.data` for `input1.id_field`.
            If `input2`'s underlying `data` dataframe is empty, returns `input1` with a column in `input1.data` for `input2.id_field`.
            If both dataframes are non-empty, returns a new PlayerSyncableContent object with synchronized identifiers from `input1` and `input2`.

        Raises:
            NotImplementedError: if this method is not overridden.
        """
        if len(input1.data) == 0 and len(input2.data) > 0:
            input2.data[input1.id_field] = pd.NA
            return input2

        if len(input1.data) > 0 and len(input2.data) == 0:
            input1.data[input2.id_field] = pd.NA
            return input1

        raise NotImplementedError()

    def synchronize(self) -> SyncableContent:
        """
        Synchronizes the full list of SyncableContent objects from `SyncEngine.content` using `SyncEngine.synchronize_pair()`.

        There are three distinct layers here:

        1. The aforementioned sync process that results in a data frame of synced identifiers.
        2. Collect remaining unsynced rows and run the sync process on those. Append any newly synced rows to the result dataframe from Layer 1.
        3. Append any remaining unsynced rows to the bottom of the result data frame.

        This result dataframe is then deduplicated: by default, the result dataframe is grouped by `SyncEngine.join_columns` and the first non-null result is selected for each data provider's identifier field.

        The result dataframe is then wrapped in a SyncableContent object using the `provider` from the first SyncableContent object in `SyncEngine.content`.

        Returns:
            * If there are no elements in `SyncEngine.content`, returns a SyncableContent object with `SyncEngine.object_type` with `provider` unknown and an empty `pandas.DataFrame`.
            * If there's only one element in `SyncEngine.content`, returns that element.
            * If there are 2+ elements in `SyncEngine.content`, returns a new SyncableContent object with synchronized identifiers based on the SyncableContent objects in `SyncEngine.content`.
        """
        if len(self.content) == 0:
            return SyncableContent(self.object_type, "unknown", pd.DataFrame())

        if len(self.content) == 1:
            return self.content[0]

        # first pass: straight matching - approach: agglomerative
        self.verbose_log(
            f"Starting {self.object_type} synchronization across {len(self.content)} datasets"
        )

        self.verbose_log(f"Layer 1: agglomeration")
        results = []
        for i in range(0, len(self.content) - 1):
            x = self.content[i]
            y = self.content[i + 1]
            z = self.synchronize_pair(x, y)
            results.append(z)

        sync_result = reduce(lambda x, y: x.merge(y), results[1:], results[0])
        id_mask = list(map(lambda x: x.id_field, self.content))

        synced = SyncableContent(
            self.object_type,
            self.content[0].provider,
            sync_result.data.dropna(subset=id_mask),
        )

        self.verbose_log(
            f"Layer 1: Using {self.content[0].provider} as sync basis, found {len(sync_result.data)} total rows and {len(synced.data)} fully synced rows."
        )

        ## second pass: relate remainders to each other
        remainders = []
        for c in self.content:
            missing = c.data[~(c.data[c.id_field].isin(synced.data[c.id_field]))]

            if len(missing) > 0:
                self.verbose_log(
                    f"Layer 2: Aggregating {len(missing)} identified unsynced rows for {c.provider}"
                )
                remainders.append(
                    SyncableContent(self.object_type, c.provider, missing)
                )

        if len(remainders) > 1:
            self.verbose_log(
                f"Layer 2: Agglomeration on remaining unsynced rows across {len(remainders)} datasets"
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
                f"Layer 2: Using remainders as sync basis, found {len(remainders_result.data)} new fully synced rows."
            )
            if len(remainders_result.data) > 0:
                synced.append(remainders_result)

        ## third pass: add remainders to end
        remainder_dfs = []
        for c in self.content:
            r = c.data[~(c.data[c.id_field].isin(synced.data[c.id_field]))]
            if len(r) > 0:
                self.verbose_log(
                    f"Layer 3: Aggregating {len(r)} identified unsynced rows for {c.provider}"
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
                remainder_dfs.append(r_appendable)

        if len(remainder_dfs) > 0:
            remainder_df = pd.concat(remainder_dfs, axis=0)
            self.verbose_log(f"Layer 3: Including {len(remainder_df)} unsynced rows")
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
