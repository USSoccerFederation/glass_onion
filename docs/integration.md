# Integrating Glass Onion

When building an object identifier sync pipeline, there are a bunch of other tasks that you may need to do that Glass Onion does not provide support for: deduplication, false positive detection, etc. Provided below is an abridged and generalized version of the data pipeline that powers our (US Soccer) unified data schema. This is not meant to be the canonical implementation pathway -- just an example based on our environment and needs. Please let us know if you see ways in which we can improve this document (and therefore our implementation)!

In general, our process looks something like this for every object type:

<img src="/assets/img/integration/Slide1.jpeg" height="100%" />

In our pipeline, each object type depends on a "higher-order" object type to have unified identifiers in order to reduce the search space of potential matches (more details in [Step 2](#step-2-glass-onion-synchronization)). Data providers modify competitions and seasons least often, so we synchronize those by hand. That manual work allows us to automate the synchronization process for teams, which unlocks that process for matches, which then unlocks that process for players.

## Start: Provider-specific object tables

## Step 1: Data Collection

We collect data from provider-specific tables into a single Spark DataFrame with the schema: data_provider, provider_object_id, (object-specific columns for matching), grouping key. More on this below.

## Step 2: Glass Onion synchronization

Stuffing all of our data into a common schema is a architectural choice forced by PySpark's [`GroupedData.applyInPandas(func, schema)`](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.applyInPandas.html) to parallelize the work of identifier synchronization across a Spark cluster's nodes. This choice also has the benefit of naturally reducing the search space for synchronization if we have unified identifiers for higher-order objects. For matches and teams, we group by internal identifiers for competition and season, and for players, we group by an internal match identifier. 

To use `GroupedData.applyInPandas(func, schema)` properly, we predefine a target schema (`schema`) and wrap `SyncEngine.synchronize()` a function (`func`) to produce that schema consistently. If given providers A to C, our target schema looks something like this:
```python linenums="1"
from pyspark.sql import types as T

target_schema = T.StructType([
    T.StructField("grouping_key", T.StringType()), # to track the group this match came from
    T.StructField("provider_a_object_id", T.StringType()),
    T.StructField("provider_b_object_id", T.StringType()),
    T.StructField("provider_c_object_id", T.StringType()),
    T.StructField("data_provider", T.StringType()),

    # ... more T.StructField() invocations for SyncEngine.join_columns ...
])
```

`func` will receive a pandas.DataFrame that contains a subset of our big object table without the grouping key added, so we need to do three things:
- Transform this subset into a list of `SyncableContent` instances
- Run those instances through `SyncEngine`
- Format `SyncEngine`'s result into the target schema.

Thus, `func` ends up looking something like this:

```python linenums="1"

import pandas as pd
# NOTE: this class doesn't actually exist. We're just using it for our example.
from glass_onion.object import ObjectSyncableContent, ObjectSyncEngine 


def transform_provider_data(dataset: pd.DataFrame, provider: str) -> pd.DataFrame:
    dataset.rename(
        {
            "provider_object_id": f"{provider}_object_id"
        },
        axis=1,
        inplace=True
    )
    dataset.drop(["data_provider"], axis=1, inplace=True)
    return dataset

# NOTE: if you use multiple grouping keys, `GroupedData.applyInPandas(func, schema)` will pass a Tuple 
# containing the keys to the first param of `func` (if you provide multiple).
def synchronize(grouping_key: str, dataset: pd.DataFrame) -> pd.DataFrame:

    ## Stage 1: convert from the common big table schema to SyncableContent
    grouped = dataset.groupby("data_provider")

    syncables = [
        ObjectSyncableContent(
            provider=p, 
            data=transform_provider_data(dataset.loc[dataset.index.isin(d), ], p)
        ) 
        for p, d in grouped.groups.items()
    ]
    syncables = [k for k in syncables if len(k.data) > 0]

    ## handle empties explicitly
    skeleton = pd.DataFrame(
        [
            {
                "grouping_key": grouping_key,
                "provider_a_object_id": pd.NA,
                "provider_b_object_id": pd.NA,
                "provider_c_object_id": pd.NA,
                # ... key/values for ObjectSyncEngine.join_columns ...
                "data_provider": pd.NA
            }
        ]
    ).head(0)

    if len(syncables) == 0:
        return skeleton

    ## Stage 2: run the list of SyncableContent through SyncEngine
    engine = ObjectSyncEngine(syncables, verbose=True)
    result = engine.synchronize()

    ## Stage 3: make SyncEngine's result match target schema
    missing_columns = skeleton.columns[~(skeleton.columns.isin(result.data.columns))]
    if len(missing_columns) > 0:
        result.data[missing_columns] = pd.NA

    return result.data[skeleton.columns.to_list()]
```

## Step 3: "Knockout" Logic

Once we have a preliminary set of synchronized identifiers (the "preliminary set" below), we can run them through our "knockout logic". First, we retrieve the list of existing synchronized identifiers from `ussf.object` and store it in a temporary dataframe (our "knockout list").
```python linenums="1"
(
    spark.read.table("ussf.object")
        .write
        .mode("overwrite")
        .option("mergeSchema", "true")
        .format("delta")
        .saveAsTable("knockout_list")
)
```

Then, for each data provider (say, Provider A) in the list:
1. Rows with instances of existing non-null identifiers for Provider A from the "knockout list" are removed from the "preliminary set" (IE: they are "knocked out").
2. We group the set of remaining synchronized identifiers by Provider A's identifiers.
3. In each group, we find the first non-null identifier for every other data provider (say, B through Z). _However_, if we find multiple identifiers in a group for, say, Provider B, we set provider B's identifier to NULL instead.
4. The rows aggregated for Provider A from this grouping process are added to the "knockout list".
5. This process repeats until we exhaust the list of data providers or there are no more rows in the "preliminary set".

Here's what this looks like in code:

```python linenums="1"
data_providers = ["provider_a", "provider_b", "provider_c"]

for k in data_providers:
    ## 1: remove any records that mention any of the IDs in the knockout list from preliminary set
    remaining_records = get_remaining_records()
    rem_records_count = get_remaining_records_count()
    if rem_records_count == 0:
      break

    base_records = (
        remaining_records
            .filter(f"{k} IS NOT NULL") # hide rows from remaining list with NULL
    )
    if base_records.count() == 0:
        continue
    
    ## 2+3: make sure that there's only one unique ID from each that `k` can be associated with. See utility functions below for more details.
    provider_rows = squish_provider_records(base_records, k)
    provider_rows_count = provider_rows.count()

    ## 4: add the complete records to the knockout list
    if provider_rows_count > 0:
        update_knockout_list(provider_rows, k)
        knock_out_records()
```


??? "Utility functions used above"

    ```python linenums="1"

    import pyspark.sql.functions as F
    from delta import DeltaTable

    def knock_out_records():
        return (
            DeltaTable.forName(spark, "preliminary_set")
                .alias("source")
                .merge(
                    spark.read.table("knockout_list").alias("target"),
                    " OR ".join([f"(source.{k} = target.{k} AND source.{k} IS NOT NULL AND target.{k} IS NOT NULL)" for k in data_providers])
                )
                .whenMatchedDelete()
                .execute()
        )

    def update_knockout_list(provider_rows, key: str):
        # merge on selected key
        midmerge = (
            DeltaTable.forName(spark, "knockout_list")
                .alias("source")
                .merge(
                    provider_rows.alias("target"), 
                    f"(source.{key} = target.{key} AND source.{key} IS NOT NULL AND target.{key} IS NOT NULL)"
                )
        )

        # update values ONLY if they're missing in the source. If they exist already, DO NOT.
        # ignore the key column obviously.
        for k in data_providers:
            if k == key:
                continue

            midmerge = (
                midmerge
                    .whenMatchedUpdate(
                        condition=f"source.{k} IS NULL",
                        set={
                            k: f"target.{k}"
                        }
                    )
            )

        # if no match, insert all. then execute the merge
        (
            midmerge
                .whenNotMatchedInsertAll()
                .execute()
        )

    def get_remaining_records():
        return spark.sql("SELECT * FROM preliminary_set")

    def get_remaining_records_count():
        tmp = spark.sql("SELECT COUNT(*) AS rem_records_count FROM preliminary_set").collect()
        rem_records_count = tmp[0]["rem_records_count"]
        return rem_records_count

    def squish_provider_records(base_records, key):
        ## 3a. In each group, we find the first non-null identifier for every other data provider (say, B through Z). 
        parent_id_agg = [
            F.first(F.col(p), ignorenulls=True).alias(p) 
            for p in data_providers # + [ ... any columns from SyncEngine.join_columns ...]
            if p != key
        ]

        ## 3b. _However_, if we find multiple identifiers in a group for, say, Provider B, we set provider B's identifier to NULL instead.
        count_agg = [
            # NOTE: by design, `count_distinct` only counts non null values
            F.count_distinct(F.col(p)).alias(f"{p}_count")
            for p in data_providers
            if p != key
        ]

        count_adj = {
            p: F.when(F.col(f"{p}_count") <= F.lit(1), F.col(p)).otherwise(F.lit(None))
            for p in data_providers if p != key
        }

        provider_counts = (
            base_records
                .groupBy(k)
                .agg(*count_agg)
                .select(
                    [k] + [f"{p}_count" for p in data_providers if p != key]
                )
        )
        
        ## 2. We group the set of remaining synchronized identifiers by Provider A's identifiers.
        return (
            base_records
                .groupBy(k)
                .agg(*parent_id_agg)
                .join(provider_counts, key)
                .withColumns(count_adj)
                .select(
                    data_providers
                    # + [ ... any columns from SyncEngine.join_columns ...]
                )
        )

    ```

    One quirk of our implementation: we store the "preliminary set" and "knockout list" in Delta tables. Why? While developing our data pipeline, we saw that using `MERGE INTO` was orders of magnitude faster than pulling a list of identifiers into a Spark `filter()` statement.

## Step 4: Flagging Duplicates

Our "knockout" logic 


## Step 5: Data Formatting

## Final: unified object table