# Integrating Glass Onion

When building an object identifier sync pipeline, there are a bunch of other tasks that you may need to do that Glass Onion does not provide support for: deduplication, false positive detection, etc. Provided below is an abridged and generalized version of the data pipeline that powers our (US Soccer) unified data schema. This is not meant to be the canonical implementation pathway -- just an example based on our environment and needs. Please let us know if you see ways in which we can improve this document (and therefore our implementation)!

In general, our process looks something like this for every object type:

<img src="/assets/img/integration/Slide1.jpeg" height="100%" />

In our pipeline, each object type depends on a "higher-order" object type to have unified identifiers in order to reduce the search space of potential matches (more details in [Step 2](#step-2-glass-onion-synchronization)). Data providers modify competitions and seasons least often, so we synchronize those by hand. That manual work allows us to automate the synchronization process for teams, which unlocks that process for matches, which then unlocks that process for players.

## Start: Provider-specific object tables

Our goal with this pipeline should be to take an object's provider-specific tables and generate one "source of truth" table for that object with identifiers we can use across our systems. To achieve that vision, this final table must meet a few different criteria:
- [ ] Does not include duplicate rows or duplicate identifiers
- [ ] Contains the most accurate metadata for a given object
- [ ] Object identifiers are durable and unique so that they can be used reliably across our systems

## Step 1: Data Collection

We collect data from the provider-specific tables into a single Spark DataFrame with the schema:
- data_provider: the data provider's name
- provider_object_id: the ID of the object in the data provider's system
- Any object-specific columns to use for synchronization
- A grouping key (More on this in [Step 2](#step-2-glass-onion-synchronization))

Here's how this might look like in code for player synchronization:

```python
## Notes:
## - ussf.competition_match and ussf.team are "higher-order" unified tables that assist us in synchronizing player identifiers.
## - provider_a.player_match contains data on a player performance in a given match. The primary key for this table is match_id + player_id.

prov_a = spark.sql(
"""
    SELECT 
        um.match_id,
        pm.player_id AS provider_player_id,
        pm.jersey_number, 
        pm.player_name, 
        pm.player_nickname, 
        pm.birth_date, 
        ut.team_id,
        pm.player_gender
    FROM provider_a.player_match pm
    INNER JOIN ussf.competition_match um 
        ON pm.match_id = um.provider_a_match_id
    INNER JOIN ussf.team ut 
        ON pm.team_id = ut.provider_a_team_id
"""
)
```


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

    One quirk of our implementation: we store the "preliminary set" and "knockout list" in Delta tables. Why? While developing our data pipeline, we saw that using `MERGE INTO` was orders of magnitude faster than pulling a list of identifiers into a Spark `filter()` statement when the tables are massive (100k+ rows).

## Step 4: Flagging Duplicates

Our "knockout logic" effectively ignores the existence of duplicates in the "knockout list". We identify these by counting the number of times a given provider's identifier appears in the "knockout list". If an identifier appears more than once, we flag those rows and then set them aside in a `object_flagged` table for manual review.

In code:

```python

id_counts = {
    f"{k}_num": F.row_number().over(
        Window.partitionBy(k)
            .orderBy("ordering_column")
    ) 
    for k in id_mask
}

knockout_list = spark.read.table("knockout_list").withColumns(id_counts)

id_count_filters = " OR ".join([f"({k}_num > 1 AND {k} IS NOT NULL)" for k in id_mask])

all_duplicates = (
    knockout_list
        .filter(id_count_filters)
        .collect()
)

dupe_insertables = {
    k.name: f"target.{k.name}" for k in all_duplicates.schema.fields
}

if all_duplicates.count() > 0:
    (
        DeltaTable.forName(spark, "ussf.object.flagged")
            .alias("source")
            .merge(
                all_duplicates.alias("target"),
                " OR ".join(
                    [
                        f"(source.{k} = target.{k} AND source.{k} IS NOT NULL AND target.{k} IS NOT NULL)" 
                        for k in override_mask
                    ]
                )
            )
            .whenNotMatchedInsert(
                values=dupe_insertables
            )
            .execute()
    )
    display(all_duplicates)

```

## Step 5: Data Formatting

Now that we have flagged duplicate rows, we can prepare the "knockout list" to be written to the table in our unified schema for this object. This has three stages:

- Removing any duplicate rows (including those we have flagged before this pipeline run)
- Joining our list back to provide-specific tables for object metadata
- Setting object metadata fields based on provider priority
- Assigning a unique and durable identifier for the object

The first two are easy: 

- `dataframe.filter()` or `MERGE INTO` take care of removing rows with duplicate identifiers.
- `dataframe.join()` on a provider identifier brings in data from the provider-specific table.


But the next task poses a philosophical question: if our table for this object is supposed to be a source of truth for this object across all systems, how do we know which data provider is "true"? How do we know which _fields_ they are "true" for? We have to come up with some prioritization strategy for each field in order to properly take object metadata from the "truest" sources first. For example, let's say we have Providers A, B, and C for player data: 

- Provider A might be really accurate with birth dates but prefer legal/official names over commonly used names
- Provider B might have common names/mononyms, but have spotty accuracy on birth dates. 
- Provider C is ok on names and birth dates, but is extremely accurate with country of birth / national team affiliation.

We might structure our data formatting like so:

```python
# assuming you have joined provider-specific tables with aliases `prov_a`, `prov_b`, `prov_c`:

knockout_list = (
    knockout_list
        .select(
            # ... other fields ...

            # prefer providers A and C for birth dates
            F.coalesce(
                F.col("prov_a.birth_date"),
                F.col("prov_c.birth_date")
                F.col("prov_b.birth_date"),
            ).alias("birth_date"),

            # prefer providers B and A for names
            F.coalesce(
                F.col("prov_b.player_name"),
                F.col("prov_a.player_name")
                F.col("prov_c.player_name"),
            ).alias("player_name"),

            # prefer provider C for country of birth
            F.coalesce(
                F.col("prov_c.country_of_birth"),
                F.col("prov_a.country_of_birth")
                F.col("prov_b.country_of_birth"),
            ).alias("country_of_birth")
        )
)

```

This brings us to our final formatting step: building durable, unique identifiers for each object row. There are a number of different ways to do this (GUID, UUID, ULID, auto-increment ints, etc), but we chose to build our own identifier using the object metadata. In the case of players, we used:

- the player's gender
- a normalized version of the player's name, cut to 10 characters (the "prefix")
- the instance of the prefix for that gender in the table: 1, 2, 3, etc. (the "index")

Given a player named "test player" whose "prefix" only appears once in our dataset, results in an identifier that looks something like: `female-testplayer-1`. 

There's one quirk with this: when using a `Window`, Spark forces you to sort that window. In our player pipeline, we sort this window by player name. This can result in _new_ player rows that are sorted ahead of _existing_ player rows, which throws off the "index" like below:

| Player ID    | Player Name |   Prefix | Gender |   Index |
| -------- | ------- | -------- | ------- | ------- |
| `female-testplayer-1`| Test Player A | `testplayer` | female | 1 |
| `NULL` | Test Player B | `testplayer` | female | 2 |
| `female-testplayer-2`| Test Player C | `testplayer` | female | 3 |

This could also come up if player rows have ever been deleted from the table. This edge case breaks our two identifier tenets: 

- Not durable: if we choose to update all identifiers when new rows appear, these identifiers become unreliable in downstream analyses.

| Player ID (Old)   | Player Name |   Prefix | Gender |   Index | Player ID (New) |
| -------- | ------- | -------- | ------- | ------- | ----- |
| `female-testplayer-1`| Test Player A | `testplayer` | female | 1 | `female-testplayer-1` |
| `NULL` | Test Player B | `testplayer` | female | 2 | `female-testplayer-2` |
| `female-testplayer-2`| Test Player C | `testplayer` | female | 3 | `female-testplayer-3` |

- Not unique: if we instead keep existing identifers and only fill in identifiers where they don't exist, we'd end up with duplicate identifiers.

| Player ID (Old)   | Player Name |   Prefix | Gender |   Index | Player ID (New) |
| -------- | ------- | -------- | ------- | ------- | ----- |
| `female-testplayer-1`| Test Player A | `testplayer` | female | 1 | `female-testplayer-1` |
| `NULL` | Test Player B | `testplayer` | female | 2 | `female-testplayer-2` |
| `female-testplayer-2`| Test Player C | `testplayer` | female | 3 | `female-testplayer-2` |

We must develop a more complex rule to generate new indices for a given prefix and gender. Here's what we've found to work:

- For all rows with existing identifiers, extract the index.
- Determine the maximum existing index value.
- Start counting new indices from this maximum value.

Here's what our example table would look like if we apply this strategy:

| Player ID (Old)    | Player Name |   Prefix | Gender |   Index (Old) |   Index (New) | Player ID (New)
| -------- | ------- | -------- | ------- | ------- | ------- | ----- |
| `female-testplayer-1`| Test Player A | `testplayer` | female | 1 | 1 | `female-testplayer-1`|
| `NULL` | Test Player B | `testplayer` | female | 2 | 3 | `female-testplayer-3`|
| `female-testplayer-2`| Test Player C | `testplayer` | female | 3 | 2 | `female-testplayer-2`|

Here's what this looks like in code:

```python

knockout_list = (
    knockout_list
        # assuming we've already created our player prefix column as `player_prefix` and we have a `player_id` column for existing identifiers
        .withColumn(
            "player_id_extract_num",
            F.regexp_extract(F.col("player_id"), r"\d+$", idx=0).cast(T.IntegerType())
        )
        .withColumn(
            "player_id_max",
            F.max(F.col("player_id_extract_num")).over(
                Window.partitionBy(["player_gender", "player_prefix"])
            )
        )
        .withColumn(
            "player_id_max",
            F.when(F.col("player_id_max").isNull(), F.lit(0)).otherwise(F.col("player_id_max"))
        )
        .withColumn(
            "player_index",
            F.when(
                F.col("player_id").isNull(),
                F.col("player_id_max") + F.row_number().over(
                    Window.partitionBy(["player_gender", "player_prefix"])
                        .orderBy(
                            [
                                # put the NULLs first so that they get indexes 1, 2, 3, etc.
                                F.col("player_id").asc_nulls_first(),
                                F.col("player_name")
                            ]
                        )
                )
            )
        )
        .withColumn(
            "player_id",
            F.when(
                F.col("player_id").isNull(), 
                F.concat_ws("-", F.col("player_gender"), F.col("player_prefix"), F.col("player_index"))
            ).otherwise(F.col("player_id"))
        )
)

```

## Final: unified object table

With our pipeline generating a "knockout list" that meets our target criteria:

- [X] Does not include duplicate rows or identifiers
- [X] Includes most accurate metadata for the object from the data provider that tends to be the most accurate for each metadata field
- [X] Assigns objects unique and durable identifiers

We can simply execute a `MERGE INTO` statement into the table for this object in our unified schema:

```python

data_providers = ["provider_a", "provider_b", "provider_c"]
table_fields = data_providers + [
    # ... other object metadata fields ...
]
insertables = {k: f"target.{k}_object_id" for k in table_fields}
updatables = {k: v for k, v in insertables.items() if k != "object_id"}

(
    DeltaTable.forName(spark, "ussf.object")
        .alias("source")
        .merge(
            spark.read.table("knockout_list").alias("target"), 
            "source.object_id = target.object_id"
        )
        .whenMatchedUpdate(
            set={
                **updatables,
                "updated_at": "CURRENT_TIMESTAMP()" # useful for tracking changes
            }
        )
        .whenNotMatchedInsert(
            values={
                **insertables,
                "created_at": "CURRENT_TIMESTAMP()",
                "updated_at": "CURRENT_TIMESTAMP()"
            }
        )
        .execute()
)

```

And just like that: we're done with our pipeline! 