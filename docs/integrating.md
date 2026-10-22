# Integrating Glass Onion

When building an object identifier sync pipeline, there are a bunch of other tasks that you may need to do that Glass Onion does not provide support for: deduplication, false positive detection, etc. Provided below is an abridged and generalized version of the data pipeline that powers our (US Soccer) unified data schema. This is not meant to be the canonical implementation pathway -- just an example based on our environment and needs. Please let us know if you see ways in which we can improve this document (and therefore our implementation)!

In general, our process looks something like this for every object type:

<img src="site:assets/img/integration/Slide1.png" style="margin: auto !important; display: block;" />

In our pipeline, each object type depends on a "higher-order" object type to have unified identifiers in order to reduce the search space of potential matches (more details in [Step 2](#step-2-glass-onion-synchronization)). Data providers modify competitions and seasons least often, so we synchronize those by hand. That manual work allows us to automate the synchronization process for teams, which unlocks that process for matches, which then unlocks that process for players.

## Start: Provider-specific object tables

Our goal with this pipeline should be to take an object's provider-specific tables and generate one "source of truth" table for that object with identifiers we can use across our systems. To achieve that vision, this final table must meet a few different criteria:

- [ ] Does not include duplicate rows or identifiers
- [ ] Includes most accurate metadata for the object from the data provider that tends to be the most accurate for each metadata field
- [ ] Assigns objects unique and durable identifiers that can be used reliably across our systems

## Step 1: Data Collection

We collect data from the provider-specific tables into a single Spark DataFrame with the schema:

- data_provider: the data provider's name
- provider_object_id: the ID of the object in the data provider's system
- Any object-specific columns to use for synchronization
- A grouping key (More on this in [Step 2](#step-2-glass-onion-synchronization))

Here's what this might look like in code for player synchronization:

```python linenums="1"
from functools import reduce

"""
Notes:
    - `ussf.competition_match` and `ussf.team` are "higher-order" unified
      tables used to synchronize player identifiers.
    - provider_a.player_match contains player performance data for a given
      match. The primary key is `match_id` + `player_id`.
"""

prov_a = spark.sql(
    """
    SELECT 
        um.match_id,
        'provider_a' AS data_provider,
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

# ... and so on for other providers ...

all_remaining_records = reduce(
    lambda a, b: a.union(b),
    [
        prov_a,  # ... other providers ...
    ],
).select(
    "match_id",
    "provider_player_id",
    "data_provider",
    "jersey_number",
    "player_name",
    "player_nickname",
    "birth_date",
    "team_id",
    "player_gender",
)
```


## Step 2: Synchronization

Stuffing all of our data into a common schema is a architectural choice forced by PySpark's [`GroupedData.applyInPandas(func, schema)`](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.applyInPandas.html) to parallelize the work of identifier synchronization across a Spark cluster's nodes. This choice also has the benefit of naturally reducing the search space for synchronization if we have unified identifiers for higher-order objects. For matches and teams, we group by internal identifiers for competition and season, and for players, we group by an internal match identifier. 

To use `GroupedData.applyInPandas(func, schema)` properly, we predefine a target schema (`schema`) and wrap [SyncEngine.synchronize][glass_onion.engine.SyncEngine.synchronize] a function (`func`) to produce that schema consistently. If given providers A to C, our target schema looks something like this:
```python linenums="1"
from pyspark.sql import types as T

target_schema = T.StructType(
    [
        T.StructField(
            "grouping_key", T.StringType()
        ),  # to track the group this match came from
        T.StructField("provider_a_object_id", T.StringType()),
        T.StructField("provider_b_object_id", T.StringType()),
        T.StructField("provider_c_object_id", T.StringType()),
        T.StructField("data_provider", T.StringType()),
        # ... more T.StructField() invocations for SyncEngine.join_columns ...
    ]
)
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
        {"provider_object_id": f"{provider}_object_id"}, axis=1, inplace=True
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
            data=transform_provider_data(dataset.loc[dataset.index.isin(d),], p),
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
                "data_provider": pd.NA,
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

With these prerequisites in place, we can actually run `GroupedData.applyInPandas(func, schema)`:

```python linenums="1"
all_synced_records = (
    all_remaining_records
        .groupBy("grouping_key")
        .applyInPandas(
            synchronize, target_schema
        )
)

(
    all_synced_records.write.mode("overwrite")
    .option("mergeSchema", "true")
    .format("delta")
    .saveAsTable("preliminary_set")
)

display(all_synced_records)
```

## Step 3: Resolution

### Applying `ObjectResolver`

Once we have this preliminary set of synchronized identifiers, we can determine which results are valid. We throw all of these "proposed results" in an instance of [ObjectResolver][glass_onion.resolver.ObjectResolver] (see [Methodology](/methodology#resolution) for more details on how this class works). 

In short, if we can use "proposed results" to form a graph "component" (representing an object) where each provider has only one ID, we accept the "proposals". If for some reason (transitively or explicitly), a "proposal" would result in multiple IDs for the same provider in the same "component", then we reject the "proposal".

We start with our existing set of synced object records from `ussf.object`. 

```python linenums="1"
from glass_onion.resolver import ObjectResolver

id_mask = ["provider_a_object_id", "provider_b_object_id", "provider_c_object_id"] # and so on and so forth...
meta_columns = [
    # ... any additional columns ...
]
flag_columns = meta_columns + id_mask + ["flag_reason"]

# object_flagged.conflicts: the records each flagged record conflicted with, in the column order of its struct
conflict_columns = ["source"] + id_mask + meta_columns
conflicts_type = T.ArrayType(T.StructType([T.StructField(c, T.StringType()) for c in conflict_columns]))

def spark_to_records(rows, source):
    # a record is one source row: the IDs it syncs together plus the fields needed to flag it or describe the player
    records = []
    for row in rows:
        row = row.asDict()
        ids = {k: row[k] for k in id_mask if row.get(k) is not None}
        if len(ids) > 0:
            records.append({"source": source, "ids": ids, "meta": {k: row.get(k) for k in meta_columns}})
    return records

# 0. Grab datasets and format into dataframes

object_records = spark_to_records(
    normalize_ids(spark.read.table("ussf.object"))   # `normalize_ids` omitted; it just turns every id column into a StringType for easier comparison
        .select(id_mask + meta_columns)
        .collect(),
    "player"
)

# one record per distinct ID combination, described by the first grouping key it was synced in
first_seen = Window.partitionBy(*id_mask).orderBy(F.col("grouping_key").asc_nulls_last(), "data_provider")
raw_records = spark_to_records(
    normalize_ids(spark.read.table("preliminary_set"))
        .filter(" OR ".join([f"{k} IS NOT NULL" for k in id_mask]))
        .withColumn("first_seen", F.row_number().over(first_seen))
        .filter("first_seen = 1")
        .select(id_mask + [F.col(c).cast(T.StringType()).alias(c) for c in meta_columns])
        .collect(),
    "raw"
)
```

We can then generate the graph of identifiers via [ObjectResolver][glass_onion.resolver.ObjectResolver]:

```python linenums="1"

# 1. build the graph from the sources of truth (ussf.object)
graph = ObjectResolver()
graph.resolve([r for r in object_records if len(r["ids"]) > 1])

all_added, all_rejected = graph.added, graph.rejected
# a single ID can't conflict; add_record skips it if a larger record already holds it
added_singles = []
for r in object_records:
    if len(r["ids"]) == 1 and graph.add_record(r):
        added_singles.append(r)

all_added = pd.concat([added_overrides, all_added, to_frame(added_singles)], ignore_index=True)
```

Then, we can layer on our proposals and add any single-identifier proposals to the graph. We hold out any proposals that have previously appeared in our `ussf.object_flagged` table to avoid adding them to a valid object component in the graph.

```python 
partial_hold_reasons = ["duplicate"]

def held_ids(vouched):
    held = set()
    flagged = normalize_ids(spark.read.table("ussf.object_flagged"))
    for row in flagged.select(id_mask + ["flag_reason"]).collect():
        vs = [(k, row[k]) for k in id_mask if row[k] is not None]
        if row["flag_reason"] in partial_hold_reasons:
            vs = [v for v in vs if v not in vouched]
        held.update(vs)
    return held

# `vouched` IDs come from the ussf.object table. We implicitly trust that these are correct.
vouched = {v for vs in frame_vertices(all_added) for v in vs}
# We grab the previously held-out/flagged IDs, ignoring the IDs we can vouch for (via helper method `held_ids`)
previously_held = held_ids(vouched)

# 2. merge in proposals. A proposal with a held-out ID is, well, held out to avoid that ID being pulled into a valid object's component
proposals = [r for r in raw_records if len(r["ids"]) > 1]
held_proposals = [r for r in proposals if any(v in previously_held for v in vertices(r))]
graph.resolve([r for r in proposals if not any(v in previously_held for v in vertices(r))])

# 3. a vertex in the graph with no edges (IE: a single-ID proposal) becomes its own record, unless it's already in the graph, held out, or in a record left out above
excluded = (
    previously_held
    | {v for r in held_proposals for v in vertices(r)}
    | {v for frame in (all_rejected, rejected_proposals) for vs in frame_vertices(frame) for v in vs}
)

for r in raw_records:
    if len(r["ids"]) == 1:
        v = vertices(r)[0]
        if v not in graph and v not in excluded:
            graph.add_record(r)
```

### Handling Rejected Proposals

But what happens to rejected proposals? We add these (and the other records they conflict with) to the `ussf.object_flagged` table for manual review:

```python linenums="1"
flag_rows = {}

def flatten(records):
    # Records, flattened as in ObjectResolver.rejected
    return to_frame(records).to_dict("records")

def add_flag(row, reason, conflicts):
    # row: one record, flattened as in ObjectResolver.rejected; conflicts: the flattened records it conflicted with
    ids = {k: value(row, k) for k in id_mask if value(row, k) is not None}
    key = (row["source"], tuple(sorted(ids.items())))
    flag = flag_rows.setdefault(key, {**{k: value(row, k) for k in meta_columns}, **{k: ids.get(k) for k in id_mask}, "flag_reason": reason, "conflicts": []})

    # keeping the conflicts in the record allows us to resolve them more easily in the future
    for x in conflicts:
        c = (x["source"],) + tuple(value(x, k) for k in conflict_columns[1:])
        if c not in flag["conflicts"]:
            flag["conflicts"].append(c)

for row in truth_rejected.to_dict("records"):
    add_flag(row, "conflicting source of truth", flatten(row["conflicts"]))

for row in rejected_proposals.to_dict("records"):
    add_flag(row, "duplicate", flatten(row["conflicts"]))

# ... some cleaning code removed ...

print(f"{datetime.now()}: records to flag: {len(flag_rows)}")
if len(flag_rows) > 0:
    flags = spark.createDataFrame(
        [tuple(row[c] for c in flag_columns) + (row["conflicts"],) for row in flag_rows.values()],
        T.StructType([T.StructField(c, T.StringType()) for c in flag_columns] + [T.StructField("conflicts", conflicts_type)])
    )
    append_new_flagged_ids(flags.dropDuplicates(id_mask), flag_columns + ["conflicts"])
```

### Assembling Results Table

We can then convert our graph components into rows, accounting for duplicates...

```python linenums="1"
# 4. write one row per component to the network table
network_columns = id_mask + [
    # ... any metadata columns ...
]

def first_seen_key(record):
    m = record["meta"]["grouping_key"]
    return (m is None, m or "")

network_rows = []
# for each identified component
for ids, records in graph.components():
    # get the records that were sourced from the ussf.object table
    objects = [r for r in records if r["source"] == "object"]
    # get the records that were sourced from the raw data, sorted by the first grouping key they were seen in
    raw = sorted([r for r in records if r["source"] == "raw"], key=first_seen_key)
    # prioritize the object record as a source of truth. If not available, use raw data. Pull metadata values from this place
    meta = (objects or raw or records)[0]["meta"]
    # two steps:
    # 1. if this row came from the raw data (is net new), then we get the first grouping key it appeared in from the dataset
    #   - if the row came from the object records (or for some reason neither source), then set to None
    # 2. IF step 1 produced None, return 'a' as a placeholder to sort above all raw data. 
    ussf_match_id = (raw[0]["meta"]["ussf_match_id"] if len(raw) > 0 and len(objects) == 0 else None) or "a"
    # create a tuple of the ID values + the appropriate metadata values from `meta`
    network_rows.append(
        tuple([ids.get(k) for k in network_id_columns])
        + (
            ussf_match_id, 
            get_right_provider(meta["data_provider"], ids), 
            # ... any metadata values ...
        )
    )
```

...and then we can save these into a table:

```python linenums="1"
(
    spark.createDataFrame(network_rows, T.StructType([T.StructField(c, T.StringType()) for c in network_columns]))
        .write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .format("delta")
        .saveAsTable("ussf.tmp_object_network")
)
```

...and then remove any previously flagged identifiers from the list:

```python linenums="1"
clean_list = anti_join_on_held(
    spark.read.table("ussf.tmp_object_network"), held_ids(vouched)
)
```

## Step 4: Data Formatting

Now that we have flagged and removed duplicate rows, we can prepare the cleaned sync results to be written to the table in our unified schema for this object. This has three stages:

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

```python linenums="1"
# assuming you have joined provider-specific tables with aliases `prov_a`, `prov_b`, `prov_c`:

clean_list = (
    clean_list
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

```python linenums="1"
clean_list = (
    clean_list
    # assuming we've already created our player prefix column as `player_prefix` and we have a `player_id` column for existing identifiers
    .withColumn(
        "player_id_extract_num",
        F.regexp_extract(F.col("player_id"), r"\d+$", idx=0).cast(T.IntegerType()),
    )
    .withColumn(
        "player_id_max",
        F.max(F.col("player_id_extract_num")).over(
            Window.partitionBy(["player_gender", "player_prefix"])
        ),
    )
    .withColumn(
        "player_id_max",
        F.when(F.col("player_id_max").isNull(), F.lit(0)).otherwise(
            F.col("player_id_max")
        ),
    )
    .withColumn(
        "player_index",
        F.when(
            F.col("player_id").isNull(),
            F.col("player_id_max")
            + F.row_number().over(
                Window.partitionBy(["player_gender", "player_prefix"]).orderBy(
                    [
                        # put the NULLs first so that they get indexes 1, 2, 3, etc.
                        F.col("player_id").asc_nulls_first(),
                        F.col("player_name"),
                    ]
                )
            ),
        ),
    )
    .withColumn(
        "player_id",
        F.when(
            F.col("player_id").isNull(),
            F.concat_ws(
                "-",
                F.col("player_gender"),
                F.col("player_prefix"),
                F.col("player_index"),
            ),
        ).otherwise(F.col("player_id")),
    )
)

(
    clean_list
        .write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .format("delta")
        .saveAsTable("ussf.tmp_object_clean")
)
```

## Final: unified object table

With our pipeline generating a "clean list" that meets our target criteria:

- [X] Does not include duplicate rows or identifiers
- [X] Includes most accurate metadata for the object from the data provider that tends to be the most accurate for each metadata field
- [X] Assigns objects unique and durable identifiers that can be used reliably across our systems

We can simply execute a `MERGE INTO` statement into the table for this object in our unified schema:

```python linenums="1"
data_providers = ["provider_a", "provider_b", "provider_c"]
table_fields = (
    data_providers
    + [
        # ... other object metadata fields ...
    ]
)
insertables = {k: f"target.{k}_object_id" for k in table_fields}
updatables = {k: v for k, v in insertables.items() if k != "object_id"}

(
    DeltaTable.forName(spark, "ussf.object")
    .alias("source")
    .merge(
        spark.read.table("ussf.tmp_object_clean").alias("target"),
        "source.object_id = target.object_id",
    )
    .whenMatchedUpdate(
        set={
            **updatables,
            "updated_at": "CURRENT_TIMESTAMP()",  # useful for tracking changes
        }
    )
    .whenNotMatchedInsert(
        values={
            **insertables,
            "created_at": "CURRENT_TIMESTAMP()",
            "updated_at": "CURRENT_TIMESTAMP()",
        }
    )
    .execute()
)
```

And just like that: we're done with our pipeline! 