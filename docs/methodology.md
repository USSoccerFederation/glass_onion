# Methodology

## Synchronization

Let's set the stage with some example code (from [Getting Started](./getting_started.md)):

```python linenums="1"
from glass_onion import PlayerSyncEngine

engine = PlayerSyncEngine(content=[impect_content, statsbomb_content], verbose=True)
result = engine.synchronize()
```

In general, Glass Onion takes a list of [SyncableContent][glass_onion.engine.SyncableContent] and uses the logic in a [SyncEngine][glass_onion.engine.SyncEngine] to sync one pair at a time. The results of all pairs are then merged together and deduplicated. Each object type corresponds to a subclass of [SyncEngine][glass_onion.engine.SyncEngine] that overrides [synchronize_pair()][glass_onion.engine.SyncEngine.synchronize_pair] to define how pairs are synchronized in [synchronize()][glass_onion.engine.SyncEngine.synchronize], which contains wrapper logic for the entire process. 

There are three distinct layers within [synchronize()][glass_onion.engine.SyncEngine.synchronize]'s wrapper logic:

<img style="margin: auto !important; display: block;" src="site:assets/img/methodology/Slide2.png" />

1. The aforementioned sync process that results in a data frame of synced identifiers. How each object type is handled is described below.
2. Collect remaining unsynced rows and run the sync process on those. Append any newly synced rows to the result dataframe from Layer 1.
3. Append any remaining unsynced rows to the bottom of the result data frame.

This result dataframe is then deduplicated: by default, the result dataframe is grouped by the specific columns defined in [SyncEngine][glass_onion.engine.SyncEngine] and the first non-null result is selected for each data provider's identifier field. 

### Match

**NOTE**: Match synchronization can be also done using competition context (IE: columns `competition_id` and `season_id`, which are assumed to already be synchronized across providers) via `use_competition_context` (more details on `use_competition_context` in [MatchSyncEngine.init()][glass_onion.match.MatchSyncEngine.__init__] and the concept of "higher-order" object types on our [home page](./index.md)).

1. Attempt to join pair using `match_date`, `home_team_id`, and `away_team_id`.
2. Account for matches with different dates across data providers (timezones, TV scheduling, etc) by adjusting `match_date` in one dataset in the pair by -3 to 3 days, then attempting synchronization using `match_date`, `home_team_id`, and `away_team_id` again. This process is then repeated for the other dataset in the pair.
3. Account for matches postponed to a different date outside the [-3, 3] day range by attempting synchronization using `matchday`, `home_team_id`, and `away_team_id`.

### Team

**NOTE**: Team synchronization can be also done using competition context (IE: columns `competition_id` and `season_id`, which are assumed to already be synchronized across providers) via `use_competition_context` (more details on `use_competition_context` in [TeamSyncEngine.init()][glass_onion.team.TeamSyncEngine.__init__] and the concept of "higher-order" object types on our [home page](./index.md)).

1. Attempt to join pair simply on `team_name`.
2. With remaining records, attempt to match via cosine similarity using a minimum threshold of 75% similarity.
3. For any remaining records, attempt to match via cosine similarity using no minimum similarity threshold.

### Player

**NOTE**: [PlayerSyncEngine][glass_onion.player.PlayerSyncEngine] ignores syncable columns that have unreliable data (IE: NULLs/NAs in `jersey_number` or `birth_date`). The process below describes the best-case scenario. Please set `verbose_log=True` when creating a [PlayerSyncEngine][glass_onion.player.PlayerSyncEngine] instance to see the full synchronization process.

1. Attempt to join pair using `player_name` with a minimum 75% cosine similarity threshold for player name. Additionally, require that `jersey_number` and `team_id` are equal for matches that meet the similarity threshold.
2. Account for players with different birth dates across data providers (timezones, human error, etc) by adjusting `birth_date` in one dataset in the pair by -1 to 1 days and/or swapping the month and day, then attempting synchronization using `birth_date`, `team_id`, and a combination of `player_name` and `player_nickname`. This process is then repeated for the other dataset in the pair. 
3. Attempt to join remaining records using combinations of `player_name` and `player_nickname` with a minimum 75% cosine similarity threshold for player name. Additionally, require that `team_id` is equal for matches that meet the similarity threshold.
4. Attempt to join remaining records using "naive similarity": looking for normalized parts of one record's `player name` (or `player_nickname`) that exist in another's. Additionally, require that `team_id` is equal for matches found via this method.
5. Attempt to join remaining records using combinations of `player_name` and `player_nickname` with no minimum cosine similarity threshold. Additionally, require that `team_id` is equal.

## Resolution

Our previous attempt at identifier resolution relied on a "knockout" strategy, which we described as follows:

<blockquote>
Once we have a preliminary set of synchronized identifiers (the "preliminary set" below), we can run them through our "knockout logic". First, we retrieve the list of existing synchronized identifiers from `ussf.object` and store it in a temporary dataframe (our "knockout list").

Then, for each data provider (say, Provider A) in the list:

<ol>
<li>Rows with instances of existing non-null identifiers for Provider A from the "knockout list" are removed from the "preliminary set" (IE: they are "knocked out").</li>
<li>We group the set of remaining synchronized identifiers by Provider A's identifiers.</li>
<li>In each group, we find the first non-null identifier for every other data provider (say, B through Z). <i>However</i>, if we find multiple identifiers in a group for, say, Provider B, we set provider B's identifier to NULL instead.</li>
<li>The rows aggregated for Provider A from this grouping process are added to the "knockout list".</li>
<li>This process repeats until we exhaust the list of data providers or there are no more rows in the "preliminary set".</li>
</ol>
</blockquote>

We ran into a few issues with this setup as our pipelines matured and we began to rely on their outputs more:

1. Transitive relationships between newly-synchronized identifiers were not properly followed (and we also had a bug in the synchronizer logic that exacerbated this).
2. It was inherently dependent on order of tabular records AND the order of the list of identifiers.
3. It was (mostly) pure pyspark/Spark code, which made it hard to debug and check edgecases within our existing pipelines without running them outright. 
4. Unresolved identifiers began to pile up in our flagged tables.

Of these issues, #3 prevented us from really debugging any underlying bad results we saw. Thus, we chose to build out a new solution in pure Python: the [ObjectResolver][glass_onion.resolver.ObjectResolver] class.

At a high level, [ObjectResolver][glass_onion.resolver.ObjectResolver] relies on a graph to connect identifiers based on sync results and identify conflicts. Within the graph, every `(provider, ID)` pair is a vertex, and synchronization of two identifiers of different providers is represented by an edge. The resolver logic (in [ObjectResolver.resolve()][glass_onion.resolver.ObjectResolver.resolve]) builds graph "components" out of these vertices and edges to represent unified objects. New sync results are considered "proposed" components to an existing graph and are dealt with like so:

- If one or more vertices from the component exist in the graph but the edge does not yet:
    - If one vertex already has an edge to a vertex that has the same data provider for the other vertex, do not add the proposed edge to the graph and flag it as a duplicate.
    - If adding the component would produce a transitive conflict (IE: connecting a Skillcorner ID to a Statsbomb ID results in two different Scoutastic IDs), flag the proposed component as a duplicate.
    - Otherwise, add the proposed component to the graph.
- If the records within the component exist in the group already:
    - If they all exist in a larger component, do not add the new/smaller component to the graph.
    - If they all exists in a smaller component, remove the smaller component from the graph and add the new/larger component.


By design, graph components may hold AT MOST one ID per provider.