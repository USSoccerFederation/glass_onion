# Methodology

Let's set the stage with some example code (from [Getting Started](./getting_started.md)):

```python linenums="1"
from glass_onion import PlayerSyncEngine

engine = PlayerSyncEngine(
    content=[impect_content, statsbomb_content],
    verbose=True
)
result = engine.synchronize() 
```

In general, Glass Onion takes a list of [SyncableContent][glass_onion.engine.SyncableContent] and uses the logic in a [SyncEngine][glass_onion.engine.SyncEngine] to sync one pair at a time. The results of all pairs are then merged together and deduplicated. Each object type corresponds to a subclass of [SyncEngine][glass_onion.engine.SyncEngine] that overrides [synchronize_pair()][glass_onion.engine.SyncEngine.synchronize_pair] to define how pairs are synchronized in [synchronize()][glass_onion.engine.SyncEngine.synchronize], which contains wrapper logic for the entire process. 

There are three distinct layers within [synchronize()][glass_onion.engine.SyncEngine.synchronize]'s wrapper logic:

1. The aforementioned sync process that results in a data frame of synced identifiers. How each object type is handled is described below.
2. Collect remaining unsynced rows and run the sync process on those. Append any newly synced rows to the result dataframe from Layer 1.
3. Append any remaining unsynced rows to the bottom of the result data frame.

This result dataframe is then deduplicated: by default, the result dataframe is grouped by the specific columns defined in [SyncEngine][glass_onion.engine.SyncEngine] and the first non-null result is selected for each data provider's identifier field. 

## Match

**NOTE**: Match synchronization can be also done using competition context (IE: columns `competition_id` and `season_id`, which are assumed to already be synchronized across providers) via `use_competition_context` (more details on `use_competition_context` in [MatchSyncEngine.init()][glass_onion.match.MatchSyncEngine.__init__] and the concept of "higher-order" object types on our [home page](./index.md)).

1. Attempt to join pair using `match_date`, `home_team_id`, and `away_team_id`.
2. Account for matches with different dates across data providers (timezones, TV scheduling, etc) by adjusting `match_date` in one dataset in the pair by -3 to 3 days, then attempting synchronization using `match_date`, `home_team_id`, and `away_team_id` again. This process is then repeated for the other dataset in the pair.
3. Account for matches postponed to a different date outside the [-3, 3] day range by attempting synchronization using `matchday`, `home_team_id`, and `away_team_id`.

## Team

**NOTE**: Team synchronization can be also done using competition context (IE: columns `competition_id` and `season_id`, which are assumed to already be synchronized across providers) via `use_competition_context` (more details on `use_competition_context` in [TeamSyncEngine.init()][glass_onion.team.TeamSyncEngine.__init__] and the concept of "higher-order" object types on our [home page](./index.md)).

1. Attempt to join pair simply on `team_name`.
2. With remaining records, attempt to match via cosine similarity using a minimum threshold of 75% similarity.
3. For any remaining records, attempt to match via cosine similarity using no minimum similarity threshold.

## Player

**NOTE**: [PlayerSyncEngine][glass_onion.player.PlayerSyncEngine] ignores syncable columns that have unreliable data (IE: NULLs/NAs in `jersey_number` or `birth_date`). The process below describes the best-case scenario. Please set `verbose_log=True` when creating a [PlayerSyncEngine][glass_onion.player.PlayerSyncEngine] instance to see the full synchronization process.

1. Attempt to join pair using `player_name` with a minimum 75% cosine similarity threshold for player name. Additionally, require that `jersey_number` and `team_id` are equal for matches that meet the similarity threshold.
2. Account for players with different birth dates across data providers (timezones, human error, etc) by adjusting `birth_date` in one dataset in the pair by -1 to 1 days and/or swapping the month and day, then attempting synchronization using `birth_date`, `team_id`, and a combination of `player_name` and `player_nickname`. This process is then repeated for the other dataset in the pair. 
3. Attempt to join remaining records using combinations of `player_name` and `player_nickname` with a minimum 75% cosine similarity threshold for player name. Additionally, require that `team_id` is equal for matches that meet the similarity threshold.
4. Attempt to join remaining records using "naive similarity": looking for normalized parts of one record's `player name` (or `player_nickname`) that exist in another's. Additionally, require that `team_id` is equal for matches found via this method.
5. Attempt to join remaining records using combinations of `player_name` and `player_nickname` with no minimum cosine similarity threshold. Additionally, require that `team_id` is equal.
