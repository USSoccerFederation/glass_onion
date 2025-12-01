# glass_onion


[![PyPI Latest Release](https://img.shields.io/pypi/v/glass_onion.svg)](https://pypi.org/project/glass_onion/)
![](https://img.shields.io/github/license/PySport/glass_onion)
![](https://img.shields.io/pypi/pyversions/glass_onion)


## Table of Contents

- [Summary](#summary)
- [Methodology](#methodology)
- [Where to get it](#where-to-get-it)
- [Documentation](#documentation)
- [Contributing to glass_onion](#contributing-to-glass_onion)
- [License](#license)
- [Why the name `Glass Onion`?](#why-the-name-glass-onion)


## Summary

Glass Onion aims to do one thing -- synchronizing soccer data object identifiers -- extremely well. Currently, this package supports the following objects:

| Object | All Considered Columns | Notes |
|----------|--------|------|
| Match    | `match_date`, `matchday`, `home_team_id`, `away_team_id` | Works best with a single competition. |
| Team     | `team_name` | Works best with a single competition. |
| Player   | `jersey_number`, `team_id`, `player_name`, `player_nickname`, `birth_date`  | Works best within a single match. |

Any identifiers _other than the ones being synchronized_ are assumed to be universal across data providers (e.g. `team_id` when synchronizing players).

When building an object identifier sync pipeline, there are a bunch of other tasks that you may need to do that Glass Onion does not provide support for: deduplication, false positive detection, etc. A suggested workflow can be found in the **[integration guide](INTEGRATION.md)**.

Inspiration and some methodology for this package comes from https://unravelsports.com/post.html?id=2022-07-11-player-id-matching-system.


## Methodology

In general, Glass Onion takes a list of `SyncableContent` and uses the logic in a `SyncEngine` to sync one pair at a time. The results of all pairs are then merged together and deduplicated. Each object type corresponds to a subclass of `SyncEngine` that overrides `synchronize_pair()` to define how pairs are synchronized in `synchronize()`, which contains wrapper logic for the entire process. 

There are three distinct layers within `synchronize()`'s wrapper logic:

1. The aforementioned sync process that results in a data frame of synced identifiers. How each object type is handled is described below.
2. Collect remaining unsynced rows and run the sync process on those. Append any newly synced rows to the result dataframe from Layer 1.
3. Append any remaining unsynced rows to the bottom of the result data frame.

This result dataframe is then deduplicated: by default, the result dataframe is grouped by the specific columns defined in `SyncEngine` and the first non-null result is selected for each data provider's identifier field. 

### Match

**NOTE**: Match synchronization can be also done using competition context (IE: columns `competition_id` and `season_id`, which are assumed to already be synchronized across providers) via `use_competition_context`.

1. Attempt to join pair using `match_date`, `home_team_id`, and `away_team_id`.
2. Account for matches with different dates across data providers (timezones, TV scheduling, etc) by adjusting `match_date` in one dataset in the pair by -3 to 3 days, then attempting synchronization using `match_date`, `home_team_id`, and `away_team_id` again. This process is then repeated for the other dataset in the pair.
3. Account for matches postponed to a different date outside the [-3, 3] day range by attempting synchronization using `matchday`, `home_team_id`, and `away_team_id`.

### Team

**NOTE**: Team synchronization can be also done using competition context (IE: columns `competition_id` and `season_id`, which are assumed to already be synchronized across providers) via `use_competition_context`.

1. Attempt to join pair simply on `team_name`.
2. With remaining records, attempt to match via cosine similarity using a minimum threshold of 75% similarity.
3. For any remaining records, attempt to match via cosine similarity using no minimum similarity threshold.

### Player

**NOTE**: `PlayerSyncEngine` ignores syncable columns that have unreliable data (IE: NULLs/NAs in `jersey_number` or `birth_date`). The process below describes the best-case scenario. Please set `verbose_log=True` when creating a `PlayerSyncEngine` instance to see the full synchronization process.

1. Attempt to join pair using `player_name` with a minimum 75% cosine similarity threshold for player name. Additionally, require that `jersey_number` and `team_id` are equal for matches that meet the similarity threshold.
2. Account for players with different birth dates across data providers (timezones, human error, etc) by adjusting `birth_date` in one dataset in the pair by -1 to 1 days and/or swapping the month and day, then attempting synchronization using `birth_date`, `team_id`, and a combination of `player_name` and `player_nickname`. This process is then repeated for the other dataset in the pair. 
3. Attempt to join remaining records using combinations of `player_name` and `player_nickname` with a minimum 75% cosine similarity threshold for player name. Additionally, require that `team_id` is equal for matches that meet the similarity threshold.
4. Attempt to join remaining records using "naive similarity": looking for normalized parts of one record's `player name` (or `player_nickname`) that exist in another's. Additionally, require that `team_id` is equal for matches found via this method.
5. Attempt to join remaining records using combinations of `player_name` and `player_nickname` with no minimum cosine similarity threshold for player name. Additionally, require that `team_id` is equal.

## Installation

The source code is hosted on GitHub at: [https://github.com/USSoccerFederation/glass_onion](https://github.com/USSoccerFederation/glass_onion).

The easiest way to install Glass Onion is via **pip**:

```bash
pip install glass_onion
```

You can also install from GitHub for the latest updates:

```sh
pip install git+https://github.com/USSoccerFederation/glass_onion.git
```

For more details, refer to the [installation guide](INSTALLATION.md).

## Documentation

TBD.

## Contributing to glass_onion
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome. An overview on how to contribute can be found in the **[contributing guide](CONTRIBUTING.md)**.

## License

Glass Onion is distributed under the terms of the [BSD 3 license](LICENSE).

## Why the name `Glass Onion`?

Syncing identifiers is often fragile because of the human error involved in recording object names and metadata. It also often takes multiple ~~approaches~~ layers to synchronize objects. So: glass onion.