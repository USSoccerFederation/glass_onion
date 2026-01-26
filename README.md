# glass_onion

[![PyPI Latest Release](https://img.shields.io/pypi/v/glass_onion.svg)](https://pypi.org/project/glass_onion/)
![](https://img.shields.io/github/license/USSoccerFederation/glass_onion)
![](https://img.shields.io/pypi/pyversions/glass_onion)


## Table of Contents

- [Summary](#summary)
- [Installation](#installation)
- [Documentation](#documentation)
- [Background](#background)
- [Methodology](#methodology)
- [Contributing](#contributing)
- [License](#license)
- [Why the name `Glass Onion`?](#why-the-name-glass-onion)


## Summary

Glass Onion aims to do one thing -- synchronizing soccer data object identifiers -- extremely well. Currently, this package supports the following objects:

| Object | All Considered Columns | "Higher-Order" Object Type(s) | Notes |
|----------|--------|------|-------|
| Match    | `match_date`, `matchday`, `home_team_id`, `away_team_id` | Competitions, Seasons, Teams |  Works best with a single competition. |
| Team     | `team_name` | Competitions and Seasons |  Works best with a single competition. |
| Player   | `jersey_number`, `team_id`, `player_name`, `player_nickname`, `birth_date` | Teams | Works best within a single match. |

Any identifiers _other than the ones being synchronized_ are assumed to be universal across data providers (e.g. `team_id` when synchronizing players). Why? Each object type depends on a "higher-order" object type to have unified identifiers in order to reduce the search space of potential matches (more details in [Integrating Glass Onion: Step 2](./docs/integrating.md#step-2-glass-onion-synchronization)). Having unified competitions and seasons unlocks the synchronizations for teams, which unlocks that process for matches, and the same for players.
When building an object identifier sync pipeline, there are a bunch of other tasks that you may need to do that Glass Onion does not provide support for: deduplication, false positive detection, etc. A example workflow can be found in the **[integration guide](docs/integrating.md)**. Our **[Getting Started page](getting_started.md)** provides a simpler example that allows you to get up and running with Glass Onion quickly.

**Please note**: this package simply combines common synchronization techniques to get the closest match for two objects from different data providers. The _accuracy_ of these matches is not necessarily guaranteed. Please review matches for accuracy (either as part of your pipeline or manually).

## Installation

The source code is hosted on GitHub at: [https://github.com/USSoccerFederation/glass_onion](https://github.com/USSoccerFederation/glass_onion).

The easiest way to install Glass Onion is via **uv** or **pip**:

```bash
uv install glass_onion
pip install glass_onion
```

You can also install the package from GitHub for the latest updates:

```sh
uv pip install "git+https://github.com/USSoccerFederation/glass_onion"
pip install git+https://github.com/USSoccerFederation/glass_onion.git
```

For more details, refer to the [installation guide](docs/installation.md).

## Documentation

Please see the [docs](docs/index.md).

## Background

The idea for this package and its public release started with [this 2022 blog post](https://unravelsports.com/post.html?id=2022-07-11-player-id-matching-system) by [@UnravelSports](https://github.com/@UnravelSports). Identifier synchronization is one of the most common problems that soccer analytics groups run into, mostly because it seems unique to soccer (at least, across the major sports): in other (read: American) sports, the organizing body (NFL, MLB, NBA, etc) uniquely identifies players and forces (or at least seems to) data providers to use those identifiers in their datasets. Every club that has multiple data subscriptions has to build their own solution to this problem (or manually synchronize players via spreadsheet) that fits into their ETL system, but few are publicly discussed and open-source (the main exception being that of Parma Calcio in Italy: [https://github.com/parmacalcio1913/players-matcher](https://github.com/parmacalcio1913/players-matcher)). 

Our hope is this package will help new data analysts and analytics groups get up to speed more quickly and deliver more robust reports that integrate all of their data sources.

## Methodology

Please see the [docs](docs/methodology.md).

## Contributing

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome. More information can be found in the **[contributing guide](docs/contributing.md)**.

## License

Glass Onion is distributed under the terms of the [BSD 3 license](LICENSE).

## Why the name `Glass Onion`?

Syncing identifiers is often fragile because of the human error involved in recording object names and metadata. It also often takes multiple ~~approaches~~ layers to synchronize objects. So: glass onion.
