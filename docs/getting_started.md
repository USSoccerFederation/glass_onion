# Getting Started

Need a very quick introduction to Glass Onion? Here's an example of identifier synchronization across two public datasets for the 2023-2024 1. Bundesliga season: [Statsbomb](https://github.com/statsbomb/open-data) and [Impect](https://github.com/ImpectAPI/open-data). If you're already familiar with how Glass Onion works, feel free to move on to the [integration guide](./integrating.md), where you can learn how to make Glass Onion part of a production workflow.

## Installing Glass Onion

The easiest way to install Glass Onion is via `uv` or `pip`. 

```bash
uv add glass_onion
pip install glass_onion
```

The [installation guide](./installation.md) has more options if you need them.

## Example: Player Synchronization 

### Loading data

First, let's use [`kloppy`](https://kloppy.pysport.org/) to retrieve sample data for a given match from both Impect and Statsbomb. For simplicity, we've picked out the August 19, 2023 fixture between Bayer Leverkusen and RB Leipzig from both datasets.

```python linenums="1" exec="true" source="above" session="getting-started"

from kloppy import impect, statsbomb

impect_dataset = impect.load_open_data(match_id="122839")
statsbomb_dataset = statsbomb.load_open_data(match_id="3895052")
```

We can pull out the player information from both of these event datasets into Pandas dataframes. For each, we'll have to iterate through the teams and pull specific fields for each of the players.

```python linenums="1" exec="true" source="above" session="getting-started"
import pandas as pd 

def get_players(dataset, provider):
    return pd.DataFrame([
        {
            f"{provider}_player_id": player.player_id,
            "jersey_number": str(player.jersey_no),
            "team_id": team.team_id,
            "team_name": team.name,
            "player_name": player.full_name,
            "player_nickname": player.name
        }
        for team in dataset.metadata.teams
        for player in team.players
    ])

impect_player_df = get_players(
    impect_dataset, 
    provider="impect"
)
statsbomb_player_df = get_players(
    statsbomb_dataset, 
    provider="statsbomb"
)
```

### Assigning unified team identifiers

We need to unify team identifiers across these two dataframes so Glass Onion can properly use `team_id` in its synchronization logic. With just two teams, we can do this manually (as below) by simply setting RB Leipzig's `team_id` to RBL and Bayer Leverkusen's to B04. If we wanted to do this across the entire competition, we could build a [more complex](./integrating.md) workflow with Glass Onion.

```python linenums="1" exec="true" source="above" session="getting-started"

import numpy as np
impect_player_df["team_id"] = np.select(
    [
        impect_player_df["team_id"] == '41',
        impect_player_df["team_id"] == '37',
    ],
    [
        "B04",
        "RBL"
    ],
    default=impect_player_df["team_id"]
)

statsbomb_player_df["team_id"] = np.select(
    [
        statsbomb_player_df["team_id"] == '904',
        statsbomb_player_df["team_id"] == '182',
    ],
    [
        "B04",
        "RBL"
    ],
    default=statsbomb_player_df["team_id"]
)
```

### Wrapping in `PlayerSyncableContent` 

Now, we just have to wrap these two dataframes in [PlayerSyncableContent][glass_onion.player.PlayerSyncableContent] instances so they can be used in [PlayerSyncEngine][glass_onion.player.PlayerSyncEngine].

```python linenums="1" exec="true" source="above" session="getting-started"

from glass_onion import PlayerSyncableContent

impect_content = PlayerSyncableContent(
    provider="impect",
    data=impect_player_df
)

statsbomb_content = PlayerSyncableContent(
    provider="statsbomb",
    data=statsbomb_player_df
)
```

### Using `PlayerSyncEngine`

Once you have two [PlayerSyncableContent][glass_onion.player.PlayerSyncableContent] instances, you can now synchronize them with [PlayerSyncEngine.synchronize][glass_onion.engine.SyncEngine.synchronize]!

```python linenums="1" exec="true" source="above" session="getting-started"

from glass_onion import PlayerSyncEngine

engine = PlayerSyncEngine(
    content=[impect_content, statsbomb_content],
    verbose=True
)
result = engine.synchronize()
```

`result` is a `PlayerSyncableContent` object, so you can view the dataframe containing synchronized identifiers by dumping out its `data` field:

```python linenums="1"
result.data.head()
```

```python exec="true" html="true" session="getting-started"
import re 
html_result = re.sub("class=\"dataframe\"", "", result.data.head().to_html(border="0", index=False, classes=''))
print(f"""<div class="md-typeset__scrollwrap"><div class="md-typeset__table">{html_result}</div></div>""")
```
You can then join other dataframes using `result.data` to link the Impect and Statsbomb datasets together. 

### Future: Using the synced identifiers

Let's say you want to compare a player's Statsbomb Shot xG to their Impect Packing xG. We'll need to parse out both KPIs from their JSON files:

```python linenums="1" exec="true" source="above" session="getting-started"
# Kloppy doesn't cover this case, so we have to parse both JSON files ourselves.
import json
import requests
import pandas as pd

impect_player_match = json.loads(
    requests.get(
        "https://raw.githubusercontent.com/ImpectAPI/open-data/refs/heads/main/data/player_kpis/player_kpis_122839.json"
    ).content
)
impect_player_match_list = []
for t in ["squadHome", "squadAway"]:
    impect_player_match_list = [
        {
            **p,
            "team_id": impect_player_match[t]["id"],
            "match_id": impect_player_match["matchId"],
        }
        for p in impect_player_match[t]["players"]
    ]

impect_player_match_df = pd.DataFrame(impect_player_match_list).explode("kpis")
impect_player_match_df["kpi_id"] = impect_player_match_df["kpis"].apply(
    lambda x: x["kpiId"]
)
impect_player_match_df["kpi_value"] = impect_player_match_df["kpis"].apply(
    lambda x: x["value"]
)
impect_player_match_df.drop("kpis", axis=1, inplace=True)
impect_player_match = (
    impect_player_match_df[impect_player_match_df["kpi_id"] == 83]
    .groupby(["match_id", "id"], as_index=False)
    .kpi_value.sum()
)
impect_player_match.rename(
    {
        "match_id": "impect_match_id",
        "id": "impect_player_id",
        "kpi_value": "impect_packing_xg",
    },
    axis=1,
    inplace=True,
)
impect_player_match["impect_player_id"] = impect_player_match[
    "impect_player_id"
].astype(str)
impect_player_match
```
```python exec="true" html="true" session="getting-started"
import re 
html_result = re.sub("class=\"dataframe\"", "", impect_player_match.head().to_html(border="0", index=False, classes=''))
print(f"""<div class="md-typeset__scrollwrap"><div class="md-typeset__table">{html_result}</div></div>""")
```

```python linenums="1" exec="true" source="above" session="getting-started"
import pandas as pd

statsbomb_player_match_df = pd.read_json(
    "https://raw.githubusercontent.com/statsbomb/open-data/refs/heads/master/data/events/3895052.json"
)
statsbomb_player_match_df["match_id"] = "3895052"
statsbomb_player_match_df = statsbomb_player_match_df[
    statsbomb_player_match_df["shot"].notna()
]
statsbomb_player_match_df["player_id"] = statsbomb_player_match_df["player"].apply(
    lambda x: x["id"]
)
statsbomb_player_match_df["player_name"] = statsbomb_player_match_df["player"].apply(
    lambda x: x["name"]
)
statsbomb_player_match_df["team_id"] = statsbomb_player_match_df["team"].apply(
    lambda x: x["id"]
)
statsbomb_player_match_df["team_name"] = statsbomb_player_match_df["team"].apply(
    lambda x: x["name"]
)
statsbomb_player_match_df["shot_statsbomb_xg"] = statsbomb_player_match_df[
    "shot"
].apply(lambda x: x["statsbomb_xg"])
statsbomb_player_match_df = statsbomb_player_match_df[
    ["match_id", "team_id", "player_id", "shot_statsbomb_xg"]
]
statsbomb_player_match = statsbomb_player_match_df.groupby(
    ["match_id", "player_id"], as_index=False
).shot_statsbomb_xg.sum()
statsbomb_player_match["player_id"] = statsbomb_player_match["player_id"].astype(str)
statsbomb_player_match.rename(
    {
        "match_id": "statsbomb_match_id",
        "player_id": "statsbomb_player_id",
        "shot_statsbomb_xg": "statsbomb_shot_xg",
    },
    axis=1,
    inplace=True,
)
statsbomb_player_match
```
```python exec="true" html="true" session="getting-started"
import re 
html_result = re.sub("class=\"dataframe\"", "", statsbomb_player_match.head().to_html(border="0", index=False, classes=''))
print(f"""<div class="md-typeset__scrollwrap"><div class="md-typeset__table">{html_result}</div></div>""")
```

But once we have both datasets, we can join them easily:

```python linenums="1" exec="true" source="above" session="getting-started"
composite_df = pd.merge(
    left=impect_player_match, right=result.data, on="impect_player_id", how="outer"
)

composite_df = pd.merge(
    left=composite_df,
    right=statsbomb_player_match,
    on="statsbomb_player_id",
    how="outer",
)

composite_result = composite_df[
    [
        "player_name",
        "team_id",
        "jersey_number",
        "impect_player_id",
        "statsbomb_player_id",
        "impect_packing_xg",
        "statsbomb_shot_xg",
    ]
].sort_values(by=["impect_packing_xg", "statsbomb_shot_xg"], ascending=False)

composite_result.head()
```

```python exec="true" html="true" session="getting-started"
import re 
html_result = re.sub("class=\"dataframe\"", "", composite_result.head().to_html(border="0", index=False, classes=''))
print(f"""<div class="md-typeset__scrollwrap"><div class="md-typeset__table">{html_result}</div></div>""")
```