import pandas as pd
from glass_onion.engine import SyncableContent


def utils_transform_provider_data(
    dataset: pd.DataFrame, provider: str, object_type: str
) -> pd.DataFrame:
    generic_id = f"provider_{object_type}_id"
    specific_id = f"{provider}_{object_type}_id"

    dataset.rename({generic_id: specific_id}, axis=1, inplace=True)
    dataset[specific_id] = dataset[specific_id].round().astype("Int64").astype(str)
    dataset.drop(["data_provider"], axis=1, inplace=True)
    return dataset


def utils_create_syncables(
    dataset: pd.DataFrame, object_type: str
) -> list[SyncableContent]:
    grouped = dataset.groupby("data_provider")
    syncables = [
        SyncableContent(
            provider=p,
            data=utils_transform_provider_data(
                dataset.loc[dataset.index.isin(d),], p, object_type
            ),
            object_type=object_type,
        )
        for p, d in grouped.groups.items()
    ]
    syncables = [k for k in syncables if len(k.data) > 0]
    return syncables
