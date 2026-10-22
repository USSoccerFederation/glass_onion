"""
Times `SyncEngine.synchronize()` on the fixture datasets in `tests/fixtures`.

Timings depend on the machine, so these tests record numbers instead of asserting on them. Run them serially and write
a markdown report to compare before and after a change:

    uv run pytest tests/performance -n0 --performance-report=performance-report.md
"""

import contextlib
import io
from pathlib import Path
import time

import pandas as pd
import pytest

from glass_onion.match import MatchSyncEngine
from glass_onion.player import PlayerSyncEngine
from glass_onion.team import TeamSyncEngine
from tests.utils import utils_create_syncables

FIXTURE_DATA_PATH = Path(__file__).resolve().parent.parent / "fixtures"

ENGINES = {
    "player": PlayerSyncEngine,
    "match": MatchSyncEngine,
    "team": TeamSyncEngine,
}

RUNS = 3

DATASETS = [
    (object_type, path.name)
    for object_type in ENGINES.keys()
    for path in sorted((FIXTURE_DATA_PATH / object_type).glob("*.csv"))
]


@pytest.mark.parametrize("object_type, file_name", DATASETS)
def test_sync_fixture_timing(object_type: str, file_name: str, record_property):
    # read the same way as tests/functional/test_synchronize.py: utils_create_syncables() expects numeric IDs
    dataset = pd.read_csv(FIXTURE_DATA_PATH / object_type / file_name)

    times = []
    for _ in range(RUNS):
        # build fresh syncables each run: synchronize() can modify them, and engines cache results
        engine = ENGINES[object_type](
            utils_create_syncables(dataset.copy(), object_type)
        )
        with contextlib.redirect_stdout(io.StringIO()):
            start = time.perf_counter()
            result = engine.synchronize()
            times.append(time.perf_counter() - start)

    assert len(result.data) > 0

    record_property(
        "performance",
        {
            "section": f"Sync (best of {RUNS} runs)",
            "row": {
                "Object type": object_type,
                "Dataset": file_name,
                "Time (s)": min(times),
                "Rows": len(result.data),
            },
            "compare": "Time (s)",
            "total": True,
        },
    )
