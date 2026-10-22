import json
from pathlib import Path
import platform
import subprocess
from typing import Optional

import pytest

# tests record report rows as a ("performance", {...}) user property: see tests/performance. The value has
# * section (str): the report table the row goes in
# * row (dict): column name -> value
# * compare (str, optional): a timing column to compare with the baseline
# * total (bool, optional): whether the section gets a total row for `compare`
PERFORMANCE_PROPERTY = "performance"

# slowdowns over this percentage are flagged in the report
SLOWDOWN_WARNING = 25.0

# recorded values (plus each test's nodeid, used to match rows with the baseline) from test reports in this process
performance_rows: list[dict] = []


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--performance-report",
        metavar="PATH",
        default=None,
        help="write timings recorded by tests/performance to a markdown file at PATH",
    )
    parser.addoption(
        "--performance-json",
        metavar="PATH",
        default=None,
        help="write timings recorded by tests/performance to a JSON file at PATH, for use as a baseline",
    )
    parser.addoption(
        "--performance-baseline",
        metavar="PATH",
        default=None,
        help="a JSON file written by --performance-json to compare --performance-report timings with",
    )


def pytest_runtest_logreport(report: pytest.TestReport):
    # with pytest-xdist this runs on the controller too, which receives each worker's reports and user properties
    if report.when != "call":
        return

    for name, value in report.user_properties:
        if name == PERFORMANCE_PROPERTY:
            row = {"Result": "✅" if report.passed else "❌", **value["row"]}
            performance_rows.append({**value, "nodeid": report.nodeid, "row": row})


def git_info() -> dict[str, Optional[str]]:
    # describe the checkout glass_onion was imported from, which is the baseline's checkout when run against it
    import glass_onion

    def git(*args: str) -> Optional[str]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=Path(glass_onion.__file__).resolve().parent,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    ref = git("rev-parse", "--abbrev-ref", "HEAD")
    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        # a detached checkout (IE: a pull request's merge commit) has no branch name
        "ref": None if ref == "HEAD" else ref,
    }


def format_cell(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def format_change(value: Optional[float], baseline: Optional[float]) -> str:
    if value is None or not baseline:
        return "—"
    change = (value - baseline) / baseline * 100
    return f"{change:+.1f}%" + (" ⚠️" if change > SLOWDOWN_WARNING else "")


def format_table(columns: list[str], rows: list[dict]) -> list[str]:
    return [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ] + [
        "| " + " | ".join(format_cell(row.get(c, "")) for c in columns) + " |"
        for row in rows
    ]


def render_performance_report(
    rows: list[dict], baseline: Optional[dict], baseline_missing: bool = False
) -> str:
    this_run = git_info()
    lines = [
        "## Performance Report",
        "",
        f"This run: commit `{this_run['commit'] or 'unknown'}`, Python {platform.python_version()} on {platform.system()} ({platform.machine()})",
        "",
    ]

    baseline_results = {}
    baseline_label = "baseline"
    if baseline is not None:
        baseline_results = baseline["results"]
        baseline_label = baseline["ref"] or baseline_label
        lines += [
            f"Baseline: `{baseline_label}` at commit `{baseline['commit'] or 'unknown'}`. Slowdowns vs baseline over {SLOWDOWN_WARNING:.0f}% marked ⚠️.",
            "",
        ]
    elif baseline_missing:
        lines += ["No baseline results were available.", ""]

    if len(rows) == 0:
        return "\n".join(lines + ["No performance results were recorded.", ""])

    sections: dict[str, list[dict]] = {}
    for record in rows:
        sections.setdefault(record["section"], []).append(record)

    for section, records in sections.items():
        display_rows = []
        compared = []
        for record in records:
            column = record.get("compare")
            if baseline is None or column is None:
                display_rows.append(record["row"])
                continue

            base = baseline_results.get(record["nodeid"], {}).get("row", {}).get(column)
            compared.append((column, record["row"].get(column), base))
            # put the baseline value and the change right after the compared column
            display_row = {}
            for k, v in record["row"].items():
                display_row[k] = v
                if k == column:
                    display_row[f"{baseline_label} (s)"] = "—" if base is None else base
                    display_row["Change"] = format_change(v, base)
            display_rows.append(display_row)

        columns = list(dict.fromkeys(k for row in display_rows for k in row))
        lines += [f"### {section}", ""] + format_table(columns, display_rows)

        total_column = next(
            (r["compare"] for r in records if r.get("total") and r.get("compare")),
            None,
        )
        if total_column is not None:
            total = {
                columns[0]: "**Total**",
                total_column: f"**{sum(r['row'].get(total_column, 0.0) for r in records):.3f}**",
            }
            if len(compared) > 0:
                # only rows with a baseline value count toward the change
                matched = [
                    (v, b) for _, v, b in compared if v is not None and b is not None
                ]
                total[f"{baseline_label} (s)"] = f"**{sum(b for _, b in matched):.3f}**"
                total["Change"] = (
                    f"**{format_change(sum(v for v, _ in matched), sum(b for _, b in matched))}**"
                    if len(matched) > 0
                    else "—"
                )
            lines += format_table(columns, [total])[2:]
        lines.append("")

    return "\n".join(lines)


def pytest_sessionfinish(session: pytest.Session):
    # pytest-xdist workers also finish a session: only the controller has every worker's rows
    if hasattr(session.config, "workerinput"):
        return

    json_path = session.config.getoption("--performance-json")
    if json_path is not None:
        Path(json_path).write_text(
            json.dumps(
                {
                    **git_info(),
                    "python": platform.python_version(),
                    "results": {r["nodeid"]: r for r in performance_rows},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    report_path = session.config.getoption("--performance-report")
    if report_path is not None:
        baseline_path = session.config.getoption("--performance-baseline")
        baseline = None
        if baseline_path is not None and Path(baseline_path).exists():
            baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))

        Path(report_path).write_text(
            render_performance_report(
                performance_rows,
                baseline,
                baseline_missing=baseline_path is not None and baseline is None,
            ),
            encoding="utf-8",
        )
