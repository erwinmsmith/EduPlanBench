from __future__ import annotations

from pathlib import Path

from eduplanbench.core.io import ensure_dir, write_csv, write_json
from eduplanbench.core.schema import MetricReport


def write_report(report: MetricReport, output_dir: str | Path) -> None:
    out = ensure_dir(output_dir)
    write_json(out / "metrics.json", report)
    rows = [{"scope": "overall", **report.metrics}]
    for track, metrics in report.by_track.items():
        rows.append({"scope": track, **metrics})
    write_csv(out / "metrics.csv", rows)
    with (out / "report.md").open("w", encoding="utf-8") as fh:
        fh.write(f"# EduPlanBench Report\n\nRun: `{report.run_id}`\n\n")
        fh.write(f"Episodes: {report.metadata.get('episodes', 0)}\n\n")
        fh.write("## Overall Metrics\n\n")
        for key, value in sorted(report.metrics.items()):
            fh.write(f"- `{key}`: {value:.4f}\n")
        if report.by_track:
            fh.write("\n## By Track\n\n")
            for track, metrics in sorted(report.by_track.items()):
                fh.write(f"### {track}\n\n")
                for key, value in sorted(metrics.items()):
                    fh.write(f"- `{key}`: {value:.4f}\n")
                fh.write("\n")
