"""Run one web job in a separate process, keeping the HTTP server responsive."""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from ContentTranscoding import ContentTranscoding, RESULT_COLUMNS

EVENT_PREFIX = "@event:"


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value == math.inf else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def emit(event):
    print(EVENT_PREFIX + json.dumps(json_safe(event), ensure_ascii=False, allow_nan=False), flush=True)


def execute(job_directory: Path) -> int:
    global EVENT_PREFIX
    manifest = json.loads((job_directory / "request.json").read_text(encoding="utf-8"))
    if manifest.get("event_token"):
        EVENT_PREFIX = f"@frame-event/{manifest['event_token']}:"
    groups = defaultdict(list)
    for entry in manifest["files"]:
        source = Path(entry["path"])
        groups[source.parent].append(source)
    records = []
    failed = False
    completed = 0
    for group_index, (parent, sources) in enumerate(groups.items()):
        def callback(event):
            event = dict(event)
            if "position" in event:
                event["position"] += completed
            event["total"] = len(manifest["files"])
            emit(event)

        args = argparse.Namespace(path=str(parent), **manifest["settings"])
        transcoder = ContentTranscoding(
            args, output_directory=job_directory / "outputs" / str(group_index),
            event_callback=callback,
        )
        try:
            succeeded = transcoder.run(sources)
            failed = failed or not succeeded
            group_records = transcoder._results
        except (OSError, ValueError) as error:
            failed = True
            group_records = []
            for source in sources:
                record = dict.fromkeys(RESULT_COLUMNS)
                record.update(file_name=source.name, status="failed", reason=str(error), attempts=0)
                group_records.append(record)
        for source, record in zip(sources, group_records):
            output = transcoder.done_path / source.name
            record = {**record, "source_path": str(source), "output_path": None}
            if record["status"] == "accepted" and output.is_file():
                record["output_path"] = str(output)
            elif record["status"] == "accepted":
                record.update(status="failed", reason="Result file was not saved")
                failed = True
            records.append(record)
        completed += len(sources)
        emit({"event": "results", "records": records, "completed": completed})
    records = json_safe(records)
    result_path = job_directory / "results.json"
    result_path.write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (job_directory / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=(*RESULT_COLUMNS, "source_path", "output_path"))
        writer.writeheader()
        writer.writerows(records)
    emit({"event": "finished", "status": "failed" if failed else "completed", "records": records})
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(execute(Path(sys.argv[1]).resolve()))
    except Exception as error:
        emit({"event": "error", "message": str(error)})
        raise
