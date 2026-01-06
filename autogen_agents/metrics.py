from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


def _load_records(log_dir: Path) -> list[dict]:
    records = []
    if not log_dir.exists():
        return records
    for path in log_dir.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def summarize_logs(log_dir: Path | str = "data/logs", session_id: str | None = None) -> dict:
    log_dir = Path(log_dir)
    records = _load_records(log_dir)
    if session_id:
        records = [
            record
            for record in records
            if record.get("metadata", {}).get("session_id") == session_id
        ]

    if not records:
        return {"total_interactions": 0}

    durations = [
        record.get("metadata", {}).get("actual_sec")
        for record in records
        if record.get("metadata", {}).get("actual_sec") is not None
    ]
    costs = [
        record.get("metadata", {}).get("cost")
        for record in records
        if record.get("metadata", {}).get("cost") is not None
    ]
    escalations = [
        record
        for record in records
        if record.get("metadata", {}).get("escalated_to")
    ]
    categories = {}
    for record in records:
        task_type = record.get("metadata", {}).get("task_type", "unknown")
        categories[task_type] = categories.get(task_type, 0) + 1

    return {
        "total_interactions": len(records),
        "avg_handle_time_sec": round(mean(durations), 2) if durations else None,
        "avg_cost": round(mean(costs), 2) if costs else None,
        "escalations": len(escalations),
        "task_breakdown": categories,
    }
