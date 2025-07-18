from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

_LOG_PATH = Path("data/logs")
_LOG_PATH.mkdir(parents=True, exist_ok=True)

# in‑memory structure ➜ {agent_name: [ { … } ]}
agent_memory = defaultdict(list)

def log_interaction(agent_name: str,
                    query: str,
                    response: str,
                    escalated_to: str | None = None,
                    confidence: float | None = None):
    record = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "query": query,
        "response": response,
        "escalated_to": escalated_to,
        "confidence": confidence,
    }
    agent_memory[agent_name].append(record)

    # append‑only JSONL on disk for long‑running sims
    with (_LOG_PATH / f"{agent_name}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
