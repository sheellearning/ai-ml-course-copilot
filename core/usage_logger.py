import json
import os
from collections import Counter
from datetime import datetime


LOG_FILE = "data/metrics/usage_logs.jsonl"


# Create the metrics folder before saving logs.
def ensure_log_folder():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


# Append one usage record to the JSONL log file.
def write_log(question, status, score=None):
    ensure_log_folder()

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "status": status,
        "score": score,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(log_entry) + "\n")


# Record a successful document-grounded answer.
def log_success(question, score=None):
    write_log(question, "success", score)


# Record a failed or refused request.
def log_failure(question, score=None):
    write_log(question, "failure", score)


# Read all saved usage records.
def read_logs():
    if not os.path.exists(LOG_FILE):
        return []

    logs = []

    with open(LOG_FILE, "r", encoding="utf-8") as file:
        for line in file:
            logs.append(json.loads(line))

    return logs


# Calculate usage metrics for the admin dashboard.
def get_metrics():
    logs = read_logs()

    total_requests = len(logs)
    successful_answers = sum(1 for log in logs if log["status"] == "success")
    failed_answers = sum(1 for log in logs if log["status"] == "failure")

    success_rate = (
        round(successful_answers / total_requests * 100, 2)
        if total_requests > 0
        else 0
    )

    most_asked = Counter(log["question"] for log in logs).most_common(5)

    return {
        "total_requests": total_requests,
        "success_rate": success_rate,
        "failed_answers": failed_answers,
        "most_asked": most_asked,
        "logs": logs,
    }