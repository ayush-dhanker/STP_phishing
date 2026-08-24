# STAGE 6 (Utilization) — PERFORM MAINTENANCE / CT COMPONENT

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime, timezone

STATUS_FILE = "monitoring/monitor_status.json"
LOG_FILE    = "CT_Trigger_Log.md"

PIPELINE = ["02_data_pipeline.py", "03_train.py", "04_evaluate.py"]



def read_monitor_status():
    if not os.path.exists(STATUS_FILE):
        return None
    with open(STATUS_FILE) as f:
        return json.load(f)


def decide(force_reason):
    if force_reason:
        return True, f"Manual trigger: {force_reason}"

    status = read_monitor_status()
    if status is None:
        return False, ("No monitor status found. Run 06_monitor.py first, "
                       "or use --force with a reason.")

    if status.get("maintenance_needed"):
        breached = ", ".join(status.get("breached_indicators", [])) or "unspecified"
        return True, f"Monitor flagged maintenance needed. Breached: {breached}"

    return False, "Monitor reports all indicators within thresholds."


def run_pipeline():
    for script in PIPELINE:
        if not os.path.exists(script):
            print(f"ERROR: {script} not found.")
            return False, f"{script} not found"

        print(f"\n{'=' * 60}")
        print(f"Running: {script}")
        print("=" * 60)
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\nERROR: {script} failed. Retraining stopped.")
            return False, f"{script} failed with exit code {result.returncode}"

    return True, "Pipeline completed"


def log_trigger(reason, outcome, detail):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if not os.path.exists(LOG_FILE):
        header = (
            "# CT Trigger Log — Phishing Email Detection\n\n"
            "*Stage 6 (Utilization) artifact. Every continuous-training "
            "trigger is appended here by `ct_retrain.py`, whether it ran or "
            "was declined.*\n\n"
            "| Timestamp (UTC) | Reason | Outcome | Detail |\n"
            "|---|---|---|---|\n"
        )
        with open(LOG_FILE, "w") as f:
            f.write(header)

    with open(LOG_FILE, "a") as f:
        f.write(f"| {now} | {reason} | {outcome} | {detail} |\n")

    print(f"  Logged -> {LOG_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="Continuous Training trigger (DSPM Perform Maintenance)."
    )
    parser.add_argument("--force", metavar="REASON",
                        help="retrain regardless of monitor status")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the decision without running the pipeline")
    args = parser.parse_args()

    print("STAGE 6 — PERFORM MAINTENANCE / CT COMPONENT")
    print("=" * 60)

    should_run, reason = decide(args.force)
    print(f"Decision: {'RETRAIN' if should_run else 'NO ACTION'}")
    print(f"Reason  : {reason}")

    if not should_run:
        log_trigger(reason, "declined", "No retraining performed")
        return

    if args.dry_run:
        print("\n[dry run] Would run: " + " -> ".join(PIPELINE))
        log_trigger(reason, "dry-run", "Decision only, pipeline not executed")
        return

    ok, detail = run_pipeline()

    if ok:
        print("\nRetraining complete.")
        print("04_evaluate.py registered a new model version and moved the")
        print("'production' alias only if the checkpoint criteria passed.")
        print("Restart 05_deploy.py so the Serving Component picks it up.")
        log_trigger(reason, "completed", detail)
    else:
        print("\nRetraining failed — production model is unchanged.")
        log_trigger(reason, "failed", detail)


if __name__ == "__main__":
    main()