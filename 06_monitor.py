# STAGE 6 (Utilization) — MONITOR SYSTEM

import os
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from dotenv import load_dotenv

load_dotenv()

LOG_FILE     = os.getenv("MONITOR_LOG_FILE", "monitoring/predictions.log")
TRAIN_CSV    = "feature_store/train_features.csv"
STATUS_FILE  = "monitoring/monitor_status.json"
REPORT_FILE  = "Monitoring_Report.md"

# minimum number of predictions
MIN_PREDICTIONS = int(os.getenv("MONITOR_MIN_PREDICTIONS", "30"))

# Jensen-Shannon distance above this means input distribution has shifted
DRIFT_THRESHOLD = float(os.getenv("MONITOR_DRIFT_THRESHOLD", "0.20"))

# predicted phishing rate 
PHISHING_RATE_DELTA = float(os.getenv("MONITOR_PHISHING_RATE_DELTA", "0.25"))

# average decision score below this means model sits close to the boundary a lot
LOW_MARGIN_THRESHOLD = float(os.getenv("MONITOR_LOW_MARGIN", "0.30"))

# request latency above this (milliseconds) is a infrastructure concern
LATENCY_THRESHOLD_MS = float(os.getenv("MONITOR_LATENCY_MS", "500"))

os.makedirs("monitoring", exist_ok=True)


# Loading the live prediction
def load_predictions(path):
    """Read the prediction log. One JSON object per line."""
    if not os.path.exists(path):
        return pd.DataFrame()

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # a half-written line (service killed mid-write) — skip it
                continue
    return pd.DataFrame(rows)

def load_reference(path):
    df = pd.read_csv(path)
    lengths = df["clean_text"].fillna("").str.len()
    phishing_rate = float(df["label"].mean())
    return lengths, phishing_rate



# drift measure (Jensen-Shannon distance between two histograms)
def to_histogram(values, bin_edges):
    """Turn a list of numbers into proportions per bin (they sum to 1)."""
    counts, _ = np.histogram(values, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return np.zeros(len(counts))
    return counts / total


def js_distance(reference_values, live_values, n_bins=10):
    lo = float(np.percentile(reference_values, 1))
    hi = float(np.percentile(reference_values, 99))
    if hi <= lo:
        hi = lo + 1
    bin_edges = np.linspace(lo, hi, n_bins + 1)

    ref_hist = to_histogram(reference_values, bin_edges)
    live_hist = to_histogram(live_values, bin_edges)

    epsilon = 1e-10
    ref_hist = ref_hist + epsilon
    live_hist = live_hist + epsilon

    distance = jensenshannon(ref_hist, live_hist, base=2)
    return float(distance) if not np.isnan(distance) else 0.0



def run_checks(live, ref_lengths, ref_phishing_rate):
    """Return a list of check results and whether maintenance is needed."""
    checks = []

    drift = js_distance(ref_lengths, live["text_length"])
    checks.append({
        "name": "Input length drift (Jensen-Shannon)",
        "type": "statistical",
        "value": round(drift, 4),
        "threshold": DRIFT_THRESHOLD,
        "breached": drift > DRIFT_THRESHOLD,
        "note": "Compares live input lengths against the training distribution.",
    })

    live_phishing_rate = float(live["label"].mean())
    rate_change = abs(live_phishing_rate - ref_phishing_rate)
    checks.append({
        "name": "Predicted phishing rate change",
        "type": "statistical",
        "value": f"{live_phishing_rate:.3f} (training: {ref_phishing_rate:.3f}, change: {rate_change:.3f})",
        "threshold": PHISHING_RATE_DELTA,
        "breached": rate_change > PHISHING_RATE_DELTA,
        "note": "A large swing can mean drift, an attack wave, or a broken input source.",
    })

    if "decision_score" in live.columns and live["decision_score"].notna().any():
        avg_margin = float(live["decision_score"].dropna().mean())
        checks.append({
            "name": "Average decision margin",
            "type": "statistical",
            "value": round(avg_margin, 4),
            "threshold": f">= {LOW_MARGIN_THRESHOLD}",
            "breached": avg_margin < LOW_MARGIN_THRESHOLD,
            "note": "Distance from the decision boundary, not a probability. "
                    "A falling margin means the model is deciding closer to the line.",
        })

    #computational monitoring- latency 
    if "latency_ms" in live.columns and live["latency_ms"].notna().any():
        latencies = live["latency_ms"].dropna()
        p95 = float(np.percentile(latencies, 95))
        checks.append({
            "name": "Request latency (95th percentile, ms)",
            "type": "computational",
            "value": round(p95, 2),
            "threshold": LATENCY_THRESHOLD_MS,
            "breached": p95 > LATENCY_THRESHOLD_MS,
            "note": "Technical metric — an infrastructure concern, "
                    "handled by Perform Infrastructure Management.",
        })
    else:
        checks.append({
            "name": "Request latency",
            "type": "computational",
            "value": "not logged",
            "threshold": LATENCY_THRESHOLD_MS,
            "breached": False,
            "note": "No latency_ms field in the log — apply the 05_deploy.py patch.",
        })

    maintenance_needed = any(c["breached"] for c in checks)
    return checks, maintenance_needed


# DSPM "Perform Maintenance" 
def root_cause_section(checks, maintenance_needed):
    def find(keyword):
        for c in checks:
            if keyword in c["name"]:
                return c
        return None

    input_drift = find("Input length drift")
    output_shift = find("phishing rate")
    margin = find("decision margin")

    input_breached = bool(input_drift and input_drift["breached"])
    output_breached = bool(output_shift and output_shift["breached"])
    margin_breached = bool(margin and margin["breached"])

    lines = ["## Root Cause Analysis\n"]

    if not maintenance_needed:
        lines.append(
            "No indicator was breached, so no root cause analysis is "
            "required in this cycle.\n")
        return lines

    lines.append(
        "The Utilization material separates **covariate shift** (the input "
        "distribution changed) from **concept shift** (the relationship "
        "between features and target changed). They call for different "
        "resolutions, so the pattern across indicators matters more than "
        "any single breach.\n")

    lines.append("| Indicator group | Breached? |")
    lines.append("|---|---|")
    lines.append(f"| Input distribution (covariate) | {'yes' if input_breached else 'no'} |")
    lines.append(f"| Output distribution (predictions) | {'yes' if output_breached else 'no'} |")
    lines.append(f"| Decision margin | {'yes' if margin_breached else 'no'} |\n")

    if input_breached and not output_breached:
        lines.append("### Pattern: consistent with COVARIATE SHIFT\n")
        lines.append(
            "The input distribution moved away from the training "
            "distribution while the prediction distribution stayed within "
            "its threshold. This is the signature of covariate shift: the "
            "model is being asked about a different kind of input than it "
            "was trained on.\n")
        lines.append("**Resolution implication.** The Utilization material "
                     "lists retraining as only one of several resolution "
                     "strategies, alongside dataset improvement (clean, "
                     "enrich, enlarge), refactoring the use case, and "
                     "changing downstream processes. Retraining on the "
                     "*same* dataset cannot fix covariate shift — it "
                     "reproduces the same model. The indicated resolution "
                     "is **dataset improvement**: obtain training data that "
                     "matches the format and length of the input the "
                     "Serving Component actually receives.\n")
    elif output_breached and not input_breached:
        lines.append("### Pattern: possible CONCEPT SHIFT\n")
        lines.append(
            "The predicted class balance moved while the input "
            "distribution stayed stable. This can indicate that the "
            "relationship between wording and label has changed, or an "
            "attack wave, or a broken upstream input source. Concept shift "
            "is hard to confirm without ground truth; labelled samples "
            "should be collected before acting.\n")
        lines.append("**Resolution implication.** If confirmed, retraining "
                     "on fresh, newly labelled data is the appropriate "
                     "strategy.\n")
    elif input_breached and output_breached:
        lines.append("### Pattern: input AND output both shifted\n")
        lines.append(
            "Both distributions moved. Covariate shift and concept shift "
            "can occur together, and the material notes that the effect of "
            "covariate shift on concept drift may be none, negative or "
            "positive. Labelled samples are needed to separate the two "
            "before choosing a resolution.\n")
    else:
        lines.append("### Pattern: technical indicator only\n")
        lines.append(
            "No statistical shift was detected in the input or output "
            "distributions. The breach concerns a technical metric, which "
            "routes to **Perform Infrastructure Management** rather than to "
            "model retraining.\n")

    if margin_breached:
        lines.append(
            "The average decision margin also fell below its threshold, "
            "meaning the model is deciding closer to its boundary than "
            "usual — consistent with input it is less equipped to judge.\n")
    elif input_breached:
        lines.append(
            "Note that the decision-margin indicator did **not** breach. "
            "Margin alone would therefore have missed this problem; the "
            "input-distribution indicator is what caught it. This is why "
            "statistical monitoring covers both input and output data "
            "rather than relying on a single health number.\n")

    return lines


# reprt
def write_report(live, checks, maintenance_needed, ref_phishing_rate):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines = [
        "# Monitoring Report — Phishing Email Detection\n",
        "*Stage 6 (Utilization) artifact — output of the **Monitor System** "
        "activity, produced by `06_monitor.py`.*\n",
        f"- Generated: {now}",
        f"- Predictions analysed: {len(live):,}",
        f"- Log file: `{LOG_FILE}`\n",
        "## What is and is not measured\n",
        "No live accuracy, precision, recall or F1 appears in this report, "
        "and that is deliberate. Measuring them needs ground truth — whether "
        "each flagged email really was phishing — which production does not "
        "provide (verification latency). Following the Utilization guidance, "
        "monitoring here focuses on data and feature monitoring instead: the "
        "input distribution, the prediction distribution, the decision "
        "margin, and technical metrics.\n",
        "## Indicators\n",
        "| Indicator | Type | Value | Threshold | Status |",
        "|---|---|---|---|---|",
    ]
    for c in checks:
        status = "**BREACHED**" if c["breached"] else "OK"
        lines.append(f"| {c['name']} | {c['type']} | {c['value']} | {c['threshold']} | {status} |")

    lines.append("\n### Notes per indicator\n")
    for c in checks:
        lines.append(f"- **{c['name']}** — {c['note']}")

    lines.append("## Outcome\n")
    if maintenance_needed:
        lines.append(
            "**Maintenance needed.** At least one indicator defined in the "
            "Support Plan was breached. Per the Utilization BPMN this routes "
            "to **Perform Maintenance**, whose first sub-activity is root "
            "cause analysis — see the next section.\n"
        )
    else:
        lines.append(
            "**No maintenance needed.** All indicators are within the "
            "thresholds defined in the Support Plan. The system continues in "
            "**Create Value** — serving domain users — with monitoring "
            "running continuously.\n"
        )

    lines.extend(root_cause_section(checks, maintenance_needed))

    lines.append("## Reference used\n")
    lines.append(
        f"Training distribution from `{TRAIN_CSV}` "
        f"(phishing rate {ref_phishing_rate:.3f}). Note that the training "
        "reference measures parsed `clean_text` while the live log measures "
        "the raw text submitted to the API; a small baseline difference is "
        "expected and is not by itself evidence of drift.\n"
    )

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved -> {REPORT_FILE}")


def main():
    print("STAGE 6 — MONITOR SYSTEM")
    print("=" * 60)

    live = load_predictions(LOG_FILE)
    if live.empty:
        print(f"No predictions found in {LOG_FILE}.")
        print("Send some requests to POST /predict first, then re-run.")
        return

    print(f"Loaded {len(live):,} predictions from {LOG_FILE}")

    if len(live) < MIN_PREDICTIONS:
        print(f"Only {len(live)} predictions — fewer than the minimum "
              f"of {MIN_PREDICTIONS} needed for a reliable reading.")
        print("Report will still be written, but treat the numbers as indicative.")

    if not os.path.exists(TRAIN_CSV):
        print(f"ERROR: {TRAIN_CSV} missing. Run 02_data_pipeline.py first.")
        return

    ref_lengths, ref_phishing_rate = load_reference(TRAIN_CSV)
    checks, maintenance_needed = run_checks(live, ref_lengths, ref_phishing_rate)

    print("\nIndicators:")
    for c in checks:
        status = "BREACHED" if c["breached"] else "OK"
        print(f"  [{status:8}] {c['name']}: {c['value']}")

    print(f"\nMaintenance needed: {maintenance_needed}")

    write_report(live, checks, maintenance_needed, ref_phishing_rate)

    # status file — ct_retrain.py reads this
    status = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_predictions": int(len(live)),
        "maintenance_needed": bool(maintenance_needed),
        "breached_indicators": [c["name"] for c in checks if c["breached"]],
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)
    print(f"  Saved -> {STATUS_FILE}")

    if maintenance_needed:
        print("\nNext: Perform Maintenance -> run ct_retrain.py if retraining "
              "is the chosen resolution.")
    else:
        print("\nSystem healthy. Continue Create Value.")


if __name__ == "__main__":
    main()