# Monitoring Report — Phishing Email Detection

*Stage 6 (Utilization) artifact — output of the **Monitor System** activity, produced by `06_monitor.py`.*

- Generated: 2026-08-16T15:37:18+00:00
- Predictions analysed: 35
- Log file: `monitoring/predictions.log`

## What is and is not measured

No live accuracy, precision, recall or F1 appears in this report, and that is deliberate. Measuring them needs ground truth — whether each flagged email really was phishing — which production does not provide (verification latency). Following the Utilization guidance, monitoring here focuses on data and feature monitoring instead: the input distribution, the prediction distribution, the decision margin, and technical metrics.

## Indicators

| Indicator | Type | Value | Threshold | Status |
|---|---|---|---|---|
| Input length drift (Jensen-Shannon) | statistical | 0.869 | 0.2 | **BREACHED** |
| Predicted phishing rate change | statistical | 0.600 (training: 0.501, change: 0.099) | 0.25 | OK |
| Average decision margin | statistical | 0.5082 | >= 0.3 | OK |
| Request latency (95th percentile, ms) | computational | 6.23 | 500.0 | OK |

### Notes per indicator

- **Input length drift (Jensen-Shannon)** — Compares live input lengths against the training distribution.
- **Predicted phishing rate change** — A large swing can mean drift, an attack wave, or a broken input source.
- **Average decision margin** — Distance from the decision boundary, not a probability. A falling margin means the model is deciding closer to the line.
- **Request latency (95th percentile, ms)** — Technical metric — an infrastructure concern, handled by Perform Infrastructure Management.
## Outcome

**Maintenance needed.** At least one indicator defined in the Support Plan was breached. Per the Utilization BPMN this routes to **Perform Maintenance**, whose first sub-activity is root cause analysis — see the next section.

## Root Cause Analysis

The Utilization material separates **covariate shift** (the input distribution changed) from **concept shift** (the relationship between features and target changed). They call for different resolutions, so the pattern across indicators matters more than any single breach.

| Indicator group | Breached? |
|---|---|
| Input distribution (covariate) | yes |
| Output distribution (predictions) | no |
| Decision margin | no |

### Pattern: consistent with COVARIATE SHIFT

The input distribution moved away from the training distribution while the prediction distribution stayed within its threshold. This is the signature of covariate shift: the model is being asked about a different kind of input than it was trained on.

**Resolution implication.** The Utilization material lists retraining as only one of several resolution strategies, alongside dataset improvement (clean, enrich, enlarge), refactoring the use case, and changing downstream processes. Retraining on the *same* dataset cannot fix covariate shift — it reproduces the same model. The indicated resolution is **dataset improvement**: obtain training data that matches the format and length of the input the Serving Component actually receives.

Note that the decision-margin indicator did **not** breach. Margin alone would therefore have missed this problem; the input-distribution indicator is what caught it. This is why statistical monitoring covers both input and output data rather than relying on a single health number.

## Reference used

Training distribution from `feature_store/train_features.csv` (phishing rate 0.501). Note that the training reference measures parsed `clean_text` while the live log measures the raw text submitted to the API; a small baseline difference is expected and is not by itself evidence of drift.
