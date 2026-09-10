# CT Trigger Log — Phishing Email Detection

*Stage 6 (Utilization) artifact. Every continuous-training trigger is appended here by `ct_retrain.py`, whether it ran or was declined.*

| Timestamp (UTC) | Reason | Outcome | Detail |
|---|---|---|---|
| 2026-08-16T13:27:43+00:00 | Monitor flagged maintenance needed. Breached: Input length drift (Jensen-Shannon) | dry-run | Decision only, pipeline not executed |
| 2026-08-16T13:35:54+00:00 | Manual trigger: Stage 6 demonstration of CT trigger mechanism | completed | Pipeline completed |
| 2026-09-09T20:28:33+00:00 | Monitor flagged maintenance needed. Breached: Input length drift (Jensen-Shannon) | failed | 03_train_kfold.py failed with exit code 1 |
| 2026-09-09T20:59:56+00:00 | Monitor flagged maintenance needed. Breached: Input length drift (Jensen-Shannon) | dry-run | Decision only, pipeline not executed |
| 2026-09-09T21:13:10+00:00 | Monitor flagged maintenance needed. Breached: Input length drift (Jensen-Shannon) | failed | 03_train_kfold.py failed with exit code 1 |
| 2026-09-09T22:01:28+00:00 | Manual trigger: mlflow.db reset after schema corruption | completed | Pipeline completed |
