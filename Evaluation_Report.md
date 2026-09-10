# Evaluation Report — Phishing Email Detection

## Selected Model
- Pipeline: **tfidf_sublinear + linear_svc**
- Selected by: 5-fold CV validation F1 = 0.9964 (+/- 0.0006)
- Overfitting risk flag: False (CV train-val gap: 0.0034, train-test gap: 0.0027)

## Independent holdout test performance
- Accuracy: 0.9970
- F1 (weighted): 0.9970
- AUC-ROC (from Stage 3): 0.999988495249423

## Success Criteria (from Stage 1)
- Accuracy >= 0.95: PASS
- F1 >= 0.94: PASS
- AUC >= 0.95: PASS

## Checkpoint Decision
**PROCEED TO DEPLOYMENT**

## Classification Report
```
              precision    recall  f1-score   support

        safe     1.0000    0.9967    0.9984      3363
    phishing     0.9683    1.0000    0.9839       336

    accuracy                         0.9970      3699
   macro avg     0.9841    0.9984    0.9911      3699
weighted avg     0.9971    0.9970    0.9970      3699

```

## Top 5 candidates considered (by CV validation F1)
```
     vectorizer               model  cv_val_f1_mean  test_accuracy  test_f1  test_roc_auc  overfitting_risk
tfidf_sublinear          linear_svc        0.996358       0.997026 0.997048      0.999988             False
   tfidf_bigram          linear_svc        0.995986       0.996486 0.996511      0.999965             False
  tfidf_unigram          linear_svc        0.995912       0.997297 0.997311      0.999977             False
        hashing          linear_svc        0.995763       0.997297 0.997314      0.999981             False
     bow_bigram logistic_regression        0.993942       0.992430 0.992557      0.999917             False
```