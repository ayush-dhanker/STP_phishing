# Evaluation Report — Phishing Email Detection

## Selected Model
- Pipeline: **tfidf_sublinear + linear_svc**
- Accuracy: 0.9973
- F1 (weighted): 0.9973
- AUC-ROC (from Stage 3): 0.9999876102686094

## Success Criteria (from Stage 1)
- Accuracy >= 0.95: PASS
- F1 >= 0.94: PASS
- AUC >= 0.95: PASS

## Checkpoint Decision
**PROCEED TO DEPLOYMENT**

## Classification Report
```
              precision    recall  f1-score   support

        safe     1.0000    0.9970    0.9985      3363
    phishing     0.9711    1.0000    0.9853       336

    accuracy                         0.9973      3699
   macro avg     0.9855    0.9985    0.9919      3699
weighted avg     0.9974    0.9973    0.9973      3699

```

## Top 5 candidates considered
```
     vectorizer         model  accuracy  f1_weighted  roc_auc
tfidf_sublinear    linear_svc  0.997297     0.997314 0.999988
        hashing    linear_svc  0.997297     0.997314 0.999981
  tfidf_unigram    linear_svc  0.997297     0.997311 0.999979
    bow_unigram random_forest  0.996756     0.996781 0.999825
  tfidf_unigram random_forest  0.996756     0.996781 0.999880
```