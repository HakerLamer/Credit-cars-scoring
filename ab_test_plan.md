# A/B Test Plan: Credit Card Default Model v1 vs v2

## 1. Hypothesis

**H0:** Model v2 (Gradient Boosting) does not significantly improve the F1-score for default detection compared to Model v1 (Logistic Regression).  
**H1:** Model v2 produces a statistically significantly higher F1-score on the default class (class=1).

---

## 2. Models

| | Model v1 (Control) | Model v2 (Treatment) |
|---|---|---|
| Algorithm | Logistic Regression | Gradient Boosting |
| F1 (default class) | 0.4613 | 0.4708 |
| Precision | 0.3672 | 0.6639 |
| Recall | 0.6202 | 0.3647 |

**Key trade-off:**  
- v1 has higher Recall (catches more actual defaults) but lower Precision (more false positives).  
- v2 has higher Precision (fewer false alarms) but lower Recall.

The right choice depends on the bank's risk appetite:
- If missing a default is very costly → prefer v1 (higher recall)
- If false alarms are expensive (blocking good customers) → prefer v2 (higher precision)

---

## 3. Traffic Split

- **Method:** Deterministic hash of `client_id` modulo 100
- **Split:** 50% → v1 (control), 50% → v2 (treatment)
- **Rationale:** Hash-based routing ensures the same client always gets the same model, making results consistent and preventing data leakage between groups.

```python
import hashlib

def get_model_version(client_id):
    bucket = int(hashlib.md5(str(client_id).encode()).hexdigest(), 16) % 100
    return 'v2' if bucket < 50 else 'v1'
```

---

## 4. Duration

| Parameter | Value |
|---|---|
| Minimum sample size per group | ~1,500 clients (power=0.8, α=0.05) |
| Expected daily volume | ~500 requests |
| Minimum test duration | **14 days** |
| Recommended duration | **21 days** (capture weekly cycles) |

---

## 5. Primary Metrics

### 5.1 Technical Metrics

| Metric | Description | Why |
|---|---|---|
| **F1-score (default class)** | Harmonic mean of Precision and Recall | Primary — balances false positives and negatives |
| **Precision** | TP / (TP + FP) | Measures false alarm rate |
| **Recall** | TP / (TP + FN) | Measures how many actual defaults are caught |
| **AUC-ROC** | Area under ROC curve | Overall discriminative power |

### 5.2 Business Metrics

**Metric 1: Expected Financial Loss Reduction**

```
Expected Loss (per model) = FN_rate × avg_default_amount × volume
Improvement = Loss(v1) - Loss(v2)
```
- A false negative (missed default) costs the bank the full credit amount.
- Higher recall → fewer missed defaults → lower expected loss.

**Metric 2: False Alarm Rate (Customer Experience)**

```
False Alarm Rate = FP / (FP + TN)
```
- A false positive means blocking a good customer.
- Lower FP rate → better customer retention and satisfaction.

---

## 6. Statistical Test

### Method: Two-proportion Z-test on binary outcomes

Since the target is a binary classification (default / no default), we compare the proportion of correct predictions between groups.

```python
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

# After test period, collect:
# n_v1, n_v2 = sample sizes per group
# hits_v1, hits_v2 = correct default predictions per group

count = np.array([hits_v2, hits_v1])
nobs  = np.array([n_v2,    n_v1])

z_stat, p_value = proportions_ztest(count, nobs)
print(f"Z={z_stat:.3f}, p={p_value:.4f}")
```

### Confidence Intervals (95%)

```python
from statsmodels.stats.proportion import proportion_confint

ci_v1 = proportion_confint(hits_v1, n_v1, alpha=0.05, method='wilson')
ci_v2 = proportion_confint(hits_v2, n_v2, alpha=0.05, method='wilson')
print(f"v1 CI: {ci_v1}")
print(f"v2 CI: {ci_v2}")
```

### Success Criteria

| Criterion | Threshold |
|---|---|
| Statistical significance | p-value < 0.05 |
| Practical improvement (F1) | Δ F1 > 0.01 |
| No regression in business metrics | ΔLoss ≤ 0 |

---

## 7. Rollback Plan

- If v2 shows no significant improvement after 21 days → keep v1, archive v2.
- If v2 shows harm (significantly higher loss metric) → immediate rollback to v1 (change routing to 100% v1 in `api.py`).

---

## 8. Implementation in Service

Both models are loaded at startup and available via the `/predict` endpoint:

```bash
# Force model v1
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL": 50000, ..., "model_version": "v1"}'

# Force model v2
curl -X POST http://localhost:5000/predict \
  -d '{"LIMIT_BAL": 50000, ..., "model_version": "v2"}'

# A/B routing by client_id (deterministic)
curl -X POST http://localhost:5000/predict \
  -d '{"LIMIT_BAL": 50000, ..., "client_id": "user_12345"}'
```
