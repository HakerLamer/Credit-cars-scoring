# Architecture: Credit Card Default Prediction Service

## Monolith vs Microservices

### Decision: Monolithic Architecture

For this project, a **monolithic** approach was chosen. The Flask application contains:
- Model loading and inference
- API routing (`/predict`, `/health`, `/models`)
- A/B routing logic
- JSON structured logging

### Rationale

| Aspect | Monolith (chosen) | Microservices |
|---|---|---|
| Deployment complexity | Low — single container | High — multiple services, service discovery |
| Team size | Solo / small team | Multiple teams |
| Latency | No network hops | Inter-service calls add latency |
| Scaling | Scale the whole app | Scale individual services independently |
| Debugging | Simple — single log stream | Complex — distributed tracing needed |
| A/B routing | In-process, low overhead | Requires API gateway or sidecar |

**Conclusion:** For a single ML model service with moderate traffic, a monolith provides faster iteration, simpler operations, and sufficient performance. Microservices would be appropriate if we needed to independently scale preprocessing, inference, and logging, or if multiple teams owned different parts.

---

## Message Brokers (Concept: RabbitMQ)

In a scaled production system, a message broker like **RabbitMQ** or **Apache Kafka** would be introduced for:

1. **Async batch predictions:** Instead of synchronous HTTP calls, clients publish feature vectors to a queue. Workers consume and return predictions — decoupling throughput from latency.
2. **Audit logging:** Every prediction event is published to a queue consumed by a logging service (e.g., writing to Elasticsearch), without blocking the main API.
3. **Model retraining triggers:** When accumulated predictions exceed a threshold, a message triggers an automated retraining pipeline.

```
Client → POST /predict → Flask API → RabbitMQ (publish event)
                                           ↓
                                     Worker: log to ELK
                                     Worker: batch inference
                                     Worker: retraining trigger
```

---

## Logging & Monitoring

All requests are logged in **structured JSON format**:

```json
{
  "time": "2025-01-15T12:00:00",
  "level": "INFO",
  "endpoint": "/predict",
  "client_id": "user_123",
  "model_version": "v1",
  "prediction": 1,
  "probability": 0.7823
}
```

In production, these logs would be collected by:
- **Filebeat** → **Elasticsearch** → **Kibana** (ELK stack)
- Dashboards: prediction volume, default rate by model version, latency percentiles, error rate

---

## MLOps Concepts

### DVC (Data Version Control)
DVC tracks the `UCI_Credit_Card.csv` file and model artifacts (`model_v1.pkl`, `model_v2.pkl`) in Git-compatible versioning. This ensures every experiment is reproducible: given a commit hash, you can reproduce the exact data, code, and trained model.

```bash
dvc add data/UCI_Credit_Card.csv
dvc add models/model_v1.pkl
git add data/UCI_Credit_Card.csv.dvc .gitignore
git commit -m "Track dataset and model with DVC"
```

### MLflow (Experiment Tracking)
MLflow logs hyperparameters, metrics, and artifacts for each training run. This allows comparison across experiments (e.g., different `n_estimators` for GBM) and promotion of the best model to production via the MLflow Model Registry.

```python
import mlflow
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("f1", 0.4708)
    mlflow.sklearn.log_model(pipeline_v2, "model_v2")
```

---

## uWSGI + NGINX in Production

The `python app.api` development server is single-threaded and not suitable for production. In production:

- **uWSGI** runs the Flask app with multiple worker processes (e.g., 4 workers × 2 threads), handling concurrent requests efficiently.
- **NGINX** sits in front as a reverse proxy: handles SSL termination, request buffering, static file serving, and rate limiting.

```
Internet → NGINX (443/80) → uWSGI (socket) → Flask App (workers)
```
