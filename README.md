# Credit Card Default Prediction Service

ML service for predicting credit card default, built with Flask, scikit-learn, and Docker. Supports **A/B testing** of two model versions with deterministic traffic routing.

---

## Project Structure

```
credit-card-ml/
├── app/
│   ├── __init__.py
│   ├── api.py              # Flask application (endpoints)
│   └── model_handler.py    # Model loading & inference
├── models/
│   ├── train_model.py      # Training script (v1 + v2)
│   ├── model_v1.pkl        # Logistic Regression (saved)
│   ├── model_v2.pkl        # Gradient Boosting (saved)
│   └── model_meta.json     # Feature list & metrics
├── tests/
│   └── test_api.py         # API unit tests (pytest)
├── data/
│   └── UCI_Credit_Card.csv # Dataset
├── docker/
│   └── nginx.conf
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── ab_test_plan.md         # Full A/B test plan
├── ARCHITECTURE.md         # Architecture decisions
└── README.md
```

---

## Quick Start (Local)

```bash
# 1. Clone and install
git clone https://github.com/HakerLamer/credit-card-ml
cd credit-card-ml
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Train models
python models/train_model.py

# 3. Run service
python -m app.api
# Service available at http://localhost:8000
```

---

## API Endpoints

### `GET /health`

Returns service health status.

```bash
curl http://localhost:8000/health
```

```json
{
  "models_loaded":
  {
    "v1":true,
    "v2":true
  },
  "status":"healthy",
  "timestamp":"2026-05-01T15:58:55.293477Z"
}
```

---

### `POST /predict`

Returns default prediction for a client.

**Request body:** JSON with 23 client features (see dataset description) + optional control fields.

Optional control fields:

- `model_version`: `"v1"` or `"v2"` — force a specific model
- `client_id`: string — enables deterministic A/B routing (same client always gets same model)

**Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{\"LIMIT_BAL\":50000,\"SEX\":2,\"EDUCATION\":2,\"MARRIAGE\":1,\"AGE\":35,\"PAY_0\":0,\"PAY_2\":0,\"PAY_3\":0,\"PAY_4\":0,\"PAY_5\":0,\"PAY_6\":0,\"BILL_AMT1\":10000,\"BILL_AMT2\":9000,\"BILL_AMT3\":8000,\"BILL_AMT4\":7000,\"BILL_AMT5\":6000,\"BILL_AMT6\":5000,\"PAY_AMT1\":2000,\"PAY_AMT2\":2000,\"PAY_AMT3\":2000,\"PAY_AMT4\":2000,\"PAY_AMT5\":2000,\"PAY_AMT6\":2000,\"model_version\":\"v1\"}'
```

**Response:**

```json
{
  "model_version":"v1",
  "prediction":1,
  "probability":0.5324,
  "timestamp":"2026-05-01T15:54:56.156351Z"
}
```

- `prediction`: `1` = default expected, `0` = no default expected
- `probability`: probability of default (class 1)

**A/B routing example (by client_id):**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"client_id": "user_12345", "LIMIT_BAL": 80000, ...}'
```

---

### `GET /models`

Returns model metadata and performance metrics.

```bash
curl http://localhost:8000/models
```

---

## Docker

### Build and Run

```bash
docker build -t credit-card-ml .
docker run -p 8000:8000 credit-card-ml
```

### Docker Compose (with NGINX)

```bash
docker-compose up --build
```

### Docker Hub

```
docker pull tomhetfrainsiden/credit-card-ml:latest
docker run -p 8000:8000 tomhetfrainsiden/credit-card-ml:latest
```

---

## Model Performance

| Metric | v1 (Logistic Regression) | v2 (Gradient Boosting) |
|---|---|---|
| F1 (default) | 0.4613 | 0.4708 |
| Precision | 0.3672 | 0.6639 |
| Recall | 0.6202 | 0.3647 |

**Trade-off:** v1 catches more actual defaults (higher recall). v2 has fewer false alarms (higher precision). See [ab_test_plan.md](ab_test_plan.md) for the full A/B test framework to determine which is better for the bank's risk profile.

---

## Run Tests

```bash
python -m pytest tests/ -v
```

---

## Architecture & MLOps

See [ARCHITECTURE.md](ARCHITECTURE.md) for:

- Monolith vs microservices decision
- RabbitMQ message broker concept
- uWSGI + NGINX explanation
- DVC and MLflow overview
- Logging strategy (structured JSON → ELK)

---

## Dataset

[Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) — UCI ML Repository. 30,000 Taiwanese credit card clients, target: `default.payment.next.month`.
