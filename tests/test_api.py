"""Tests for the prediction API."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
from app.api import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


SAMPLE_PAYLOAD = {
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 10000, "BILL_AMT2": 9000, "BILL_AMT3": 8000,
    "BILL_AMT4": 7000, "BILL_AMT5": 6000, "BILL_AMT6": 5000,
    "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
    "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000
}


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['status'] == 'healthy'
    assert 'v1' in data['models_loaded']
    assert 'v2' in data['models_loaded']


def test_predict_v1(client):
    payload = {**SAMPLE_PAYLOAD, "model_version": "v1"}
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['model_version'] == 'v1'
    assert data['prediction'] in [0, 1]
    assert 0.0 <= data['probability'] <= 1.0


def test_predict_v2(client):
    payload = {**SAMPLE_PAYLOAD, "model_version": "v2"}
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['model_version'] == 'v2'
    assert data['prediction'] in [0, 1]


def test_predict_ab_routing(client):
    """A/B routing by client_id should be deterministic."""
    payload = {**SAMPLE_PAYLOAD, "client_id": "user123"}
    r1 = client.post('/predict', json=payload)
    r2 = client.post('/predict', json={**SAMPLE_PAYLOAD, "client_id": "user123"})
    d1, d2 = json.loads(r1.data), json.loads(r2.data)
    assert d1['model_version'] == d2['model_version']


def test_predict_missing_feature(client):
    r = client.post('/predict', json={"LIMIT_BAL": 50000})
    assert r.status_code in [400, 500]


def test_models_endpoint(client):
    r = client.get('/models')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'features' in data
