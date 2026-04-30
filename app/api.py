"""Credit Card Default Prediction API (Flask)."""
import os
import json
import logging
import hashlib
from datetime import datetime
from flask import Flask, request, jsonify
from app.model_handler import ModelHandler

# JSON structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
handler = ModelHandler(
    v1_path=os.path.join(BASE, 'models/model_v1.pkl'),
    v2_path=os.path.join(BASE, 'models/model_v2.pkl'),
    meta_path=os.path.join(BASE, 'models/model_meta.json')
)


def get_model_version(client_id=None):
    """A/B routing: 50/50 split by client_id hash, or random."""
    if client_id:
        bucket = int(hashlib.md5(str(client_id).encode()).hexdigest(), 16) % 100
        return 'v2' if bucket < 50 else 'v1'
    return 'v1'


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body: JSON with client features + optional 'client_id' and 'model_version'.
    Returns: prediction (0/1), probability, model_version.
    """
    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({'error': 'Invalid JSON body'}), 400

        # A/B routing
        client_id = data.pop('client_id', None)
        model_version = data.pop('model_version', None) or get_model_version(client_id)

        prediction, probability = handler.predict(data, model_version)

        response = {
            'prediction': int(prediction),
            'probability': round(float(probability), 4),
            'model_version': model_version,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        logger.info(json.dumps({
            'endpoint': '/predict',
            'client_id': client_id,
            'model_version': model_version,
            'prediction': int(prediction),
            'probability': round(float(probability), 4)
        }))

        return jsonify(response), 200

    except KeyError as e:
        return jsonify({'error': f'Missing feature: {e}'}), 400
    except Exception as e:
        logger.error(str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """GET /health — service health check."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': handler.models_loaded(),
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 200


@app.route('/models', methods=['GET'])
def models_info():
    """GET /models — info about available model versions and metrics."""
    return jsonify(handler.get_meta()), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
