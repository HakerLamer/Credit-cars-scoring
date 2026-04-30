"""Model loading and inference handler."""
import os
import json
import numpy as np
import joblib


class ModelHandler:
    def __init__(self, v1_path: str, v2_path: str, meta_path: str):
        self.models = {}
        self.meta = {}
        self.features = []

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)
            self.features = self.meta.get('features', [])

        if os.path.exists(v1_path):
            self.models['v1'] = joblib.load(v1_path)
        if os.path.exists(v2_path):
            self.models['v2'] = joblib.load(v2_path)

    def predict(self, data: dict, version: str = 'v1'):
        """Run inference. Returns (prediction, probability)."""
        if version not in self.models:
            raise ValueError(f"Model version '{version}' not available. Choose from: {list(self.models.keys())}")

        model = self.models[version]

        # Build feature DataFrame in correct column order
        import pandas as pd
        features_arr = pd.DataFrame(
            [[float(data[f]) for f in self.features]],
            columns=self.features
        )

        prediction = model.predict(features_arr)[0]
        probability = model.predict_proba(features_arr)[0][1]
        return prediction, probability

    def models_loaded(self) -> dict:
        return {v: True for v in self.models}

    def get_meta(self) -> dict:
        return self.meta
