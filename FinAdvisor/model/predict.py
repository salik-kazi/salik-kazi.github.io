"""Prediction and human-readable recommendation logic."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
from model.train_model import ensure_artifacts

class ModelPredictor:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir; self.meta = ensure_artifacts(base_dir); self.model = joblib.load(base_dir / "model" / "saved_model.pkl")
    def predict(self, values: dict) -> dict:
        record = {}
        for col in self.meta["features"]:
            raw = values.get(col, self.meta["defaults"][col])
            record[col] = float(raw) if col in self.meta["numeric_features"] else str(raw)
        probabilities = self.model.predict_proba(pd.DataFrame([record]))[0]; label = str(self.model.classes_[probabilities.argmax()]); confidence = round(float(probabilities.max()) * 100, 1)
        label_lower = label.lower()
        if any(x in label_lower for x in ("1", "low", "safe")): recommendation = "A conservative profile: prioritise stable, diversified funds and review costs."
        elif any(x in label_lower for x in ("5", "high", "aggressive")): recommendation = "An elevated-risk profile: diversify across categories and align exposure to your long-term horizon."
        else: recommendation = "A balanced profile: combine growth opportunities with resilient, diversified holdings."
        return {"risk_level": label, "confidence": confidence, "recommendation": recommendation, "probabilities": {str(c): round(float(p) * 100, 1) for c, p in zip(self.model.classes_, probabilities)}}
