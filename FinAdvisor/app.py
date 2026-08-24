"""Flask application for the FinAdvisor investment risk platform."""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from model.predict import ModelPredictor
from model.train_model import ensure_artifacts
from preprocessing.preprocess import find_dataset, profile_dataset

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["SECRET_KEY"] = "finadvisor-local-development"


def get_context() -> dict:
    """Load training artifacts and present a template-friendly dashboard context."""
    artifacts = ensure_artifacts(BASE_DIR)
    dataset = find_dataset(BASE_DIR / "data")
    profile = profile_dataset(dataset)
    return {"artifacts": artifacts, "profile": profile, "dataset": dataset.name}


@app.route("/")
def index():
    return render_template("index.html", **get_context())


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", **get_context())


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    predictor = ModelPredictor(BASE_DIR)
    result = predictor.predict(payload)
    return jsonify(result)


@app.route("/api/summary")
def summary():
    context = get_context()
    return jsonify({"profile": context["profile"], "metrics": context["artifacts"]["metrics"]})


if __name__ == "__main__":
    ensure_artifacts(BASE_DIR)
    app.run(debug=True, host="127.0.0.1", port=5000)
