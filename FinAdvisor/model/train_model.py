"""Adaptive model training and visualization generation."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from preprocessing.preprocess import clean_dataset, create_risk_target, detect_target, find_dataset, load_dataset, profile_dataset

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid", palette="Blues")


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value.lower()).strip("_")


def _save_plot(fig, destination: Path) -> None:
    fig.tight_layout(); fig.savefig(destination, dpi=160, bbox_inches="tight", facecolor="#f5f5f7"); plt.close(fig)


def generate_visualizations(frame: pd.DataFrame, target: str, output: Path, feature_names: list[str] | None = None, importances=None) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    plots = []
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    if len(numeric) >= 2:
        fig, ax = plt.subplots(figsize=(10, 7)); sns.heatmap(frame[numeric].corr(), cmap="Blues", center=0, ax=ax); ax.set_title("Feature correlation"); _save_plot(fig, output / "correlation_heatmap.png"); plots.append("correlation_heatmap.png")
    if numeric:
        fig, ax = plt.subplots(figsize=(9, 5)); sns.histplot(frame[numeric[0]], kde=True, ax=ax, color="#5b8def"); ax.set_title(f"Distribution of {numeric[0]}"); _save_plot(fig, output / "distribution.png"); plots.append("distribution.png")
        fig, ax = plt.subplots(figsize=(9, 5)); sns.boxplot(data=frame[numeric[:min(6, len(numeric))]], ax=ax, color="#9bb9f4"); ax.set_title("Outlier review"); ax.tick_params(axis="x", rotation=30); _save_plot(fig, output / "boxplot.png"); plots.append("boxplot.png")
    if target in frame:
        fig, ax = plt.subplots(figsize=(7, 5)); frame[target].astype(str).value_counts().plot(kind="bar", ax=ax, color="#5b8def"); ax.set_title("Risk distribution"); ax.set_xlabel("Risk level"); _save_plot(fig, output / "risk_distribution.png"); plots.append("risk_distribution.png")
    category = next((c for c in frame.columns if "category" in c.lower() and c != target), None)
    if category:
        fig, ax = plt.subplots(figsize=(9, 5)); frame[category].value_counts().head(10).sort_values().plot(kind="barh", ax=ax, color="#789eea"); ax.set_title("Investment category distribution"); _save_plot(fig, output / "category_distribution.png"); plots.append("category_distribution.png")
    if importances is not None and feature_names:
        top = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(9, 5)); top.sort_values().plot(kind="barh", ax=ax, color="#5b8def"); ax.set_title("Feature importance"); _save_plot(fig, output / "feature_importance.png"); plots.append("feature_importance.png")
    return plots


def train(base_dir: Path) -> dict:
    data_path = find_dataset(base_dir / "data")
    frame = clean_dataset(load_dataset(data_path))
    target = detect_target(frame)
    inferred = target is None
    if inferred:
        frame, target = create_risk_target(frame)
    # Identifier-like fields are meaningful for reporting but harm generalization.
    features = [c for c in frame.columns if c != target and frame[c].nunique() < len(frame) * .92]
    X, y = frame[features], frame[target].astype(str)
    numeric = X.select_dtypes(include=np.number).columns.tolist(); categorical = [c for c in features if c not in numeric]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.22, random_state=42, stratify=y if y.nunique() > 1 else None)
    # Do not pass ``class_weight='balanced'`` here. Some recent scikit-learn
    # builds incorrectly reconcile stringified labels with the encoded class
    # values when that option is used, raising a ValueError during fit.
    candidates = {"Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42), "Gradient Boosting": GradientBoostingClassifier(random_state=42)}
    trials = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train); trials[name] = (pipe, accuracy_score(y_test, pipe.predict(X_test)))
    name, (best, score) = max(trials.items(), key=lambda item: item[1][1])
    predictions = best.predict(X_test)
    smallest_class = int(y.value_counts().min())
    cv = cross_val_score(best, X, y, cv=min(5, smallest_class), scoring="f1_weighted").mean() if smallest_class >= 2 else score
    model_path = base_dir / "model" / "saved_model.pkl"; joblib.dump(best, model_path)
    transformed = best.named_steps["preprocessor"].get_feature_names_out()
    importance = getattr(best.named_steps["model"], "feature_importances_", None)
    plots = generate_visualizations(frame, target, base_dir / "static" / "images", list(transformed), importance)
    metrics = {"model": name, "accuracy": round(float(score), 3), "f1": round(float(f1_score(y_test, predictions, average="weighted")), 3), "cross_validation": round(float(cv), 3), "report": classification_report(y_test, predictions, output_dict=True), "confusion_matrix": confusion_matrix(y_test, predictions).tolist()}
    metadata = {"target": target, "inferred_target": inferred, "features": features, "numeric_features": numeric, "categorical_features": categorical, "defaults": {c: (float(frame[c].median()) if c in numeric else str(frame[c].mode().iloc[0])) for c in features}, "choices": {c: frame[c].astype(str).value_counts().head(30).index.tolist() for c in categorical}, "classes": sorted(y.unique().tolist()), "metrics": metrics, "plots": plots, "profile": profile_dataset(data_path)}
    (base_dir / "model" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def ensure_artifacts(base_dir: Path) -> dict:
    metadata = base_dir / "model" / "metadata.json"; model = base_dir / "model" / "saved_model.pkl"
    if not metadata.exists() or not model.exists(): return train(base_dir)
    return json.loads(metadata.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print(json.dumps(train(Path(__file__).resolve().parents[1])["metrics"], indent=2))
