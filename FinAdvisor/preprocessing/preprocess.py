"""Schema discovery, cleaning and reusable preprocessing helpers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RISK_HINTS = ("risk", "risk_level", "risk category", "risk_category")


def find_dataset(data_dir: Path) -> Path:
    files = [*data_dir.glob("*.csv"), *data_dir.glob("*.xlsx"), *data_dir.glob("*.xls")]
    if not files:
        raise FileNotFoundError("Place a CSV or Excel investment dataset in data/.")
    return files[0]


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    frame.columns = [re.sub(r"\s+", "_", str(column).strip().lower()) for column in frame.columns]
    # Convert metric-like text columns (e.g. '1.43') where the data supports it.
    for col in frame.select_dtypes(include="object"):
        converted = pd.to_numeric(frame[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        if converted.notna().mean() >= 0.75:
            frame[col] = converted
    return frame


def detect_target(frame: pd.DataFrame) -> str | None:
    normalized = {str(column).lower().replace(" ", "_"): column for column in frame.columns}
    for hint in RISK_HINTS:
        key = hint.replace(" ", "_")
        if key in normalized:
            return normalized[key]
    # Prefer a compact categorical column that explicitly indicates a class/label.
    candidates = [c for c in frame.columns if any(h in c.lower() for h in ("label", "class", "category"))]
    for col in candidates:
        if 2 <= frame[col].nunique(dropna=True) <= min(12, len(frame) // 4):
            return col
    return None


def create_risk_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Create Low/Medium/High labels from volatility and return metrics if absent."""
    numeric = frame.select_dtypes(include=np.number)
    volatility = next((c for c in numeric if any(h in c.lower() for h in ("volatility", "std", "sd", "beta"))), None)
    returns = [c for c in numeric if "return" in c.lower()]
    if volatility:
        score = frame[volatility].rank(pct=True)
    elif returns:
        score = 1 - frame[returns].mean(axis=1).rank(pct=True)
    else:
        score = pd.Series(0.5, index=frame.index)
    frame = frame.copy()
    frame["inferred_risk_level"] = pd.cut(score, [-np.inf, .33, .67, np.inf], labels=["Low", "Medium", "High"]).astype(str)
    return frame, "inferred_risk_level"


def profile_dataset(path: Path) -> dict[str, Any]:
    frame = load_dataset(path)
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = frame.select_dtypes(exclude=np.number).columns.tolist()
    target = detect_target(frame)
    return {
        "shape": list(frame.shape), "columns": frame.columns.tolist(), "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
        "missing": {c: int(v) for c, v in frame.isna().sum().items() if v}, "duplicates": int(frame.duplicated().sum()),
        "numeric_columns": numeric, "categorical_columns": categorical, "target": target or "Inferred at training time",
        "risk_labels_exist": bool(target and "risk" in target.lower()),
    }


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.drop_duplicates().copy()
    for col in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = frame[col].fillna(frame[col].median())
            low, high = frame[col].quantile([.01, .99])
            frame[col] = frame[col].clip(low, high)
        else:
            frame[col] = frame[col].fillna("Unknown").astype(str).str.strip()
    return frame
