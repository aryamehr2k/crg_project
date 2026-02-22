from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.domain_mapper import map_domain_insight
from models.gnn_model import RiskGNN
from utils.graph_builder import build_graph
from utils.preprocessing import RISK_WEIGHT_MAP, prepare_dataset


def _compute_top_drivers(df: pd.DataFrame, label: str) -> pd.Series:
    contributions = []
    for feature, weight, direction in RISK_WEIGHT_MAP[label]:
        values = df[f"{feature}_norm"].to_numpy(dtype=np.float32)
        if direction == "inverse":
            values = 1 - values
        contributions.append(weight * values)
    contributions = np.vstack(contributions).T
    top_indices = np.argmax(contributions, axis=1)
    top_features = [RISK_WEIGHT_MAP[label][idx][0] for idx in top_indices]
    return pd.Series(top_features)


def predict_risks(
    data_dir: Path,
    processed_dir: Path,
    model_path: Path,
    similarity: str = "cosine",
    k_neighbors: int = 6,
) -> pd.DataFrame:
    bundle = prepare_dataset(data_dir=data_dir, processed_dir=processed_dir)
    edge_index, x = build_graph(bundle.features, similarity=similarity, k=k_neighbors)

    checkpoint = torch.load(model_path, map_location="cpu")
    model = RiskGNN(
        in_channels=checkpoint["in_channels"],
        hidden_channels=checkpoint["hidden_channels"],
        out_channels=checkpoint["out_channels"],
        dropout=0.3,
    )
    model.load_state_dict(checkpoint["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(x.to(device), edge_index.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()

    df = bundle.df.copy()
    for idx, label in enumerate(checkpoint["label_names"]):
        df[label] = probs[:, idx]
        df[f"top_driver_{label}"] = _compute_top_drivers(df, label)

    df["final_score"] = (
        0.3 * df.get("health_risk", 0.0)
        + 0.3 * df.get("climate_risk", 0.0)
        + 0.2 * df.get("education_risk", 0.0)
        + 0.2 * df.get("policy_risk", 0.0)
    ) * 100.0
    df["final_score"] = df["final_score"].clip(0.0, 100.0)

    df["model_confidence"] = probs.mean(axis=1)

    insights = df.apply(lambda row: map_domain_insight(row), axis=1)
    df["explanation_text"] = insights.apply(lambda item: item[0])
    df["severity_level"] = insights.apply(lambda item: item[1])

    return df


def predict_and_save(
    data_dir: Path,
    processed_dir: Path,
    model_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    df = predict_risks(
        data_dir=data_dir,
        processed_dir=processed_dir,
        model_path=model_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    predict_and_save(
        data_dir=base_dir / "data" / "raw",
        processed_dir=base_dir / "data" / "processed",
        model_path=base_dir / "models" / "model.pt",
        output_path=base_dir / "outputs" / "predictions.csv",
    )
