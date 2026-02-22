from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch_geometric.data import Data

from models.gnn_model import RiskGNN
from utils.graph_builder import build_graph
from utils.preprocessing import prepare_dataset, save_artifacts


def train_model(
    data_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    model_path: Path,
    epochs: int = 200,
    hidden_channels: int = 32,
    similarity: str = "cosine",
    k_neighbors: int = 6,
) -> Dict[str, List[float]]:
    bundle = prepare_dataset(data_dir=data_dir, processed_dir=processed_dir)

    edge_index, x = build_graph(bundle.features, similarity=similarity, k=k_neighbors)
    y = torch.tensor(bundle.labels, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, y=y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RiskGNN(
        in_channels=x.shape[1],
        hidden_channels=hidden_channels,
        out_channels=y.shape[1],
        dropout=0.3,
    ).to(device)

    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    history = {"loss": []}
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        history["loss"].append(float(loss.item()))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_channels": x.shape[1],
            "hidden_channels": hidden_channels,
            "out_channels": y.shape[1],
            "feature_names": bundle.feature_names,
            "label_names": bundle.label_names,
        },
        model_path,
    )

    save_artifacts(bundle, output_dir)

    return history


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    train_model(
        data_dir=base_dir / "data" / "raw",
        processed_dir=base_dir / "data" / "processed",
        output_dir=base_dir / "outputs",
        model_path=base_dir / "models" / "model.pt",
    )
