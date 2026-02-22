from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def build_graph(
    node_features: np.ndarray,
    similarity: str = "cosine",
    k: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if node_features.ndim != 2:
        raise ValueError("node_features must be a 2D array")

    num_nodes = node_features.shape[0]
    if num_nodes < 2:
        raise ValueError("need at least 2 nodes to build a graph")

    k = max(1, min(k, num_nodes - 1))

    if similarity == "cosine":
        sim_matrix = cosine_similarity(node_features)
    elif similarity == "euclidean":
        distances = euclidean_distances(node_features)
        sim_matrix = 1.0 / (1.0 + distances)
    else:
        raise ValueError("similarity must be 'cosine' or 'euclidean'")

    np.fill_diagonal(sim_matrix, -np.inf)

    edges = []
    for i in range(num_nodes):
        neighbors = np.argpartition(sim_matrix[i], -k)[-k:]
        for j in neighbors:
            edges.append((i, j))
            edges.append((j, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_features, dtype=torch.float32)
    return edge_index, x
