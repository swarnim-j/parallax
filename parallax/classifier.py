import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class HazardClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, embedding: np.ndarray) -> float:
        self.eval()
        with torch.no_grad():
            x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            return torch.sigmoid(self.net(x)).item()

    def predict_batch(self, embeddings: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = torch.tensor(embeddings, dtype=torch.float32)
            return torch.sigmoid(self.net(x)).squeeze(-1).numpy()

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        json.dump({"input_dim": self.net[0].in_features}, open(path / "config.json", "w"))

    @classmethod
    def load(cls, path: str) -> "HazardClassifier":
        path = Path(path)
        config = json.load(open(path / "config.json"))
        model = cls(config["input_dim"])
        model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
        model.eval()
        return model


def train_classifier(
    hazard_embeddings: np.ndarray,
    benign_embeddings: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> tuple[HazardClassifier, dict]:
    X = np.vstack([hazard_embeddings, benign_embeddings])
    y = np.concatenate([np.ones(len(hazard_embeddings)), np.zeros(len(benign_embeddings))])

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split 90/10
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    input_dim = X.shape[1]
    model = HazardClassifier(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = torch.sigmoid(model(torch.tensor(X_val, dtype=torch.float32)).squeeze(-1))
                val_acc = ((val_pred > 0.5).float().numpy() == y_val).mean()
            print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/len(X_train):.4f}  val_acc={val_acc:.3f}", flush=True)

    model.eval()
    with torch.no_grad():
        val_pred = torch.sigmoid(model(torch.tensor(X_val, dtype=torch.float32)).squeeze(-1)).numpy()
        val_acc = ((val_pred > 0.5) == y_val).mean()
        val_pred_labels = (val_pred > 0.5).astype(int)
        tp = ((val_pred_labels == 1) & (y_val == 1)).sum()
        fp = ((val_pred_labels == 1) & (y_val == 0)).sum()
        fn = ((val_pred_labels == 0) & (y_val == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    metrics = {"val_acc": float(val_acc), "precision": float(precision), "recall": float(recall),
               "n_train": len(X_train), "n_val": len(X_val)}
    return model, metrics
