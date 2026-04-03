"""PyTorch neural network model for F1 race position prediction."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class F1NeuralNet(nn.Module):
    """Feed-forward neural network for F1 position classification.

    Uses BatchNorm, Dropout, and ReLU activations.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_classes: int,
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
    ) -> None:
        """Initialize F1NeuralNet architecture.

        Args:
            input_size: Number of input features.
            hidden_layers: List of hidden layer sizes.
            output_classes: Number of output classes (positions).
            dropout_rate: Dropout probability.
            batch_norm: Whether to add BatchNorm after each layer.
        """
        super().__init__()
        layers: List[nn.Module] = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Logit tensor of shape (batch_size, output_classes).
        """
        return self.network(x)


class NeuralNetModel(BaseModel):
    """Wrapper around F1NeuralNet with training loop and evaluation."""

    def __init__(self, config: Dict, input_size: Optional[int] = None) -> None:
        """Initialize NeuralNetModel.

        Args:
            config: Full model configuration dictionary.
            input_size: Number of input features. Can be inferred at train time.
        """
        super().__init__("neural_net", config)
        nn_cfg = config.get("models", {}).get("neural_net", {})
        arch = nn_cfg.get("architecture", {})
        hp = nn_cfg.get("hyperparameters", {})
        train_cfg = nn_cfg.get("training", {})

        self._input_size = input_size or arch.get("input_features")
        self._hidden_layers: List[int] = arch.get("hidden_layers", [256, 128, 64])
        self._output_classes: int = int(arch.get("output_classes", 20))
        self._dropout_rate: float = float(arch.get("dropout_rate", 0.3))
        self._batch_norm: bool = bool(arch.get("batch_norm", True))
        self._lr: float = float(hp.get("learning_rate", 0.001))
        self._weight_decay: float = float(hp.get("weight_decay", 0.0001))
        self._epochs: int = int(train_cfg.get("epochs", 100))
        self._batch_size: int = int(train_cfg.get("batch_size", 32))
        self._early_stopping: int = int(train_cfg.get("early_stopping", 20))
        self._device = torch.device(
            train_cfg.get("device", "cpu")
        )
        self._train_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }
        logger.info("NeuralNetModel initialized (device=%s)", self._device)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train neural network with early stopping.

        Args:
            X_train: Training features.
            y_train: Training labels (1-indexed, converted to 0-indexed).
            X_val: Optional validation features.
            y_val: Optional validation labels.
        """
        if self._input_size is None:
            self._input_size = X_train.shape[1]

        self._model = F1NeuralNet(
            input_size=self._input_size,
            hidden_layers=self._hidden_layers,
            output_classes=self._output_classes,
            dropout_rate=self._dropout_rate,
            batch_norm=self._batch_norm,
        ).to(self._device)

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        y_train_0 = np.clip(y_train.astype(int) - 1, 0, 19)
        train_loader = self._prepare_dataloader(
            X_train, y_train_0, self._batch_size, shuffle=True
        )
        val_loader = None
        if X_val is not None and y_val is not None:
            y_val_0 = np.clip(y_val.astype(int) - 1, 0, 19)
            val_loader = self._prepare_dataloader(
                X_val, y_val_0, self._batch_size, shuffle=False
            )

        best_val_loss = float("inf")
        no_improve = 0

        for epoch in range(self._epochs):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            self._train_history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self._validate(val_loader, criterion)
                self._train_history["val_loss"].append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self._early_stopping:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        "Epoch %d — train_loss=%.4f val_loss=%.4f",
                        epoch + 1,
                        train_loss,
                        val_loss,
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        "Epoch %d — train_loss=%.4f", epoch + 1, train_loss
                    )

        self._is_trained = True
        logger.info("NeuralNetModel training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict race positions (1-indexed).

        Args:
            X: Feature matrix.

        Returns:
            Integer array of positions 1-20.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("NeuralNetModel must be trained before predict()")
        proba = self.predict_proba(X)
        return (np.argmax(proba, axis=1) + 1).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probability matrix.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix of shape (n_samples, 20).
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("NeuralNetModel must be trained before predict_proba()")
        self._model.eval()
        tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Neural nets do not provide direct feature importance.

        Returns:
            None always.
        """
        return None

    def _prepare_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """Create a PyTorch DataLoader from numpy arrays.

        Args:
            X: Feature array.
            y: Label array.
            batch_size: Mini-batch size.
            shuffle: Whether to shuffle samples.

        Returns:
            DataLoader instance.
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Run one training epoch.

        Args:
            loader: Training DataLoader.
            optimizer: Optimizer instance.
            criterion: Loss function.

        Returns:
            Mean training loss for the epoch.
        """
        self._model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self._device)
            y_batch = y_batch.to(self._device)
            optimizer.zero_grad()
            logits = self._model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _validate(
        self, loader: DataLoader, criterion: nn.Module
    ) -> float:
        """Compute validation loss.

        Args:
            loader: Validation DataLoader.
            criterion: Loss function.

        Returns:
            Mean validation loss.
        """
        self._model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                logits = self._model(X_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)
