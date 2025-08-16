import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
import numpy as np
import os
from typing import Union, Callable, Dict, List

# -----------------------------
# Base Module & Layers
# -----------------------------
class BaseModel(nn.Module):
    """BaseModel"""
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError("BaseModel requires forward() implementation.")

# 기본 레이어 래퍼
class Module(nn.Module):
    def __init__(self):
        super().__init__()

Linear = nn.Linear
ReLU = nn.ReLU
MSELoss = nn.MSELoss
CrossEntropyLoss = nn.CrossEntropyLoss
Adam = optim.Adam

# -----------------------------
# RNN / GRU / LSTM / Transformer / CNN / Sequential
# -----------------------------
class RNN(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers,
                          nonlinearity=nonlinearity, batch_first=batch_first)
    def forward(self, x, h0=None):
        output, hn = self.rnn(x, h0)
        return output, hn

class GRU(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
    def forward(self, x, h0=None):
        output, hn = self.gru(x, h0)
        return output, hn

class LSTM(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
    def forward(self, x, h0=None):
        output, (hn, cn) = self.lstm(x, h0)
        return output, (hn, cn)

class TransformerEncoder(BaseModel):
    def __init__(self, d_model, nhead, num_layers=1, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        return self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

class CNN(BaseModel):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Sequential(BaseModel):
    def __init__(self, *modules):
        super().__init__()
        self.net = nn.Sequential(*modules)
    def forward(self, x):
        return self.net(x)

# -----------------------------
# DLCore
# -----------------------------
class DLCore(BaseEstimator, RegressorMixin):
    def __init__(self,
                 model: nn.Module,
                 epochs: int = 10,
                 lr: float = 0.001,
                 batch_size: int = 32,
                 device: str = None,
                 loss_fn: str = 'mse',
                 metrics: Union[List[str], Dict[str, Callable]] = None,
                 early_stopping: bool = False,
                 patience: int = 5,
                 checkpoint_path: str = None,
                 task: str = 'regression',
                 verbose: int = 1):

        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # ----- Loss 선택 -----
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_fn == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_fn == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("loss_fn must be one of ['mse','mae','cross_entropy']")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # ----- Metrics -----
        if metrics is None:
            self.metrics = {}
        elif isinstance(metrics, list):
            mapping = {
                "mse": mean_squared_error,
                "mae": mean_absolute_error,
                "r2": r2_score,
                "accuracy": accuracy_score,
                "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted")
            }
            self.metrics = {m: mapping[m] for m in metrics if m in mapping}
        elif isinstance(metrics, dict):
            self.metrics = metrics
        else:
            raise ValueError("metrics must be list[str] or dict[str, callable]")

        self.early_stopping = early_stopping
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.task = task
        self.verbose = verbose

        self.best_loss = float('inf')
        self.no_improve_count = 0
        self.history = {'loss': [], 'val_loss': [], 'metrics': []}

    # ------------------------------
    # Utility
    # ------------------------------
    def _to_tensor(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        return data.float().to(self.device)

    def _calc_metrics(self, y_true, y_pred):
        # flatten multi-dim output
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        if y_true.ndim > 1:
            y_true = y_true.reshape(y_true.shape[0], -1)
            y_pred = y_pred.reshape(y_pred.shape[0], -1)
        
        results = {}
        for name, fn in self.metrics.items():
            try:
                if name == 'r2':
                    with np.errstate(divide='ignore', invalid='ignore'):
                        results[name] = fn(y_true, y_pred)
                        if np.isnan(results[name]):
                            results[name] = 0.0
                else:
                    results[name] = fn(y_true, y_pred)
            except Exception:
                results[name] = np.nan
        return results


    # ------------------------------
    # Training / Evaluation
    # ------------------------------
    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            if self.task == 'classification':
                yb = yb.long()

            self.optimizer.zero_grad()
            output = self.model(xb)
            if isinstance(output, tuple):
                output = output[0]
            loss = self.criterion(output, yb)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def fit(self, X, y, validation_data=None):
        X, y = self._to_tensor(X), self._to_tensor(y)
        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if validation_data:
            X_val, y_val = validation_data
            X_val, y_val = self._to_tensor(X_val), self._to_tensor(y_val)
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(1, self.epochs+1):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = (None, None)
            if val_loader:
                val_loss, val_metrics = self.evaluate(val_loader)

            self.history['loss'].append(train_loss)
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                self.history['metrics'].append(val_metrics)

            if self.verbose:
                msg = f"Epoch {epoch}/{self.epochs} - Train loss: {train_loss:.4f}"
                if val_loss is not None:
                    msg += f", Val loss: {val_loss:.4f}"
                    for k,v in val_metrics.items():
                        msg += f", {k}: {v:.4f}"
                print(msg)

            if self.early_stopping and val_loss is not None:
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.no_improve_count = 0
                    self.save_model(self.checkpoint_path)
                else:
                    self.no_improve_count += 1
                    if self.no_improve_count >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
            else:
                if self.checkpoint_path:
                    self.save_model(self.checkpoint_path)

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self.load_model(self.checkpoint_path)

        return self

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss, all_preds, all_targets = 0, [], []
        with torch.no_grad():
            for xb, yb in data_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                if self.task == 'classification':
                    yb = yb.long()
                output = self.model(xb)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.criterion(output, yb)
                total_loss += loss.item()
                preds = output.argmax(dim=1).cpu().numpy() if self.task=='classification' else output.cpu().numpy()
                all_preds.append(preds)
                all_targets.append(yb.cpu().numpy())
        avg_loss = total_loss / len(data_loader)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics_res = self._calc_metrics(all_targets, all_preds)
        return avg_loss, metrics_res

    def predict(self, X):
        self.model.eval()
        X_tensor = self._to_tensor(X)
        with torch.no_grad():
            preds = self.model(X_tensor)
            if isinstance(preds, tuple):
                preds = preds[0]
            if self.task == 'classification':
                preds = preds.argmax(dim=1)
        return preds.cpu().numpy()

    # ------------------------------
    # Save / Load
    # ------------------------------
    def save_model(self, path: str):
        if path:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        if path and os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
