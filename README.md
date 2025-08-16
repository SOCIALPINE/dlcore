# 🚀 DLCore

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-green)
![License](https://img.shields.io/badge/license-MIT-orange)

DLCore is a **lightweight, flexible deep learning framework** built on PyTorch.
It allows you to train and evaluate neural networks (RNN, GRU, LSTM, Transformer, CNN) **easily** with minimal boilerplate code. Perfect for **regression** and **classification** tasks.

---

## ✨ Features

* ✅ Supports **RNN, GRU, LSTM, CNN, Transformer, Sequential**
* ✅ Compatible with **PyTorch tensors & NumPy arrays**
* ✅ Built-in **training loops, evaluation, metrics**
* ✅ **Early stopping** & checkpoint saving
* ✅ Flexible **metrics**: MSE, MAE, R2, Accuracy, F1
* ✅ Lightweight and **easy to integrate** into existing projects

---

## 📦 Installation

You can install DLCore via `pip` (recommended) or from GitHub:

```bash
# Install from PyPI
pip install deeplcore

# Or install latest version from GitHub
git clone https://github.com/SOCIALPINE/dlcore.git
cd dlcore
pip install -r requirements.txt
pip install .

```

---

## 🏗 Quick Start

### Import DLCore

```python
import numpy as np
import os
import matplotlib.pyplot as plt
import dlcore as tk

# checkpoints
os.makedirs("checkpoints", exist_ok=True)

# ------------------------------
# 1) RNN
# ------------------------------
rnn_model = tk.RNN(input_size=10, hidden_size=20, num_layers=2)
trainer = tk.DLCore(
    rnn_model,
    epochs=20,
    lr=0.001,
    batch_size=32,
    metrics=['mse', 'mae', 'r2'],
    early_stopping=True,
    patience=3,
    checkpoint_path="checkpoints/rnn_best.pt",
    task='regression',
    verbose=1,
)

X_train = np.random.rand(200, 5, 10)
y_train = np.random.rand(200, 5, 20)
X_val = np.random.rand(50, 5, 10)
y_val = np.random.rand(50, 5, 20)

print("=== RNN Regression Training ===")
trainer.fit(X_train, y_train, validation_data=(X_val, y_val))
preds = trainer.predict(X_val)
print("Prediction shape:", preds.shape)

# ------------------------------
# 2) CNN
# ------------------------------
cnn_model = tk.CNN(in_channels=3, out_channels=16)
trainer_cnn = tk.DLCore(
    cnn_model,
    epochs=10,
    batch_size=32,
    metrics=['mse','mae','r2'],
    checkpoint_path="checkpoints/cnn_best.pt",
    task='regression',
    verbose=1,
)

X_cnn = np.random.rand(150, 3, 50)
y_cnn = np.random.rand(150, 16, 50)

print("\n=== CNN Regression Training ===")
trainer_cnn.fit(X_cnn, y_cnn)
preds_cnn = trainer_cnn.predict(X_cnn)
print("CNN Prediction shape:", preds_cnn.shape)

# ------------------------------
# 3) MLP
# ------------------------------
mlp_model = tk.Sequential(
    tk.Linear(20, 50),
    tk.ReLU(),
    tk.Linear(50, 3),
)

trainer_cls = tk.DLCore(
    mlp_model,
    epochs=30,
    lr=0.005,
    batch_size=64,
    loss_fn='cross_entropy',
    metrics=['accuracy', 'f1'],
    early_stopping=True,
    patience=5,
    checkpoint_path="checkpoints/mlp_cls_best.pt",
    task='classification',
    verbose=1,
)

X_cls_train = np.random.rand(500, 20)
y_cls_train = np.random.randint(0, 3, size=(500,))
X_cls_val = np.random.rand(100, 20)
y_cls_val = np.random.randint(0, 3, size=(100,))

print("\n=== MLP Classification Training ===")
trainer_cls.fit(X_cls_train, y_cls_train, validation_data=(X_cls_val, y_cls_val))
preds_cls = trainer_cls.predict(X_cls_val)
print("Classification Prediction shape:", preds_cls.shape)
print("Sample predictions:", preds_cls[:10])

# ------------------------------
# 4) GRU
# ------------------------------
gru_model = tk.GRU(input_size=10, hidden_size=15, num_layers=2)
trainer_gru = tk.DLCore(
    gru_model,
    epochs=15,
    batch_size=32,
    metrics=['mse', 'r2'],
    checkpoint_path="checkpoints/gru_best.pt",
    task='regression',
    verbose=1
)
X_gru = np.random.rand(100, 5, 10)
y_gru = np.random.rand(100, 5, 15)
trainer_gru.fit(X_gru, y_gru)
print("\nGRU Regression Prediction shape:", trainer_gru.predict(X_gru).shape)

# ------------------------------
# 5) Transformer
# ------------------------------
transformer_model = tk.TransformerEncoder(d_model=10, nhead=2, num_layers=2)
trainer_trans = tk.DLCore(
    transformer_model,
    epochs=10,
    batch_size=16,
    metrics=['mse','r2'],
    checkpoint_path="checkpoints/trans_best.pt",
    task='regression',
    verbose=1
)
X_trans = np.random.rand(50, 5, 10)  # (seq_len, batch, features)
y_trans = np.random.rand(50, 5, 10)
trainer_trans.fit(X_trans, y_trans)
print("\nTransformer Regression Prediction shape:", trainer_trans.predict(X_trans).shape)


trainer.save_model("checkpoints/rnn_saved.pt")

trainer.load_model("checkpoints/rnn_saved.pt")
print("RNN model save/load success!")

```

---

## 📊 Supported Metrics

| Metric     | Task           |
| ---------- | -------------- |
| `mse`      | Regression     |
| `mae`      | Regression     |
| `r2`       | Regression     |
| `accuracy` | Classification |
| `f1`       | Classification |

---

## 🔧 Checkpoints & Early Stopping

DLCore supports:

```python
trainer = DLCore(model=model, early_stopping=True, patience=5, checkpoint_path='best_model.pth')
```

* Saves the **best model automatically**
* Stops training when validation loss **does not improve**

---

## 💡 Examples

See the `examples/` folder for ready-to-run examples:

* `example.py`
---

## 📄 License

MIT License © 2025
Feel free to use, modify, and distribute DLCore in your projects.

---

## ❤️ Support / Contribute

If you enjoy DLCore, give it a ⭐ and feel free to submit pull requests or issues!

---

Made with ❤️ by **dev_pine**
