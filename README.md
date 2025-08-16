# 🚀 DLCore

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
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

You can install DLCore via `pip` (after cloning or downloading):

```bash
git clone https://github.com/yourusername/dlcore.git
cd dlcore
pip install -r requirements.txt
```

---

## 🏗 Quick Start

### Import DLCore

```python
from dlcore.core import DLCore, GRU
import torch
import numpy as np

# Dummy data
X = np.random.rand(100, 10, 5)  # 100 samples, 10 timesteps, 5 features
y = np.random.rand(100, 1)      # Regression target

# Define model
model = GRU(input_size=5, hidden_size=32)

# Create DLCore trainer
trainer = DLCore(model=model, epochs=20, batch_size=16, loss_fn='mse', metrics=['mse', 'r2'])

# Train
trainer.fit(X, y)

# Predict
preds = trainer.predict(X)
print(preds[:5])
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

* `rnn_example.py`
* `cnn_example.py`
* `transformer_example.py`

---

## 📄 License

MIT License © 2025
Feel free to use, modify, and distribute DLCore in your projects.

---

## ❤️ Support / Contribute

If you enjoy DLCore, give it a ⭐ and feel free to submit pull requests or issues!

---

Made with ❤️ by **dev_pine**
