import numpy as np
import os
import dlcore as tk

os.makedirs("checkpoints", exist_ok=True)

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

print("RNN Training")
trainer.fit(X_train, y_train, validation_data=(X_val, y_val))
print("RNN Prediction:", trainer.predict(X_val).shape)

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
print("CNN Training")
trainer_cnn.fit(X_cnn, y_cnn)
print("CNN Prediction:", trainer_cnn.predict(X_cnn).shape)

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
print("MLP Training")
trainer_cls.fit(X_cls_train, y_cls_train, validation_data=(X_cls_val, y_cls_val))
print("MLP Prediction:", trainer_cls.predict(X_cls_val).shape)

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
print("GRU Training")
trainer_gru.fit(X_gru, y_gru)
print("GRU Prediction:", trainer_gru.predict(X_gru).shape)

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
X_trans = np.random.rand(50, 5, 10)
y_trans = np.random.rand(50, 5, 10)
print("Transformer Training")
trainer_trans.fit(X_trans, y_trans)
print("Transformer Prediction:", trainer_trans.predict(X_trans).shape)

trainer.save_model("checkpoints/rnn_saved.pt")
trainer.load_model("checkpoints/rnn_saved.pt")
print("RNN save/load OK")
