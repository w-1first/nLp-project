"""
Further experiments for the NLP report.
Exp 1: Learning rate comparison (CNN)
Exp 2: Pooling method comparison (LSTM: last vs mean vs max)
Exp 3: Dropout comparison (CNN)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import (
    set_seed, load_jsonl, split_data, get_label_encoder,
    GloVeDataset, compute_metrics, count_parameters,
    EarlyStopping, train_epoch, evaluate, get_device
)
from src.download_embeddings import load_glove_embeddings

device = get_device()
print(f"Device: {device}")

# Load data once
data = load_jsonl("data/emotion.jsonl")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
label_encoder = get_label_encoder(y_train)
label_names = label_encoder.classes_.tolist()
num_classes = len(label_names)

embeddings, emb_dim = load_glove_embeddings("data/glove.6B.300d.txt")
train_ds = GloVeDataset(X_train, y_train, embeddings, emb_dim, label_encoder)
val_ds = GloVeDataset(X_val, y_val, embeddings, emb_dim, label_encoder)
test_ds = GloVeDataset(X_test, y_test, embeddings, emb_dim, label_encoder)

os.makedirs("results/figures", exist_ok=True)

# ═══════════════════════════════════════════════════════════
# Further Experiment 1: Learning Rate Comparison
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FURTHER EXPERIMENT 1: Learning Rate Comparison (CNN)")
print("="*60)

from src.train_cnn import CNNClassifier

learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
lr_results = []

for lr in learning_rates:
    set_seed(42)
    print(f"\n--- LR = {lr} ---")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = CNNClassifier(emb_dim, num_classes, kernel_sizes=[2,3,4,5],
                          num_filters=256, dropout=0.4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    early_stop = EarlyStopping(patience=5)

    start = time.time()
    for epoch in range(30):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        preds, labels = evaluate(model, val_loader, device)
        val_acc = (np.array(preds) == np.array(labels)).mean()
        if early_stop(val_acc):
            print(f"  Early stop at epoch {epoch+1}")
            break

    train_time = time.time() - start
    preds, labels = evaluate(model, test_loader, device)
    m = compute_metrics(labels, preds, label_names)
    m['lr'] = lr
    m['train_time'] = train_time
    m['params'] = count_parameters(model)
    lr_results.append(m)
    print(f"  Acc={m['accuracy']:.4f} Macro-F1={m['macro_f1']:.4f} Time={train_time:.1f}s")

# Plot LR comparison
df_lr = pd.DataFrame(lr_results)
df_lr.to_csv("results/lr_comparison.csv", index=False)

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
ax1.plot(df_lr['lr'].astype(str), df_lr['accuracy'], 'b-o', label='Accuracy')
ax1.plot(df_lr['lr'].astype(str), df_lr['macro_f1'], 'g-s', label='Macro-F1')
ax2.plot(df_lr['lr'].astype(str), df_lr['train_time'], 'r--D', label='Train Time')
ax1.set_xlabel('Learning Rate')
ax1.set_ylabel('Score')
ax2.set_ylabel('Time (s)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Effect of Learning Rate on CNN Performance')
plt.tight_layout()
plt.savefig("results/figures/lr_comparison.png", dpi=150)
plt.close()

print("\n=== LR Comparison Table ===")
print(df_lr[['lr', 'accuracy', 'macro_f1', 'train_time']].to_string(index=False))

# ═══════════════════════════════════════════════════════════
# Further Experiment 2: Pooling Method Comparison
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FURTHER EXPERIMENT 2: Pooling Method Comparison (LSTM)")
print("="*60)

class LSTMWithPooling(nn.Module):
    """LSTM classifier with configurable pooling."""
    def __init__(self, emb_dim, num_classes, hidden_size=256, num_layers=2,
                 dropout=0.3, bidirectional=False, pooling='mean'):
        super().__init__()
        self.pooling = pooling
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0)
        dirs = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * dirs, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        if self.pooling == 'last':
            out = out[:, -1, :]
        elif self.pooling == 'max':
            out = out.max(dim=1)[0]
        else:  # mean
            out = out.mean(dim=1)
        return self.fc(out)

pooling_results = []
for pooling in ['last', 'mean', 'max']:
    set_seed(42)
    print(f"\n--- Pooling: {pooling} ---")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = LSTMWithPooling(emb_dim, num_classes, hidden_size=256, num_layers=2,
                            dropout=0.3, bidirectional=False, pooling=pooling).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    early_stop = EarlyStopping(patience=5)

    start = time.time()
    for epoch in range(30):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        preds, labels = evaluate(model, val_loader, device)
        val_acc = (np.array(preds) == np.array(labels)).mean()
        if early_stop(val_acc):
            print(f"  Early stop at epoch {epoch+1}")
            break

    train_time = time.time() - start
    preds, labels = evaluate(model, test_loader, device)
    m = compute_metrics(labels, preds, label_names)
    m['pooling'] = pooling
    m['train_time'] = train_time
    m['params'] = count_parameters(model)
    pooling_results.append(m)
    print(f"  Acc={m['accuracy']:.4f} Macro-F1={m['macro_f1']:.4f} Time={train_time:.1f}s")

df_pool = pd.DataFrame(pooling_results)
df_pool.to_csv("results/pooling_comparison.csv", index=False)

# Plot pooling comparison
fig, ax = plt.subplots(figsize=(7, 5))
x = df_pool['pooling']
ax.bar(x, df_pool['accuracy'], color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
for i, (acc, f1) in enumerate(zip(df_pool['accuracy'], df_pool['macro_f1'])):
    ax.text(i, acc + 0.005, f'Acc: {acc:.3f}\nF1: {f1:.3f}', ha='center', fontsize=10)
ax.set_ylabel('Accuracy')
ax.set_title('Effect of Pooling Method on LSTM Performance')
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig("results/figures/pooling_comparison.png", dpi=150)
plt.close()

print("\n=== Pooling Comparison Table ===")
print(df_pool[['pooling', 'accuracy', 'macro_f1', 'train_time']].to_string(index=False))

# ═══════════════════════════════════════════════════════════
# Further Experiment 3: Dropout Comparison
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FURTHER EXPERIMENT 3: Dropout Comparison (CNN)")
print("="*60)

dropouts = [0.0, 0.2, 0.4, 0.6, 0.8]
dropout_results = []

for do in dropouts:
    set_seed(42)
    print(f"\n--- Dropout = {do} ---")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = CNNClassifier(emb_dim, num_classes, kernel_sizes=[2,3,4,5],
                          num_filters=256, dropout=do).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    early_stop = EarlyStopping(patience=5)

    train_losses, val_accs = [], []
    start = time.time()
    for epoch in range(30):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(loss)
        scheduler.step()
        preds, labels = evaluate(model, val_loader, device)
        val_acc = (np.array(preds) == np.array(labels)).mean()
        val_accs.append(val_acc)
        if early_stop(val_acc):
            print(f"  Early stop at epoch {epoch+1}")
            break

    train_time = time.time() - start
    preds, labels = evaluate(model, test_loader, device)
    m = compute_metrics(labels, preds, label_names)
    m['dropout'] = do
    m['train_time'] = train_time
    m['params'] = count_parameters(model)

    # Get val metrics too
    val_preds, val_labels = evaluate(model, val_loader, device)
    val_m = compute_metrics(val_labels, val_preds, label_names)
    m['val_accuracy'] = val_m['accuracy']
    m['best_epoch'] = len(train_losses)
    dropout_results.append(m)
    print(f"  Train Acc={m['accuracy']:.4f} Val Acc={val_m['accuracy']:.4f} Macro-F1={m['macro_f1']:.4f}")

df_do = pd.DataFrame(dropout_results)
df_do.to_csv("results/dropout_comparison.csv", index=False)

# Plot dropout comparison with train/val gap
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(dropouts))
w = 0.35
ax.bar(x - w/2, df_do['accuracy'], w, label='Test Accuracy', color='#2ecc71')
ax.bar(x + w/2, df_do['val_accuracy'], w, label='Val Accuracy', color='#3498db')
gap = df_do['accuracy'] - df_do['val_accuracy']
for i, g in enumerate(gap):
    ax.annotate(f'gap={g:.3f}', (i, max(df_do['accuracy'].iloc[i], df_do['val_accuracy'].iloc[i]) + 0.01),
                ha='center', fontsize=9, color='red' if abs(g) > 0.05 else 'green')
ax.set_xticks(x)
ax.set_xticklabels([str(d) for d in dropouts])
ax.set_xlabel('Dropout')
ax.set_ylabel('Accuracy')
ax.set_title('Effect of Dropout on CNN (Train/Val Gap)')
ax.legend()
plt.tight_layout()
plt.savefig("results/figures/dropout_comparison.png", dpi=150)
plt.close()

print("\n=== Dropout Comparison Table ===")
print(df_do[['dropout', 'accuracy', 'val_accuracy', 'macro_f1', 'best_epoch']].to_string(index=False))

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ALL FURTHER EXPERIMENTS COMPLETE")
print("Results saved to results/")
print("="*60)
print("Files: lr_comparison.csv, pooling_comparison.csv, dropout_comparison.csv")
