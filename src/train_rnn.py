"""
RNN text classifier using GloVe embeddings.
"""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import (
    set_seed, load_jsonl, split_data, get_label_encoder,
    GloVeDataset, compute_metrics, compute_per_class_metrics,
    count_parameters, EarlyStopping, train_epoch, evaluate, get_device,
    plot_confusion_matrix, plot_training_curves
)


class RNNClassifier(nn.Module):
    def __init__(self, emb_dim, num_classes, hidden_size=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(emb_dim, hidden_size), nn.Tanh())
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        out, _ = self.rnn(x)
        out = out.mean(dim=1)
        return self.fc(out)


def run_rnn(embeddings=None, emb_dim=None, epochs=50, lr=0.001, batch_size=64, hidden_size=128):
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    data = load_jsonl("data/emotion.jsonl")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    label_encoder = get_label_encoder(y_train)
    label_names = label_encoder.classes_.tolist()
    num_classes = len(label_names)

    if embeddings is None:
        raise FileNotFoundError("Need GloVe embeddings file")

    train_ds = GloVeDataset(X_train, y_train, embeddings, emb_dim, label_encoder)
    val_ds = GloVeDataset(X_val, y_val, embeddings, emb_dim, label_encoder)
    test_ds = GloVeDataset(X_test, y_test, embeddings, emb_dim, label_encoder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = RNNClassifier(emb_dim, num_classes, hidden_size=hidden_size).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stop = EarlyStopping(patience=10)

    train_losses, val_accs = [], []
    start_time = time.time()

    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(loss)
        scheduler.step()

        preds, labels = evaluate(model, val_loader, device)
        val_acc = (np.array(preds) == np.array(labels)).mean()
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")
        if early_stop(val_acc):
            print(f"Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - start_time

    preds, labels = evaluate(model, test_loader, device)
    metrics = compute_metrics(labels, preds, label_names)
    metrics['model'] = 'RNN'
    metrics['train_time'] = train_time
    metrics['params'] = count_parameters(model)
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")

    per_class = compute_per_class_metrics(labels, preds, label_names)
    for lbl, m in per_class.items():
        print(f"  {lbl}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    plot_confusion_matrix(cm, label_names, "RNN Confusion Matrix", "results/figures/cm_rnn.png")
    plot_training_curves(train_losses, val_accs, "RNN", "results/figures/curve_rnn.png")

    torch.save({'state_dict': model.state_dict(), 'config': {
        'emb_dim': emb_dim, 'num_classes': num_classes, 'label_names': label_names,
        'hidden_size': hidden_size
    }}, "models/rnn.pth")

    return metrics, labels, preds


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    from src.download_embeddings import load_glove_embeddings
    embeddings, emb_dim = load_glove_embeddings("data/glove.6B.300d.txt")
    run_rnn(embeddings=embeddings, emb_dim=emb_dim, epochs=50)
