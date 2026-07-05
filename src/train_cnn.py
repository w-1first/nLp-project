"""
CNN text classifier using GloVe embeddings.
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


class CNNClassifier(nn.Module):
    def __init__(self, emb_dim, num_classes, num_filters=256, kernel_sizes=[2, 3, 4, 5], dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, k, padding=k//2) for k in kernel_sizes
        ])
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters * len(kernel_sizes), 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_outs = [torch.max_pool1d(torch.relu(conv(x)), x.shape[2]) for conv in self.convs]
        x = torch.cat([c.squeeze(2) for c in conv_outs], dim=1)
        return self.fc(x)


def run_cnn(embeddings=None, emb_dim=None, epochs=20, lr=0.001, batch_size=64):
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    data = load_jsonl("data/emotion.jsonl")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    label_encoder = get_label_encoder(y_train)
    label_names = label_encoder.classes_.tolist()
    num_classes = len(label_names)

    if embeddings is None:
        print("Loading GloVe embeddings from file...")
        import gzip, urllib.request, os
        emb_file = "glove.6B.300d.txt"
        if not os.path.exists(emb_file):
            # Use a local file instead
            raise FileNotFoundError("Need GloVe embeddings file")
        embeddings = {}
        with open(emb_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                embeddings[parts[0]] = np.array([float(x) for x in parts[1:]])
        emb_dim = 300

    train_ds = GloVeDataset(X_train, y_train, embeddings, emb_dim, label_encoder)
    val_ds = GloVeDataset(X_val, y_val, embeddings, emb_dim, label_encoder)
    test_ds = GloVeDataset(X_test, y_test, embeddings, emb_dim, label_encoder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = CNNClassifier(emb_dim, num_classes).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stop = EarlyStopping(patience=5)

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
    metrics['model'] = 'CNN'
    metrics['train_time'] = train_time
    metrics['params'] = count_parameters(model)
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")

    per_class = compute_per_class_metrics(labels, preds, label_names)
    for lbl, m in per_class.items():
        print(f"  {lbl}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    plot_confusion_matrix(cm, label_names, "CNN Confusion Matrix", "results/figures/cm_cnn.png")
    plot_training_curves(train_losses, val_accs, "CNN", "results/figures/curve_cnn.png")

    torch.save({'state_dict': model.state_dict(), 'config': {
        'emb_dim': emb_dim, 'num_classes': num_classes, 'label_names': label_names
    }}, "models/cnn.pth")

    return metrics, labels, preds


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    from src.download_embeddings import load_glove_embeddings
    embeddings, emb_dim = load_glove_embeddings("data/glove.6B.300d.txt")
    run_cnn(embeddings=embeddings, emb_dim=emb_dim)
