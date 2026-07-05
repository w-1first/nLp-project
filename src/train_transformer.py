"""
Transformer Encoder text classifier built from scratch.
"""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import math
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, emb_dim, num_classes, d_model=256, n_heads=8, n_layers=4,
                 d_ff=512, max_len=128, dropout=0.2, pooling='cls'):
        super().__init__()
        self.input_proj = nn.Linear(emb_dim, d_model) if emb_dim != d_model else nn.Identity()
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.pooling = pooling
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        for layer in self.encoder:
            x = layer(x)
        if self.pooling == 'cls':
            out = x[:, 0, :]
        elif self.pooling == 'mean':
            out = x.mean(dim=1)
        elif self.pooling == 'max':
            out = x.max(dim=1)[0]
        else:
            out = x[:, -1, :]
        return self.classifier(out)


def run_transformer(embeddings=None, emb_dim=300, d_model=256, n_heads=8, n_layers=4,
                    d_ff=512, dropout=0.2, pooling='mean',
                    epochs=30, lr=0.0005, batch_size=64):
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

    model = TransformerClassifier(
        emb_dim, num_classes, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, dropout=dropout, pooling=pooling
    ).to(device)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stop = EarlyStopping(patience=8)

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
    metrics['model'] = 'Transformer'
    metrics['train_time'] = train_time
    metrics['params'] = n_params
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")

    per_class = compute_per_class_metrics(labels, preds, label_names)
    for lbl, m in per_class.items():
        print(f"  {lbl}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    plot_confusion_matrix(cm, label_names, "Transformer Confusion Matrix",
                          "results/figures/cm_transformer.png")
    plot_training_curves(train_losses, val_accs, "Transformer", "results/figures/curve_transformer.png")

    torch.save({'state_dict': model.state_dict(), 'config': {
        'emb_dim': emb_dim, 'num_classes': num_classes, 'label_names': label_names,
        'd_model': d_model, 'n_heads': n_heads, 'n_layers': n_layers
    }}, "models/transformer.pth")

    return metrics, labels, preds


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    from src.download_embeddings import load_glove_embeddings
    embeddings, emb_dim = load_glove_embeddings("data/glove.6B.300d.txt")
    run_transformer(embeddings=embeddings, emb_dim=emb_dim)
