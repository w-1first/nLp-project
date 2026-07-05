"""
LSTM and BiLSTM text classifiers using GloVe embeddings.
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


class LSTMClassifier(nn.Module):
    def __init__(self, emb_dim, num_classes, hidden_size=256, num_layers=2,
                 dropout=0.3, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0)
        lstm_out = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.fc(out)


class BiLSTMWithAttention(nn.Module):
    """BiLSTM with additive attention pooling."""
    def __init__(self, emb_dim, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        lstm_out = hidden_size * 2
        self.attn = nn.Sequential(
            nn.Linear(lstm_out, lstm_out),
            nn.Tanh(),
            nn.Linear(lstm_out, 1)
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(out).squeeze(2), dim=1)
        out = torch.sum(out * weights.unsqueeze(2), dim=1)
        return self.fc(out)


def run_lstm_variant(model_cls, model_name, embeddings, emb_dim, label_encoder,
                     label_names, X_train, X_val, X_test, y_train, y_val, y_test,
                     epochs=20, lr=0.001, batch_size=64, **kwargs):
    device = get_device()
    num_classes = len(label_names)

    train_ds = GloVeDataset(X_train, y_train, embeddings, emb_dim, label_encoder)
    val_ds = GloVeDataset(X_val, y_val, embeddings, emb_dim, label_encoder)
    test_ds = GloVeDataset(X_test, y_test, embeddings, emb_dim, label_encoder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = model_cls(emb_dim, num_classes, **kwargs).to(device)
    n_params = count_parameters(model)
    print(f"{model_name} Parameters: {n_params:,}")

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
    metrics['model'] = model_name
    metrics['train_time'] = train_time
    metrics['params'] = n_params
    print(f"Test Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")

    per_class = compute_per_class_metrics(labels, preds, label_names)
    for lbl, m in per_class.items():
        print(f"  {lbl}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    safe_name = model_name.replace(' ', '_').replace('+', '_')
    plot_confusion_matrix(cm, label_names, f"{model_name} Confusion Matrix",
                          f"results/figures/cm_{safe_name}.png")
    plot_training_curves(train_losses, val_accs, model_name,
                         f"results/figures/curve_{safe_name}.png")

    torch.save({'state_dict': model.state_dict(), 'config': {
        'emb_dim': emb_dim, 'num_classes': num_classes, 'label_names': label_names
    }}, f"models/{safe_name}.pth")

    return metrics, labels, preds


def run_lstm(embeddings=None, emb_dim=None, epochs=20, lr=0.001, batch_size=64):
    set_seed(42)
    data = load_jsonl("data/emotion.jsonl")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    label_encoder = get_label_encoder(y_train)
    label_names = label_encoder.classes_.tolist()

    if embeddings is None:
        raise FileNotFoundError("Need GloVe embeddings file")

    results = {}
    all_preds = {}

    # LSTM
    print("\n" + "="*60)
    print("LSTM")
    print("="*60)
    m, lbls, preds = run_lstm_variant(
        LSTMClassifier, "LSTM", embeddings, emb_dim, label_encoder, label_names,
        X_train, X_val, X_test, y_train, y_val, y_test,
        epochs=epochs, lr=lr, batch_size=batch_size, bidirectional=False
    )
    results['LSTM'] = m
    all_preds['LSTM'] = (lbls, preds)

    # BiLSTM
    print("\n" + "="*60)
    print("BiLSTM")
    print("="*60)
    m, lbls, preds = run_lstm_variant(
        LSTMClassifier, "BiLSTM", embeddings, emb_dim, label_encoder, label_names,
        X_train, X_val, X_test, y_train, y_val, y_test,
        epochs=epochs, lr=lr, batch_size=batch_size, bidirectional=True
    )
    results['BiLSTM'] = m
    all_preds['BiLSTM'] = (lbls, preds)

    # BiLSTM + Attention
    print("\n" + "="*60)
    print("BiLSTM + Attention")
    print("="*60)
    m, lbls, preds = run_lstm_variant(
        BiLSTMWithAttention, "BiLSTM+Attention", embeddings, emb_dim, label_encoder, label_names,
        X_train, X_val, X_test, y_train, y_val, y_test,
        epochs=epochs, lr=lr, batch_size=batch_size
    )
    results['BiLSTM+Attention'] = m
    all_preds['BiLSTM+Attention'] = (lbls, preds)

    return results, all_preds


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    from src.download_embeddings import load_glove_embeddings
    embeddings, emb_dim = load_glove_embeddings("data/glove.6B.300d.txt")
    run_lstm(embeddings=embeddings, emb_dim=emb_dim)
