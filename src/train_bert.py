"""
BERT fine-tuning for text classification.
"""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from src.utils import (
    set_seed, load_jsonl, split_data, get_label_encoder,
    BERTDataset, compute_metrics, compute_per_class_metrics,
    count_parameters, EarlyStopping, train_epoch, evaluate, get_device,
    plot_confusion_matrix, plot_training_curves
)


class BERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.2, freeze_bert=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)


def run_bert(model_name="bert-base-uncased", freeze_bert=False, suffix="",
             epochs=5, lr=2e-5, batch_size=32, max_len=128):
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    model_label = f"BERT{suffix}"

    data = load_jsonl("data/emotion.jsonl")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    label_encoder = get_label_encoder(y_train)
    label_names = label_encoder.classes_.tolist()
    num_classes = len(label_names)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = BERTDataset(X_train, y_train, tokenizer, label_encoder, max_len)
    val_ds = BERTDataset(X_val, y_val, tokenizer, label_encoder, max_len)
    test_ds = BERTDataset(X_test, y_test, tokenizer, label_encoder, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = BERTClassifier(model_name, num_classes, freeze_bert=freeze_bert).to(device)
    n_params = count_parameters(model)
    print(f"{model_label} Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

    train_time = time.time() - start_time

    preds, labels = evaluate(model, test_loader, device)
    metrics = compute_metrics(labels, preds, label_names)
    metrics['model'] = model_label
    metrics['train_time'] = train_time
    metrics['params'] = n_params
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")

    per_class = compute_per_class_metrics(labels, preds, label_names)
    for lbl, m in per_class.items():
        print(f"  {lbl}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    plot_confusion_matrix(cm, label_names, f"{model_label} Confusion Matrix",
                          f"results/figures/cm_{model_label.lower()}.png")
    plot_training_curves(train_losses, val_accs, model_label,
                         f"results/figures/curve_{model_label.lower()}.png")

    safe_name = model_label.lower().replace(' ', '_')
    torch.save({'state_dict': model.state_dict(), 'config': {
        'model_name': model_name, 'num_classes': num_classes,
        'label_names': label_names, 'freeze_bert': freeze_bert
    }}, f"models/{safe_name}.pth")

    return metrics, labels, preds


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)

    print("\n" + "="*60)
    print("BERT (Frozen) - Only train classifier head")
    print("="*60)
    bert_frozen = run_bert(freeze_bert=True, suffix=" (frozen)", epochs=10, lr=5e-4)

    print("\n" + "="*60)
    print("BERT (Full Fine-tune)")
    print("="*60)
    bert_full = run_bert(freeze_bert=False, suffix=" (full)")
