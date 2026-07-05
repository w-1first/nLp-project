"""Add BERT and traditional ML per-class metrics to CSV."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.utils import (
    set_seed, load_jsonl, split_data, get_label_encoder,
    BERTDataset, compute_metrics, compute_per_class_metrics,
    count_parameters, evaluate, get_device
)
from src.train_bert import BERTClassifier

set_seed(42)
device = get_device()
print(f"Device: {device}")

data = load_jsonl("data/emotion.jsonl")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
label_encoder = get_label_encoder(y_train)
label_names = label_encoder.classes_.tolist()
num_classes = len(label_names)
y_test_ids = label_encoder.transform(y_test)

# ═══════════════════════════════════════
# BERT evaluation
# ═══════════════════════════════════════
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
test_ds = BERTDataset(X_test, y_test, tokenizer, label_encoder)
test_loader = DataLoader(test_ds, batch_size=32)

for variant, ckpt_path, freeze in [
    ("BERT (full)", "models/bert_(full).pth", False),
    ("BERT (frozen)", "models/bert_(frozen).pth", True),
]:
    print(f"\n=== {variant} ===")
    model = BERTClassifier(num_classes=num_classes, freeze_bert=freeze).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    preds, labels = evaluate(model, test_loader, device)
    m = compute_metrics(labels, preds, label_names)
    print(f"Acc={m['accuracy']:.4f} Macro-F1={m['macro_f1']:.4f}")

    pcm = compute_per_class_metrics(labels, preds, label_names)
    for lbl, v in pcm.items():
        print(f"  {lbl}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}")

    # Update all_metrics.csv with real P/R
    df = pd.read_csv("results/all_metrics.csv")
    df.loc[df['model'] == variant, ['macro_precision', 'macro_recall', 'macro_f1']] = [
        m['macro_precision'], m['macro_recall'], m['macro_f1']
    ]
    df = df.sort_values('accuracy', ascending=False)
    df.to_csv("results/all_metrics.csv", index=False)

    # Update per_class_metrics.csv
    df_pc = pd.read_csv("results/per_class_metrics.csv")
    df_pc = df_pc[df_pc['model'] != variant]
    for lbl, v in pcm.items():
        df_pc = pd.concat([df_pc, pd.DataFrame([{
            'model': variant, 'class': lbl,
            'precision': v['precision'], 'recall': v['recall'],
            'f1': v['f1'], 'support': v['support']
        }])], ignore_index=True)
    df_pc.to_csv("results/per_class_metrics.csv", index=False)

print("\n=== Updated all_metrics.csv ===")
df = pd.read_csv("results/all_metrics.csv")
print(df.to_string(index=False))

print("\n=== Updated Per-Class F1 ===")
df_pc = pd.read_csv("results/per_class_metrics.csv")
pivot = df_pc.pivot(index='model', columns='class', values='f1')
print(pivot.round(3).to_string())

print("\nDONE - BERT per-class metrics added.")
