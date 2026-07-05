"""
Error analysis: confusion matrices, per-class metrics comparison, error cases.
"""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
from src.utils import (
    set_seed, load_jsonl, split_data, get_label_encoder,
    compute_metrics, compute_per_class_metrics, plot_confusion_matrix
)


def analyze_errors(model_name, y_true, y_pred, label_names, texts, save_dir="results"):
    """Analyze and visualize classification errors."""
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, label_names, f"{model_name} Confusion Matrix",
                          f"{save_dir}/figures/cm_{model_name.lower().replace(' ', '_')}.png")

    # Normalized confusion matrix (recall)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Normalized Confusion Matrix (Recall)')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/cm_norm_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()

    # Find error cases
    errors = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p:
            errors.append({
                'text': texts[i][:200],
                'true': label_names[t],
                'predicted': label_names[p],
                'idx': i
            })

    print(f"\n{model_name}: {len(errors)} errors out of {len(y_true)} ({len(errors)/len(y_true)*100:.1f}%)")

    # Most confused pairs
    pair_counts = Counter()
    for e in errors:
        pair_counts[f"{e['true']} -> {e['predicted']}"] += 1
    print("Top confused pairs:")
    for pair, count in pair_counts.most_common(10):
        print(f"  {pair}: {count}")

    # Sample error cases
    print("\nSample error cases:")
    rng = np.random.RandomState(42)
    samples = rng.choice(errors, min(10, len(errors)), replace=False)
    for s in samples:
        print(f"  True: {s['true']:10s} -> Pred: {s['predicted']:10s} | {s['text'][:120]}")

    return errors


def run_error_analysis(all_preds, save_dir="results"):
    """Run error analysis for all models."""
    data = load_jsonl("data/emotion.jsonl")
    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]
    label_encoder = get_label_encoder(labels)
    label_names = label_encoder.classes_.tolist()

    _, _, X_test, _, _, y_test = split_data(data)
    y_test_ids = label_encoder.transform(y_test)

    all_errors = {}
    for model_name, (true_ids, pred_ids) in all_preds.items():
        print(f"\n{'='*60}")
        print(f"Error Analysis: {model_name}")
        print(f"{'='*60}")
        # Get test texts
        errors = analyze_errors(model_name, true_ids, pred_ids, label_names,
                                X_test, save_dir)
        all_errors[model_name] = errors

    # Per-class F1 comparison across models
    print("\n\n" + "="*60)
    print("PER-CLASS F1 COMPARISON")
    print("="*60)
    per_class_data = {}
    for name, (true_ids, pred_ids) in all_preds.items():
        pcm = compute_per_class_metrics(true_ids, pred_ids, label_names)
        per_class_data[name] = {lbl: pcm[lbl]['f1'] for lbl in label_names}

    df_pc = pd.DataFrame(per_class_data).T
    print(df_pc.round(3))

    # Plot per-class F1
    df_pc.plot(kind='bar', figsize=(12, 5))
    plt.title("Per-Class F1 Score by Model")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=30, ha='right')
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/per_class_f1.png", dpi=150)
    plt.close()

    return all_errors


if __name__ == "__main__":
    import os
    os.makedirs("results/figures", exist_ok=True)
    # This should be called from run_all.py with actual predictions
    print("Error analysis module loaded. Use run_error_analysis(all_preds) to run.")
