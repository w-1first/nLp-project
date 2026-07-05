"""
Traditional ML baselines: TF-IDF + Logistic Regression, SVM, Naive Bayes.
"""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from src.utils import (
    set_seed, load_jsonl, split_data, get_label_encoder,
    compute_metrics, compute_per_class_metrics, plot_confusion_matrix
)


def run_traditional():
    set_seed(42)
    data = load_jsonl("data/emotion.jsonl")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

    # Combine train+val for sklearn (no validation needed for these models)
    X_train_val = X_train + X_val
    y_train_val = y_train + y_val

    label_encoder = get_label_encoder(y_train_val)
    label_names = label_encoder.classes_.tolist()

    models = {
        "TF-IDF + LR": LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
        "TF-IDF + SVM": LinearSVC(max_iter=2000, random_state=42, dual=False),
        "TF-IDF + NB": MultinomialNB(),
    }

    results = []
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), lowercase=True)
        X_tr = vectorizer.fit_transform(X_train_val)
        X_te = vectorizer.transform(X_test)

        start = time.time()
        model.fit(X_tr, y_train_val)
        train_time = time.time() - start

        y_pred = model.predict(X_te)

        metrics = compute_metrics(y_test, y_pred, label_names)
        metrics['model'] = name
        metrics['train_time'] = train_time
        metrics['params'] = 0  # Not parameterized the same way
        results.append(metrics)

        y_test_ids = label_encoder.transform(y_test)
        y_pred_ids = label_encoder.transform(y_pred)
        per_class = compute_per_class_metrics(y_test_ids, y_pred_ids, label_names)
        print(f"Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}, Time: {train_time:.1f}s")
        for lbl, m in per_class.items():
            print(f"  {lbl}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

        cm = np.zeros((len(label_names), len(label_names)), dtype=int)
        for t, p in zip(y_test, y_pred):
            cm[label_encoder.transform([t])[0], label_encoder.transform([p])[0]] += 1
        plot_confusion_matrix(cm, label_names, f"{name} Confusion Matrix",
                              f"results/figures/cm_{name.replace(' ', '_').replace('+', '_')}.png")

    pd.DataFrame(results).to_csv("results/traditional_metrics.csv", index=False)
    print("\nTraditional ML done!")
    return results


if __name__ == "__main__":
    run_traditional()
