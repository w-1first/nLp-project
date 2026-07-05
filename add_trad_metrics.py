"""Add traditional ML per-class metrics to CSV."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from src.utils import (
    set_seed, load_jsonl, split_data, get_label_encoder,
    compute_metrics, compute_per_class_metrics
)

set_seed(42)

data = load_jsonl("data/emotion.jsonl")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
label_encoder = get_label_encoder(y_train)
label_names = label_encoder.classes_.tolist()

vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

y_train_ids = label_encoder.transform(y_train)
y_test_ids = label_encoder.transform(y_test)

df_pc = pd.read_csv("results/per_class_metrics.csv")

for name, clf in [
    ("TF-IDF + LR", LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)),
    ("TF-IDF + SVM", LinearSVC(max_iter=2000, dual=False, random_state=42)),
    ("TF-IDF + NB", MultinomialNB()),
]:
    print(f"\n=== {name} ===")
    clf.fit(X_train_tfidf, y_train_ids)
    preds = clf.predict(X_test_tfidf)
    m = compute_metrics(y_test_ids, preds, label_names)
    print(f"Acc={m['accuracy']:.4f} Macro-F1={m['macro_f1']:.4f}")

    pcm = compute_per_class_metrics(y_test_ids, preds, label_names)
    for lbl, v in pcm.items():
        print(f"  {lbl}: P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}")

    df_pc = df_pc[df_pc['model'] != name]
    for lbl, v in pcm.items():
        df_pc = pd.concat([df_pc, pd.DataFrame([{
            'model': name, 'class': lbl,
            'precision': v['precision'], 'recall': v['recall'],
            'f1': v['f1'], 'support': v['support']
        }])], ignore_index=True)

df_pc.to_csv("results/per_class_metrics.csv", index=False)

print("\n=== Full Per-Class F1 ===")
pivot = df_pc.pivot(index='model', columns='class', values='f1')
print(pivot.round(3).to_string())

print("\nDONE - Traditional ML per-class metrics added.")
