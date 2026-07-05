import sys, os
sys.path.insert(0, "/root/nlp_project")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from src.utils import load_jsonl, split_data, get_label_encoder, set_seed

set_seed(42)
data = load_jsonl("data/emotion.jsonl")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
label_encoder = get_label_encoder(y_train)
label_names = label_encoder.classes_.tolist()

vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
y_train_ids = label_encoder.transform(y_train)
y_test_ids = label_encoder.transform(y_test)

clf = LinearSVC(max_iter=2000, dual=False, random_state=42)
clf.fit(X_train_tfidf, y_train_ids)
preds = clf.predict(X_test_tfidf)

cm = confusion_matrix(y_test_ids, preds)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("TF-IDF + SVM - Normalized Confusion Matrix (Recall)")
plt.tight_layout()
plt.savefig("results/figures/cm_norm_TF-IDF___SVM.png", dpi=150)
plt.close()
print("Done: cm_norm_TF-IDF___SVM.png")
