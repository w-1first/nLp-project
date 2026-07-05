import json
import random
import time
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_jsonl(filepath="data/emotion.jsonl"):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]

    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=test_ratio,
        stratify=labels, random_state=seed
    )
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size,
        stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_label_encoder(labels):
    le = LabelEncoder()
    le.fit(labels)
    return le


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, label_encoder, max_len=100):
        self.texts = texts
        self.labels = label_encoder.transform(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].lower().split()[:self.max_len]
        ids = [self.vocab.get(t, self.vocab.get("<unk>", 0)) for t in tokens]
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [self.vocab.get("<pad>", 0)] * pad_len
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class GloVeDataset(Dataset):
    def __init__(self, texts, labels, embeddings, emb_dim, label_encoder, max_len=100):
        self.texts = texts
        self.labels = label_encoder.transform(labels)
        self.embeddings = embeddings
        self.emb_dim = emb_dim
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].lower().split()[:self.max_len]
        embeds = []
        for t in tokens:
            if t in self.embeddings:
                embeds.append(torch.tensor(self.embeddings[t]))
            else:
                embeds.append(torch.zeros(self.emb_dim))
        pad_len = self.max_len - len(embeds)
        if pad_len > 0:
            embeds.extend([torch.zeros(self.emb_dim)] * pad_len)
        return torch.stack(embeds), torch.tensor(self.labels[idx], dtype=torch.long)


class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_encoder, max_len=128):
        self.texts = texts
        self.labels = label_encoder.transform(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding='max_length',
            max_length=self.max_len, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }, torch.tensor(self.labels[idx], dtype=torch.long)


def compute_metrics(y_true, y_pred, label_names=None):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    return {
        'accuracy': acc,
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1': f1,
        'weighted_f1': weighted_f1
    }


def compute_per_class_metrics(y_true, y_pred, label_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(label_names)), zero_division=0
    )
    return {
        name: {'precision': p, 'recall': r, 'f1': f, 'support': s}
        for name, p, r, f, s in zip(label_names, precision, recall, f1, support)
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        return self.early_stop


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        if isinstance(batch[0], dict):
            inputs = {k: v.to(device) for k, v in batch[0].items()}
            labels = batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(**inputs)
        else:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch[0], dict):
                inputs = {k: v.to(device) for k, v in batch[0].items()}
                labels = batch[1]
                outputs = model(**inputs)
            else:
                inputs, labels = batch[0].to(device), batch[1]
                outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    return all_preds, all_labels


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_confusion_matrix(cm, label_names, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_curves(train_losses, val_accs, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Training Loss')
    ax2.plot(val_accs, marker='o', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} - Validation Accuracy')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_model_comparison(results_df, metric, save_path):
    plt.figure(figsize=(10, 5))
    bars = plt.bar(results_df['model'], results_df[metric])
    for bar, val in zip(bars, results_df[metric]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=9)
    plt.xlabel('Model')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
