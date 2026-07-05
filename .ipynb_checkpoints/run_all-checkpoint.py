"""
Master script: run all models, collect results, generate comparisons.

Usage:
    python run_all.py                    # Run everything
    python run_all.py --skip-bert        # Skip BERT (slowest)
    python run_all.py --models cnn,lstm  # Run only specific models
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.utils import (
    set_seed, load_jsonl, split_data, get_label_encoder,
    compute_metrics, plot_model_comparison
)
from src.download_embeddings import download_glove, load_glove_embeddings
from src.data_analysis import run_data_analysis


def run_all(skip_models=None, only_models=None):
    set_seed(42)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Determine which models to run
    all_model_names = ['traditional', 'cnn', 'rnn', 'lstm', 'transformer', 'bert']
    if only_models:
        model_names = [m for m in only_models if m in all_model_names]
    elif skip_models:
        model_names = [m for m in all_model_names if m not in skip_models]
    else:
        model_names = all_model_names

    print("=" * 60)
    print(f"Running models: {model_names}")
    print("=" * 60)

    # Step 1: Data analysis
    print("\n\n" + "#"*60)
    print("# STEP 1: Data Analysis")
    print("#"*60)
    run_data_analysis()

    # Step 2: Download GloVe embeddings (needed for CNN/RNN/LSTM/Transformer)
    embeddings, emb_dim = None, None
    if any(m in model_names for m in ['cnn', 'rnn', 'lstm', 'transformer']):
        print("\n\n" + "#"*60)
        print("# STEP 2: Loading GloVe Embeddings")
        print("#"*60)
        embeddings, emb_dim = load_glove_embeddings("data/glove.6B.300d.txt")

    all_metrics = []
    all_preds = {}

    # Step 3: Traditional ML
    if 'traditional' in model_names:
        print("\n\n" + "#"*60)
        print("# STEP 3: Traditional ML Baselines")
        print("#"*60)
        from src.train_traditional import run_traditional
        trad_results = run_traditional()
        all_metrics.extend(trad_results)

    # Step 4: CNN
    if 'cnn' in model_names:
        print("\n\n" + "#"*60)
        print("# STEP 4: CNN")
        print("#"*60)
        from src.train_cnn import run_cnn
        metrics, labels, preds = run_cnn(embeddings=embeddings, emb_dim=emb_dim)
        all_metrics.append(metrics)
        all_preds['CNN'] = (labels, preds)

    # Step 5: RNN
    if 'rnn' in model_names:
        print("\n\n" + "#"*60)
        print("# STEP 5: RNN")
        print("#"*60)
        from src.train_rnn import run_rnn
        metrics, labels, preds = run_rnn(embeddings=embeddings, emb_dim=emb_dim)
        all_metrics.append(metrics)
        all_preds['RNN'] = (labels, preds)

    # Step 6: LSTM variants
    if 'lstm' in model_names:
        print("\n\n" + "#"*60)
        print("# STEP 6: LSTM / BiLSTM")
        print("#"*60)
        from src.train_lstm import run_lstm
        results, preds_dict = run_lstm(embeddings=embeddings, emb_dim=emb_dim)
        for name, m in results.items():
            all_metrics.append(m)
        for name, (lbls, p) in preds_dict.items():
            all_preds[name] = (lbls, p)

    # Step 7: Transformer
    if 'transformer' in model_names:
        print("\n\n" + "#"*60)
        print("# STEP 7: Transformer Encoder")
        print("#"*60)
        from src.train_transformer import run_transformer
        metrics, labels, preds = run_transformer(embeddings=embeddings, emb_dim=emb_dim)
        all_metrics.append(metrics)
        all_preds['Transformer'] = (labels, preds)

    # Step 8: BERT
    if 'bert' in model_names:
        print("\n\n" + "#"*60)
        print("# STEP 8: BERT Fine-tuning")
        print("#"*60)
        from src.train_bert import run_bert

        print("\nBERT (Frozen)")
        metrics, labels, preds = run_bert(freeze_bert=True, suffix=" (frozen)")
        all_metrics.append(metrics)
        all_preds['BERT (frozen)'] = (labels, preds)

        print("\nBERT (Full Fine-tune)")
        metrics, labels, preds = run_bert(freeze_bert=False, suffix=" (full)")
        all_metrics.append(metrics)
        all_preds['BERT (full)'] = (labels, preds)

    # Step 9: Collect and compare results
    print("\n\n" + "#"*60)
    print("# STEP 9: Results Comparison")
    print("#"*60)

    df = pd.DataFrame(all_metrics)
    cols = ['model', 'accuracy', 'macro_f1', 'weighted_f1', 'train_time', 'params']
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values('accuracy', ascending=False)
    df.to_csv("results/all_metrics.csv", index=False)

    print("\n" + "="*80)
    print("FINAL COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))

    # Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_to_plot = ['accuracy', 'macro_f1', 'weighted_f1']
    for ax, metric in zip(axes, metrics_to_plot):
        if metric in df.columns:
            bars = ax.bar(df['model'], df[metric])
            ax.set_title(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            for bar, val in zip(bars, df[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{val:.3f}', ha='center', fontsize=7)
    plt.tight_layout()
    plt.savefig("results/figures/model_comparison.png", dpi=150)
    plt.close()

    # Training time vs accuracy
    if 'train_time' in df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(df['train_time'], df['accuracy'], s=100)
        for _, row in df.iterrows():
            plt.annotate(row['model'], (row['train_time'], row['accuracy']),
                        fontsize=8, xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Training Time (s)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Training Time')
        plt.tight_layout()
        plt.savefig("results/figures/accuracy_vs_time.png", dpi=150)
        plt.close()

    # Step 10: Error analysis
    if all_preds:
        print("\n\n" + "#"*60)
        print("# STEP 10: Error Analysis")
        print("#"*60)
        from src.error_analysis import run_error_analysis
        run_error_analysis(all_preds)

    print("\n\n" + "="*60)
    print("ALL DONE!")
    print(f"Results saved to results/")
    print("="*60)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', nargs='+', default=[], help='Models to skip')
    parser.add_argument('--only', nargs='+', default=None, help='Only run these models')
    parser.add_argument('--skip-bert', action='store_true', help='Skip BERT')
    args = parser.parse_args()

    skip = args.skip
    if args.skip_bert and 'bert' not in skip:
        skip = list(skip) + ['bert']

    run_all(skip_models=skip if skip else None, only_models=args.only)


if __name__ == "__main__":
    main()
