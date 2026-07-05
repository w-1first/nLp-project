"""
Dataset analysis: class distribution, text length, etc.
"""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from src.utils import load_jsonl


def run_data_analysis():
    data = load_jsonl("data/emotion.jsonl")
    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]

    lengths = [len(t.split()) for t in texts]
    label_counts = Counter(labels)
    total = len(data)

    print("=" * 50)
    print("DATASET ANALYSIS")
    print("=" * 50)
    print(f"Total samples: {total}")
    print(f"Number of classes: {len(label_counts)}")
    print(f"\nClass distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label:12s}: {count:5d} ({count/total*100:5.1f}%)")

    print(f"\nText length statistics (words):")
    print(f"  Min:    {np.min(lengths)}")
    print(f"  Max:    {np.max(lengths)}")
    print(f"  Mean:   {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.0f}")
    print(f"  Std:    {np.std(lengths):.1f}")
    print(f"  95th percentile: {np.percentile(lengths, 95):.0f}")

    # Class distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    colors = sns.color_palette("husl", len(label_counts))
    bars = ax.bar(label_counts.keys(), label_counts.values(), color=colors)
    for bar, v in zip(bars, label_counts.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{v}\n({v/total*100:.1f}%)", ha='center', fontsize=9)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=30)

    ax = axes[1]
    ax.hist(lengths, bins=40, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
    ax.axvline(np.median(lengths), color='green', linestyle='--', label=f'Median: {np.median(lengths):.0f}')
    ax.set_title("Text Length Distribution")
    ax.set_xlabel("Length (words)")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/figures/data_analysis.png", dpi=150)
    plt.close()
    print("\nSaved data_analysis.png")

    # Print sample texts per class
    print("\nSample texts per class:")
    seen = set()
    for d in data:
        if d["label"] not in seen:
            print(f"  [{d['label']}] {d['text'][:100]}")
            seen.add(d["label"])
        if len(seen) == len(label_counts):
            break


if __name__ == "__main__":
    import os
    os.makedirs("results/figures", exist_ok=True)
    run_data_analysis()
