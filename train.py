"""Train + evaluate on 20 Newsgroups and save the model for the GUI.
Usage:  python train.py --topics 20 --method LDA
"""
import argparse
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score
from topic_pipeline import TopicModeler

p = argparse.ArgumentParser()
p.add_argument("--topics", type=int, default=20)
p.add_argument("--method", default="LDA", choices=["LDA", "NMF"])
a = p.parse_args()

data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
print(f"Loaded {len(data.data)} documents, {len(data.target_names)} true categories")

tm = TopicModeler(n_topics=a.topics, method=a.method).fit(data.data)

print("\n=== Discovered topics ===")
for k, words in enumerate(tm.top_words(10)):
    print(f"Topic {k+1:2d}: " + ", ".join(w for w, _ in words))

# Evaluation: compare discovered topics with the known newsgroup labels
y = np.array(data.target)[tm.kept_idx]
pred = tm.doc_topics.argmax(axis=1)
print("\n=== Evaluation ===")
if tm.perplexity:
    print(f"Perplexity (lower is better): {tm.perplexity:.1f}")
print(f"NMI         : {normalized_mutual_info_score(y, pred):.3f}")
print(f"Homogeneity : {homogeneity_score(y, pred):.3f}")
purity = sum(np.bincount(y[pred == k]).max() for k in set(pred)) / len(y)
print(f"Purity      : {purity:.3f}")

tm.save(f"models/topic_model_{a.method.lower()}.joblib")
print("\nModel saved in models/")
