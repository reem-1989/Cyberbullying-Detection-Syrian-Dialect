"""Train and compare 9 classifiers on the Syrian-dialect cyberbullying dataset.

Run:  python train.py
Writes results/*.csv, results/summary.json and models/*.joblib for the dashboard.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import LABELS, clean

ROOT = Path(__file__).parent
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)

df = pd.read_excel(ROOT / "data" / "cyberbullying.xlsx")
df = df.dropna(subset=["comment", "bullying"])
df["bullying"] = df["bullying"].astype(int)
df["clean"] = df["comment"].map(clean)
df = df[df["clean"].str.len() > 0]

# class distribution
dist = df["bullying"].value_counts().rename_axis("label").reset_index(name="comments")
dist["category"] = dist["label"].map(LABELS)
dist.to_csv(ROOT / "results" / "class_distribution.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["bullying"], test_size=0.3, random_state=42, stratify=df["bullying"]
)
vec = CountVectorizer()
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

models = [
    MultinomialNB(), DecisionTreeClassifier(random_state=42), LinearSVC(random_state=42),
    AdaBoostClassifier(random_state=42), RandomForestClassifier(random_state=42),
    BaggingClassifier(random_state=42), LogisticRegression(max_iter=2000),
    SGDClassifier(random_state=42), KNeighborsClassifier(),
]

rows, fitted = [], {}
for m in models:
    name = m.__class__.__name__
    t0 = time.time(); m.fit(Xtr, y_train); t_train = time.time() - t0
    pred_te, pred_tr = m.predict(Xte), m.predict(Xtr)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_test, pred_te, average="weighted", zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_test, pred_te, average="macro", zero_division=0)
    # binary view: any bullying (label > 0) vs not bullying
    pb, rb, fb, _ = precision_recall_fscore_support(y_test > 0, pred_te > 0, average="binary", zero_division=0)
    rows.append({
        "model": name,
        "test_accuracy": accuracy_score(y_test, pred_te),
        "train_accuracy": accuracy_score(y_train, pred_tr),
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f_m,
        "binary_precision": pb, "binary_recall": rb, "binary_f1": fb,
        "binary_accuracy": accuracy_score(y_test > 0, pred_te > 0),
        "train_time_sec": t_train,
    })
    fitted[name] = (m, pred_te)
    print(f"{name:24s} test acc {rows[-1]['test_accuracy']:.3f}  weighted F1 {f_w:.3f}  binary precision {pb:.3f}")

res = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)
res.to_csv(ROOT / "results" / "model_comparison.csv", index=False)

best = res.iloc[0]["model"]
best_model, best_pred = fitted[best]
labels = sorted(df["bullying"].unique())
cm = confusion_matrix(y_test, best_pred, labels=labels)
pd.DataFrame(cm, index=labels, columns=labels).to_csv(ROOT / "results" / "confusion_matrix.csv")

joblib.dump(vec, ROOT / "models" / "vectorizer.joblib")
joblib.dump(best_model, ROOT / "models" / "best_model.joblib")

summary = {
    "comments": int(len(df)), "categories": int(df["bullying"].nunique()),
    "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
    "vocabulary": int(len(vec.vocabulary_)), "best_model": best,
}
(ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2))
print("\nBest model:", best)
