"""Streamlit dashboard: Cyberbullying Detection in Syrian-Dialect Arabic.

Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from preprocessing import LABELS, clean

ROOT = Path(__file__).parent
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"

st.set_page_config(page_title="Cyberbullying Detection Dashboard", page_icon="📊", layout="wide")


@st.cache_data
def load():
    res = pd.read_csv(ROOT / "results" / "model_comparison.csv")
    dist = pd.read_csv(ROOT / "results" / "class_distribution.csv")
    cm = pd.read_csv(ROOT / "results" / "confusion_matrix.csv", index_col=0)
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    return res, dist, cm, summary


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "vectorizer.joblib"), joblib.load(ROOT / "models" / "best_model.joblib")


def style(fig, height=380):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=20, t=10, b=10),
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=INK2),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(bgcolor="white", font_color=INK),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


res, dist, cm, summary = load()
best = res.iloc[0]

st.title("Cyberbullying Detection in Syrian-Dialect Arabic")
st.caption(
    "Master's thesis project · Facebook comments labeled into 13 categories · "
    "9 machine-learning models compared with Python and scikit-learn"
)

# ---- KPI row
k = st.columns(5)
k[0].metric("Comments analyzed", f"{summary['comments']:,}")
k[1].metric("Categories", summary["categories"])
k[2].metric("Best model", best["model"].replace("Classifier", ""))
k[3].metric("Test accuracy (13 classes)", f"{best['test_accuracy']:.1%}")
k[4].metric("Precision: flagging bullying", f"{best['binary_precision']:.1%}",
            help="Of the comments the best model flagged as bullying, the share that really were bullying.")

st.divider()

# ---- Data + model comparison
c1, c2 = st.columns(2)
with c1:
    st.subheader("What's in the dataset")
    d = dist.sort_values("comments")
    fig = px.bar(d, x="comments", y="category", orientation="h", text="comments")
    fig.update_traces(marker_color=BLUE, marker_cornerradius=4, textposition="outside",
                      hovertemplate="%{y}: %{x:,} comments<extra></extra>")
    fig.update_traces(cliponaxis=False)
    fig.update_layout(xaxis_title="Comments", yaxis_title=None, xaxis_range=[0, d.comments.max() * 1.12])
    st.plotly_chart(style(fig, 440), use_container_width=True)
    st.caption(f"{dist.loc[dist.label == 0, 'comments'].iloc[0] / dist.comments.sum():.0%} of comments are not bullying, "
               "so rare categories are hard to learn.")

with c2:
    st.subheader("Model comparison")
    metrics = {
        "Weighted F1 (13 classes)": "f1_weighted",
        "Test accuracy (13 classes)": "test_accuracy",
        "Macro F1 (13 classes)": "f1_macro",
        "Precision: flagging bullying": "binary_precision",
        "Recall: flagging bullying": "binary_recall",
    }
    choice = st.selectbox("Metric", list(metrics), label_visibility="collapsed")
    col = metrics[choice]
    r = res.sort_values(col)
    fig = px.bar(r, x=col, y="model", orientation="h", text=r[col].map("{:.1%}".format))
    fig.update_traces(marker_color=BLUE, marker_cornerradius=4, textposition="outside",
                      hovertemplate="%{y}: %{x:.1%}<extra></extra>")
    fig.update_layout(xaxis_title=choice, yaxis_title=None, xaxis_tickformat=".0%", xaxis_range=[0, 1.08])
    st.plotly_chart(style(fig, 400), use_container_width=True)

# ---- Overfitting + precision/recall trade-off
c3, c4 = st.columns(2)
with c3:
    st.subheader("Training vs. test accuracy")
    r = res.sort_values("test_accuracy", ascending=False)
    fig = go.Figure([
        go.Bar(name="Training", x=r.model, y=r.train_accuracy, marker_color=BLUE, marker_cornerradius=4,
               hovertemplate="%{x}<br>Training: %{y:.1%}<extra></extra>"),
        go.Bar(name="Test", x=r.model, y=r.test_accuracy, marker_color=ORANGE, marker_cornerradius=4,
               hovertemplate="%{x}<br>Test: %{y:.1%}<extra></extra>"),
    ])
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, yaxis_tickformat=".0%", xaxis_tickangle=-30)
    st.plotly_chart(style(fig), use_container_width=True)
    st.caption("A large gap between training and test accuracy means the model memorizes the training data (overfitting).")

with c4:
    st.subheader("Flagging bullying: precision vs. recall")
    r = res.sort_values("binary_precision", ascending=False)
    fig = go.Figure([
        go.Bar(name="Precision", x=r.model, y=r.binary_precision, marker_color=BLUE, marker_cornerradius=4,
               hovertemplate="%{x}<br>Precision: %{y:.1%}<extra></extra>"),
        go.Bar(name="Recall", x=r.model, y=r.binary_recall, marker_color=ORANGE, marker_cornerradius=4,
               hovertemplate="%{x}<br>Recall: %{y:.1%}<extra></extra>"),
    ])
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, yaxis_tickformat=".0%", xaxis_tickangle=-30)
    st.plotly_chart(style(fig), use_container_width=True)
    st.caption("All 12 bullying categories grouped into one. High precision means few false alarms; "
               "high recall means few bullying comments missed.")

# ---- Confusion matrix
st.subheader(f"Where the best model ({best['model']}) gets confused")
names = [LABELS[int(c)] for c in cm.columns]
fig = px.imshow(cm.values, x=names, y=names, color_continuous_scale="Blues", text_auto=True, aspect="auto")
fig.update_traces(hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Comments: %{z}<extra></extra>")
fig.update_layout(xaxis_title="Predicted category", yaxis_title="Actual category", coloraxis_showscale=False)
fig.update_xaxes(tickangle=-35, gridcolor="rgba(0,0,0,0)")
fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
st.plotly_chart(style(fig, 560), use_container_width=True)

# ---- Live prediction
st.divider()
st.subheader("Try it: classify a comment")
text = st.text_input("Type a comment in Syrian Arabic", placeholder="اكتب تعليقاً هنا…")
if text.strip():
    vec, model = load_model()
    cleaned = clean(text)
    if not cleaned:
        st.info("Nothing left after cleaning. Try a longer comment.")
    else:
        label = int(model.predict(vec.transform([cleaned]))[0])
        (st.success if label == 0 else st.warning)(f"**{LABELS[label]}**  (category {label})")
        st.caption(f"Cleaned text: {cleaned}")

with st.expander("Results table"):
    show = res[["model", "test_accuracy", "f1_weighted", "f1_macro", "binary_precision",
                "binary_recall", "train_accuracy", "train_time_sec"]].copy()
    st.dataframe(show.style.format({c: "{:.1%}" for c in show.columns[1:-1]} | {"train_time_sec": "{:.2f}s"}),
                 hide_index=True, use_container_width=True)
