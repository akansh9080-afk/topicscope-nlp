"""TopicScope - Text Topic Modelling GUI (Streamlit).  Run: streamlit run app.py"""
import io
import pandas as pd
import streamlit as st
from pathlib import Path
from topic_pipeline import TopicModeler

st.set_page_config(page_title="TopicScope", page_icon="🧠", layout="wide")
st.title("🧠 TopicScope – Text Topic Modelling")
st.caption("Discover hidden themes in a collection of documents using LDA / NMF.")


@st.cache_data(show_spinner="Loading 20 Newsgroups sample...")
def load_20ng(n):
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    return pd.Series(d.data).sample(n, random_state=1).tolist()


# ------------------------- sidebar -------------------------
st.sidebar.header("⚙️ Settings")
method = st.sidebar.radio("Algorithm", ["LDA", "NMF"], help="LDA = probabilistic; NMF = TF-IDF based")
n_topics = st.sidebar.slider("Number of topics", 2, 20, 8)
n_words = st.sidebar.slider("Words per topic", 5, 15, 10)

# ------------------------- data input -------------------------
st.subheader("1. Provide documents")
source = st.radio("Source", ["Upload files (.txt / .csv)", "20 Newsgroups sample", "Paste text"],
                  horizontal=True)
docs, names = [], []

if source.startswith("Upload"):
    files = st.file_uploader("Upload multiple .txt files or one .csv", type=["txt", "csv"],
                             accept_multiple_files=True)
    for f in files or []:
        if f.name.endswith(".csv"):
            df = pd.read_csv(f)
            col = st.selectbox(f"Text column in {f.name}", df.columns, key=f.name)
            docs += df[col].dropna().astype(str).tolist()
            names += [f"{f.name}#{i+1}" for i in range(len(df[col].dropna()))]
        else:
            docs.append(f.read().decode("utf-8", errors="ignore"))
            names.append(f.name)
elif source.startswith("20"):
    n = st.slider("Sample size", 500, 5000, 2000, step=500)
    try:
        docs = load_20ng(n)
        names = [f"doc_{i+1}" for i in range(len(docs))]
    except Exception as e:
        st.error(f"Could not download dataset (internet needed once): {e}")
else:
    txt = st.text_area("One document per paragraph (separate with a blank line)", height=200)
    docs = [d.strip() for d in txt.split("\n\n") if d.strip()]
    names = [f"para_{i+1}" for i in range(len(docs))]

st.write(f"**{len(docs)}** documents ready.")

# ------------------------- run -------------------------
if st.button("🚀 Run Topic Modelling", type="primary", disabled=len(docs) < 2):
    with st.spinner("Cleaning text and training model..."):
        try:
            tm = TopicModeler(n_topics=n_topics, method=method).fit(docs)
            st.session_state.update(tm=tm, docs=docs, names=names)
        except ValueError as e:
            st.error(str(e))

if "tm" in st.session_state:
    tm, docs, names = st.session_state.tm, st.session_state.docs, st.session_state.names
    kept = tm.kept_idx
    t1, t2, t3, t4 = st.tabs(["📌 Topics", "📄 Document → Topic", "📊 Distribution", "🔮 Predict new text"])

    with t1:
        c = st.columns(3)
        c[0].metric("Documents used", len(kept))
        c[1].metric("Vocabulary size", tm.vocab_size)
        c[2].metric("Perplexity", f"{tm.perplexity:.0f}" if tm.perplexity else "n/a (NMF)")
        tw = tm.top_words(n_words)
        cols = st.columns(2)
        for k, words in enumerate(tw):
            with cols[k % 2].container(border=True):
                st.markdown(f"**Topic {k+1}**")
                st.bar_chart(pd.DataFrame(words, columns=["word", "weight"]).set_index("word"), height=220)

    res = pd.DataFrame({
        "document": [names[i] for i in kept],
        "preview": [docs[i][:120].replace("\n", " ") for i in kept],
        "topic": tm.doc_topics.argmax(axis=1) + 1,
        "confidence": tm.doc_topics.max(axis=1).round(3)})

    with t2:
        pick = st.multiselect("Filter by topic", sorted(res.topic.unique()))
        view = res[res.topic.isin(pick)] if pick else res
        st.dataframe(view, use_container_width=True, height=420)
        st.download_button("⬇️ Download CSV", res.to_csv(index=False), "document_topics.csv", "text/csv")

    with t3:
        st.bar_chart(res.topic.value_counts().sort_index().rename("documents"))
        st.write("Topic labels:")
        for k in range(tm.n_topics):
            st.write("• " + tm.topic_label(k))

    with t4:
        new = st.text_area("Enter new text", height=150)
        if st.button("Predict Topic") and new.strip():
            dist = tm.transform([new])[0]
            k = int(dist.argmax())
            st.success(f"Dominant: **{tm.topic_label(k, 5)}**  ({dist[k]:.0%})")
            st.bar_chart(pd.Series(dist, index=[f"T{i+1}" for i in range(len(dist))]))
