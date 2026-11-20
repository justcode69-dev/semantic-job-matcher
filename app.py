"""Streamlit UI: streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from src.matcher import JobMatcher, read_resume

st.set_page_config(page_title="Semantic Job Matcher", page_icon="🔎")
st.title("🔎 Semantic Job Matcher")


@st.cache_resource
def load_matcher():
    return JobMatcher.from_json(Path(__file__).parent / "data" / "jobs.json")


matcher = load_matcher()
method = "keyword similarity (TF-IDF)" if matcher.embedder.name == "tfidf" else "semantic similarity (sentence embeddings)"
st.caption(f"Upload a resume — jobs are ranked by {method} with {matcher.backend.upper()} vector search.")
st.sidebar.write(f"**Embedder:** `{matcher.embedder.name}`")
st.sidebar.write(f"**Search backend:** `{matcher.backend}`")
st.sidebar.write(f"**Jobs indexed:** {len(matcher.jobs)}")
k = st.sidebar.slider("Top K results", 1, 10, 5)

uploaded = st.file_uploader("Resume (PDF or TXT)", type=["pdf", "txt"])
pasted = st.text_area("…or paste resume text", height=150)

text = ""
if uploaded:
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as f:
        f.write(uploaded.getbuffer())
    text = read_resume(f.name)
elif pasted.strip():
    text = pasted

if text and st.button("Find matching jobs", type="primary"):
    for rank, m in enumerate(matcher.search(text, k), 1):
        with st.container(border=True):
            st.subheader(f"{rank}. {m.job['title']}")
            st.write(f"{m.job['company']} · {m.job['location']}")
            st.progress(max(0.0, min(1.0, m.score)), text=f"Similarity: {m.score:.2f}")
            st.write("✅ **Matching skills:** " + (", ".join(m.shared_skills) or "—"))
            st.write("📚 **Skills to learn:** " + (", ".join(m.missing_skills) or "—"))
            with st.expander("Job description"):
                st.write(m.job["description"])
