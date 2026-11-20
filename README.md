# 🔎 Semantic Job Matcher

Upload a resume and get the most relevant jobs, ranked by **semantic similarity** rather than exact keyword overlap.
Each match also shows **which skills you share** with the job and **which skills you'd need to learn**.

## How it works

```
resume (PDF/TXT) ──► text ──► sentence embedding ─┐
                                                  ├──► FAISS (inner product = cosine) ──► top-K jobs
job descriptions ──► sentence embeddings ─────────┘                │
                                                                   ▼
                                               skill extraction ──► shared / missing skills
```

1. **Embedding** – resume and every job (`title + description`) are encoded with
   `sentence-transformers/all-MiniLM-L6-v2` (384-dim), L2-normalised.
2. **Indexing & search** – job vectors go into a `faiss.IndexFlatIP`; with normalised vectors the inner product
   equals cosine similarity, so the top-K results are the most semantically similar jobs.
3. **Explanation** – a skill dictionary with aliases (e.g. `CNNs → cnn`, `sklearn → scikit-learn`) extracts skills
   from both texts; the intersection explains the match and the difference shows skill gaps.
4. **FAISS fallback** – if FAISS cannot be imported, an exact NumPy inner-product index with the same interface is used.
5. **Embedder fallbacks** – if PyTorch / sentence-transformers can't load, the app uses `model2vec` static embeddings
   (`minishlab/potion-base-8M`, no PyTorch); if that also fails it falls back to TF-IDF vectors
   (keyword-based) behind the same interface, so it still runs.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py                                   # web UI
python match_cli.py my_resume.pdf --k 5                # command line
python -m pytest -q                                    # tests
```

The first run downloads the MiniLM model (~90 MB).

## Project structure

```
app.py            Streamlit UI
match_cli.py      command-line interface
src/matcher.py    embedders, FAISS index, search, skill extraction
data/jobs.json    24 sample job descriptions (synthetic, for demo)
tests/            pytest unit tests
```

## Next steps

- LLM-generated explanation of *why* a resume fits a job (and how to improve it)
- Hybrid ranking: BM25 + dense embeddings
- Evaluate ranking quality with a small labelled set (precision@k / NDCG)
- Scrape or load real job postings; swap `IndexFlatIP` for `IndexHNSWFlat` at larger scale
