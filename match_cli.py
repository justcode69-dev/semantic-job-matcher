"""Command line: python match_cli.py path/to/resume.pdf [--k 5] [--embedder auto|sentence|model2vec|tfidf]"""
import argparse

from src.matcher import JobMatcher, make_embedder, read_resume


def main():
    ap = argparse.ArgumentParser(description="Rank jobs for a resume using semantic search")
    ap.add_argument("resume", help="resume file (.pdf or .txt)")
    ap.add_argument("--jobs", default="data/jobs.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--embedder", default="auto", choices=["auto", "sentence", "model2vec", "tfidf"])
    a = ap.parse_args()

    matcher = JobMatcher.from_json(a.jobs, make_embedder(a.embedder))
    print(f"Embedder: {matcher.embedder.name}\n")
    for rank, m in enumerate(matcher.search(read_resume(a.resume), a.k), 1):
        print(f"{rank}. {m.job['title']} — {m.job['company']} ({m.job['location']})  score={m.score:.3f}")
        print(f"   matching skills: {', '.join(m.shared_skills) or '-'}")
        print(f"   skills to learn: {', '.join(m.missing_skills) or '-'}\n")


if __name__ == "__main__":
    main()
