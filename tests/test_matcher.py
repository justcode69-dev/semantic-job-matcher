from pathlib import Path

from src.matcher import JobMatcher, TfidfEmbedder, extract_skills

JOBS = Path(__file__).parent.parent / "data" / "jobs.json"


def matcher():
    return JobMatcher.from_json(JOBS, TfidfEmbedder())


def test_extract_skills_handles_aliases():
    s = extract_skills("Built CNNs in TensorFlow and REST APIs with Flask; know DSA")
    assert {"cnn", "tensorflow", "rest api", "flask", "data structures"} <= s


def test_android_resume_matches_android_job():
    top = matcher().search("Android developer, Kotlin, Java, Play Store apps, SQLite, REST APIs", k=1)[0]
    assert "Android" in top.job["title"]


def test_scores_sorted_and_bounded():
    res = matcher().search("Python machine learning TensorFlow transfer learning", k=5)
    scores = [m.score for m in res]
    assert scores == sorted(scores, reverse=True)
    assert all(-1e-5 <= s <= 1 + 1e-5 for s in scores)
