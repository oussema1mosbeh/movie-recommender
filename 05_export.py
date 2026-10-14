"""Step 5: export the winning model (Collaborative v3) for the web demo.

This time we train on ALL ratings - the evaluation is done, so there is
nothing left to hide. The result is demo/model.js, which the web page loads.

Run with:  run.bat 05_export.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data/ml-latest-small")
OUT = Path("demo/model.js")
MIN_RATINGS = 20
NEIGHBORS = 50   # same settings that won in 04_evaluate.py

movies = pd.read_csv(DATA / "movies.csv").set_index("movieId")
ratings = pd.read_csv(DATA / "ratings.csv")

counts = ratings.groupby("movieId").size()
pool = np.array(sorted(counts[counts >= MIN_RATINGS].index))
col = pd.Series(range(len(pool)), index=pool)
users = pd.Series(range(ratings.userId.nunique()), index=sorted(ratings.userId.unique()))
ratings = ratings[ratings.movieId.isin(pool)]

# Who watched what: 1 = rated it, 0 = didn't.
watched = np.zeros((len(users), len(pool)))
watched[users[ratings.userId], col[ratings.movieId]] = 1

sim = cosine_similarity(watched.T)
np.fill_diagonal(sim, 0)


def clean_title(raw):
    """'Matrix, The (1999)' -> ('The Matrix', 1999)"""
    m = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", raw)
    title, year = (m.group(1), int(m.group(2))) if m else (raw, None)
    title = re.sub(r"^(.*), (The|A|An)( \(.*\))?$", r"\2 \1\3", title)
    return title, year


out_movies, out_neighbors = [], []
for j, movie_id in enumerate(pool):
    title, year = clean_title(movies.title[movie_id])
    out_movies.append({
        "t": title,
        "y": year,
        "g": movies.genres[movie_id].replace("(no genres listed)", "").split("|"),
        "n": int(counts[movie_id]),
    })
    # The 50 movies this one "listens to", as [index, similarity] pairs.
    top = np.argsort(-sim[:, j])[:NEIGHBORS]
    out_neighbors.append([[int(i), round(float(sim[i, j]), 3)] for i in top])

OUT.parent.mkdir(exist_ok=True)
payload = json.dumps({"movies": out_movies, "neighbors": out_neighbors}, separators=(",", ":"))
# A .js file (not .json) so the page also works when opened straight from disk.
OUT.write_text(f"window.MODEL={payload};\n", encoding="utf-8")
print(f"Exported {len(pool):,} movies to {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
