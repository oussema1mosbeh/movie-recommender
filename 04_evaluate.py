"""Step 4: measure which recommender is actually best.

For each user we HIDE 20% of their ratings, train on the other 80%,
recommend 10 movies, and count how many hidden 4-5 star movies we found.

Run with:  run.bat 04_evaluate.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data/ml-latest-small")
CHARTS = Path("charts")
MIN_RATINGS = 20   # candidate movies: rated by at least 20 people
TOP_N = 10         # how many movies we recommend per user
LIKED = 4.0        # a hidden rating of 4 stars or more counts as "loved it"
rng = np.random.default_rng(42)  # fixed seed -> same results every run

movies = pd.read_csv(DATA / "movies.csv")
ratings = pd.read_csv(DATA / "ratings.csv")
tags = pd.read_csv(DATA / "tags.csv")

# All methods choose from the same pool of movies, so the comparison is fair.
counts = ratings.groupby("movieId").size()
pool = np.array(sorted(counts[counts >= MIN_RATINGS].index))
col = pd.Series(range(len(pool)), index=pool)  # movieId -> column number
ratings = ratings[ratings.movieId.isin(pool)].copy()

# --- 1. Split: hide 20% of each user's ratings --------------------------------
ratings["is_test"] = rng.random(len(ratings)) < 0.2
train = ratings[~ratings.is_test].copy()
test = ratings[ratings.is_test]
print(f"Training on {len(train):,} ratings, hiding {len(test):,} for the test.\n")

# User x movie table built ONLY from training data, centered per user.
train["centered"] = train.rating - train.groupby("userId").rating.transform("mean")
users = np.array(sorted(train.userId.unique()))
user_row = pd.Series(range(len(users)), index=users)
R = np.zeros((len(users), len(pool)))
R[user_row[train.userId], col[train.movieId]] = train.centered
seen = np.zeros_like(R, dtype=bool)
seen[user_row[train.userId], col[train.movieId]] = True

# --- 2. Build the two similarity tables (movie x movie) -----------------------
# Content-based: same TF-IDF idea as 02_content_based.py
desc = movies.set_index("movieId").loc[pool]
genre_words = (desc.genres.str.replace("(no genres listed)", "", regex=False)
               .str.replace("-", "").str.replace("|", " ").str.lower())
tag_words = tags.groupby("movieId").tag.apply(lambda t: " ".join(t.astype(str)).lower())
text = (genre_words + " ") * 2 + desc.index.map(tag_words).fillna("").to_series(index=desc.index)
sim_content = cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(text))

# Collaborative: same idea as 03_collaborative.py, but on training data only
sim_collab = cosine_similarity(R.T)

# Collaborative v3: two improvements found by experimenting, then
# re-checked on a different random split so we didn't just get lucky.
#  a) Use only WHO watched each movie (1 = rated it, 0 = didn't), not the stars.
#     Choosing to watch a movie already says a lot about your taste.
#  b) Each movie only listens to its 50 most similar movies (its "neighbors"),
#     so thousands of tiny, noisy similarities stop adding up.
NEIGHBORS = 50
watched = seen.astype(float)
sim_v3 = cosine_similarity(watched.T)

for sim in (sim_content, sim_collab, sim_v3):
    np.fill_diagonal(sim, 0)  # a movie shouldn't vote for itself

weakest = np.argsort(-sim_v3, axis=0)[NEIGHBORS:, :]
np.put_along_axis(sim_v3, weakest, 0, axis=0)  # keep only the top 50 per movie

# --- 3. Score every movie for every user --------------------------------------
# A user's score for a movie = sum over movies they rated of
#   (how similar the two movies are) x (how much they liked the one they rated).
# Matrix multiplication does this for all users and all movies at once.
popularity = np.bincount(col[train.movieId], minlength=len(pool)).astype(float)
scores = {
    "Random": rng.random(R.shape),
    "Popular": np.tile(popularity, (len(users), 1)),
    "Content (v1)": R @ sim_content,
    "Collaborative (v2)": R @ sim_collab,
    "Collaborative (v3)": watched @ sim_v3,
}

# --- 4. Compare recommendations with the hidden favorites ---------------------
loved = test[test.rating >= LIKED]
loved_by_user = loved.groupby("userId").movieId.apply(lambda m: set(col[m]))

results = {}
for name, s in scores.items():
    s = np.where(seen, -np.inf, s)  # never recommend something already rated
    top = np.argsort(-s, axis=1)[:, :TOP_N]
    precision, recall, hit = [], [], []
    for user, hidden in loved_by_user.items():
        found = len(hidden & set(top[user_row[user]]))
        precision.append(found / TOP_N)
        recall.append(found / len(hidden))
        hit.append(found > 0)
    results[name] = {
        "Precision@10": np.mean(precision),
        "Recall@10": np.mean(recall),
        "Hit rate": np.mean(hit),
    }

table = pd.DataFrame(results).T
print(f"Evaluated on {len(loved_by_user)} users with hidden 4-5 star movies.\n")
print(table.map(lambda v: f"{v:.1%}").to_string())
print("""
Precision@10 : of the 10 movies we recommend, how many did the user love?
Recall@10    : of the hidden loved movies, how many did we find?
Hit rate     : % of users who got at least one loved movie in their 10.""")

# --- 5. Chart -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))
table.plot.bar(ax=ax, color=["#7c5cbf", "#b39ddb", "#4a3a7a"], rot=0)
ax.set_title("Which recommender finds the movies people love?")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.tick_params(axis="x", labelsize=8)
fig.tight_layout()
fig.savefig(CHARTS / "evaluation.png", dpi=120)
print("\nChart saved to charts/evaluation.png")
