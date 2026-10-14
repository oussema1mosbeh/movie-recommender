"""Step 1: get to know the data before doing any machine learning.

Run with:  .venv/Scripts/python.exe 01_explore.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA = Path("data/ml-latest-small")
CHARTS = Path("charts")
CHARTS.mkdir(exist_ok=True)

# A DataFrame is like a spreadsheet in Python: rows and named columns.
movies = pd.read_csv(DATA / "movies.csv")    # movieId, title, genres
ratings = pd.read_csv(DATA / "ratings.csv")  # userId, movieId, rating, timestamp
tags = pd.read_csv(DATA / "tags.csv")        # userId, movieId, tag, timestamp

print("=== First rows of each table ===")
print(movies.head(), "\n")
print(ratings.head(), "\n")
print(tags.head(), "\n")

print("=== Size of the dataset ===")
print(f"{len(movies):,} movies")
print(f"{len(ratings):,} ratings from {ratings.userId.nunique():,} users")
print(f"{len(tags):,} tags\n")

# Genres are stored as one string like "Adventure|Animation|Children".
# Splitting on "|" lets us count each genre separately.
genre_counts = movies.genres.str.split("|").explode().value_counts()
print("=== Movies per genre ===")
print(genre_counts, "\n")

# How many ratings does each movie get? Most get very few - this matters later,
# because it is hard to learn anything about a movie only one person rated.
per_movie = ratings.groupby("movieId").size()
print("=== Ratings per movie ===")
print(per_movie.describe().round(1), "\n")
print(f"Movies with only 1 rating: {(per_movie == 1).sum():,}\n")

# Most-rated movies, with their average score.
stats = ratings.groupby("movieId").rating.agg(["count", "mean"])
top = stats.sort_values("count", ascending=False).head(10).join(movies.set_index("movieId").title)
print("=== 10 most-rated movies ===")
print(top[["title", "count", "mean"]].round(2).to_string(index=False), "\n")

# --- Charts (saved as PNG files in the charts/ folder) ---
fig, ax = plt.subplots(figsize=(8, 5))
genre_counts.sort_values().plot.barh(ax=ax, color="#7c5cbf")
ax.set_title("Movies per genre")
ax.set_xlabel("Number of movies")
fig.tight_layout()
fig.savefig(CHARTS / "genres.png", dpi=120)

fig, ax = plt.subplots(figsize=(7, 4))
ratings.rating.value_counts().sort_index().plot.bar(ax=ax, color="#7c5cbf")
ax.set_title("How people rate (0.5 to 5 stars)")
ax.set_xlabel("Stars")
ax.set_ylabel("Number of ratings")
fig.tight_layout()
fig.savefig(CHARTS / "rating_distribution.png", dpi=120)

fig, ax = plt.subplots(figsize=(7, 4))
per_movie.clip(upper=100).plot.hist(bins=50, ax=ax, color="#7c5cbf")
ax.set_title("Ratings per movie (capped at 100)")
ax.set_xlabel("Number of ratings")
ax.set_ylabel("Number of movies")
fig.tight_layout()
fig.savefig(CHARTS / "ratings_per_movie.png", dpi=120)

print("Charts saved to the charts/ folder.")
