"""Step 2: content-based recommender.

"If you liked this movie, here are movies that are DESCRIBED the same way."

Run with:  run.bat 02_content_based.py "toy story"
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data/ml-latest-small")
MIN_RATINGS = 10  # only recommend movies that at least 10 people rated

movies = pd.read_csv(DATA / "movies.csv")
ratings = pd.read_csv(DATA / "ratings.csv")
tags = pd.read_csv(DATA / "tags.csv")

# --- 1. Build a text description for every movie --------------------------

# Genres: "Adventure|Sci-Fi" -> "adventure scifi" (one word per genre).
movies["genre_words"] = (
    movies.genres.str.replace("(no genres listed)", "", regex=False)
    .str.replace("-", "")
    .str.replace("|", " ")
    .str.lower()
)

# Tags: glue together every tag users gave the movie, e.g. "pixar fun toys".
tag_words = tags.groupby("movieId").tag.apply(lambda t: " ".join(t.astype(str)).lower())
movies["tag_words"] = movies.movieId.map(tag_words).fillna("")

# Genres are repeated twice so they count a bit more than random tags.
movies["description"] = (movies.genre_words + " ") * 2 + movies.tag_words

# How many ratings each movie has (used to skip obscure movies).
movies["n_ratings"] = movies.movieId.map(ratings.groupby("movieId").size()).fillna(0)

# --- 2. Turn descriptions into numbers with TF-IDF -------------------------

# Each movie becomes a row of numbers, one column per word in the vocabulary.
# stop_words="english" drops useless words like "the", "and", "of".
vectorizer = TfidfVectorizer(stop_words="english")
tfidf = vectorizer.fit_transform(movies.description)
print(f"Each movie is now a list of {tfidf.shape[1]:,} numbers (one per word).\n")


# --- 3. Recommend with cosine similarity -----------------------------------

def find_movie(query):
    """Return the row index of the best-matching title (most-rated wins)."""
    matches = movies[movies.title.str.contains(query, case=False, regex=False)]
    if matches.empty:
        return None
    return matches.n_ratings.idxmax()


def recommend(query, n=10):
    idx = find_movie(query)
    if idx is None:
        print(f'No movie found matching "{query}".')
        return None

    # Similarity between the chosen movie and EVERY movie: 9,742 numbers in 0..1.
    scores = cosine_similarity(tfidf[idx], tfidf).ravel()

    result = movies.assign(similarity=scores)
    result = result[(result.index != idx) & (result.n_ratings >= MIN_RATINGS)]
    # Highest similarity first; if two movies tie, the more popular one wins.
    result = result.sort_values(["similarity", "n_ratings"], ascending=False).head(n)

    print(f"Because you liked: {movies.title[idx]}")
    print(f"  described as: {movies.description[idx][:100]}...\n")
    print(result[["title", "similarity", "genres"]].round(2).to_string(index=False))
    return result


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "Toy Story"
    recommend(query)
