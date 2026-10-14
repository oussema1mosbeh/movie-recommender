"""Step 3: collaborative filtering (item-based).

"People who loved this movie also loved these movies."
No genres or tags are used - only who rated what, and how much.

Run with:  run.bat 03_collaborative.py "matrix"
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data/ml-latest-small")
MIN_RATINGS = 20  # a movie needs at least this many ratings to be compared fairly

movies = pd.read_csv(DATA / "movies.csv").set_index("movieId")
ratings = pd.read_csv(DATA / "ratings.csv")

# --- 1. Keep only movies with enough ratings --------------------------------
# With 2 or 3 ratings, two movies can look "identical" just by chance.
counts = ratings.groupby("movieId").size()
popular = counts[counts >= MIN_RATINGS].index
ratings = ratings[ratings.movieId.isin(popular)].copy()
print(f"Using {ratings.movieId.nunique():,} movies with at least {MIN_RATINGS} ratings.")

# --- 2. Remove each person's personal bias ------------------------------------
# Rating minus that user's average: +1.5 means "much better than I usually rate",
# -1.0 means "worse than usual". A 4-star from a harsh critic now counts more.
ratings["centered"] = ratings.rating - ratings.groupby("userId").rating.transform("mean")

# --- 3. Build the movie x user table ------------------------------------------
# 'Sparse' = we only store the cells that have a rating (about 5% are filled),
# empty cells count as 0, meaning "no opinion".
movie_ids = ratings.movieId.astype("category")
user_ids = ratings.userId.astype("category")
table = csr_matrix((ratings.centered, (movie_ids.cat.codes, user_ids.cat.codes)))
row_to_movie = movie_ids.cat.categories  # row number -> movieId
movie_to_row = pd.Series(range(len(row_to_movie)), index=row_to_movie)
print(f"Table size: {table.shape[0]:,} movies x {table.shape[1]:,} users\n")


# --- 4. Recommend -------------------------------------------------------------

def find_movie(query):
    """Return the movieId of the best-matching title among usable movies."""
    usable = movies.loc[row_to_movie]
    matches = usable[usable.title.str.contains(query, case=False, regex=False)]
    if matches.empty:
        return None
    return counts[matches.index].idxmax()  # most-rated match wins


def recommend(query, n=10):
    movie_id = find_movie(query)
    if movie_id is None:
        print(f'No movie matching "{query}" with at least {MIN_RATINGS} ratings.')
        return None

    row = movie_to_row[movie_id]
    scores = cosine_similarity(table[row], table).ravel()

    result = pd.DataFrame({"movieId": row_to_movie, "similarity": scores})
    result = result[result.movieId != movie_id].nlargest(n, "similarity")
    result = result.join(movies, on="movieId")

    print(f"People who loved {movies.title[movie_id]} also loved:\n")
    print(result[["title", "similarity", "genres"]].round(2).to_string(index=False))
    return result


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "Toy Story"
    recommend(query)
