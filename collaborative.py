import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

def recommend_movies(user_id, n=5):
    BASE_DIR = os.path.dirname(__file__)

    ratings = pd.read_csv(os.path.join(BASE_DIR, "Data", "ratings.csv")).head(100000)
    movies = pd.read_csv(os.path.join(BASE_DIR, "Data", "movies.csv"))

    df = ratings.merge(movies, on="movieId")

    user_movie_matrix = df.pivot_table(
        index='userId',
        columns='title',
        values='rating'
    ).fillna(0)

    user_similarity = cosine_similarity(user_movie_matrix)

    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )

    # FIX 1: cast to int — Streamlit number_input returns float
    user_id = int(user_id)

    if user_id not in user_movie_matrix.index:
        return [f"User ID {user_id} not found in dataset"]

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]

    similar_users_ratings = user_movie_matrix.loc[similar_users.index]

    mean_ratings = similar_users_ratings.mean().sort_values(ascending=False)

    # FIX 2: exclude movies the user has already rated
    already_watched = user_movie_matrix.loc[user_id]
    already_watched = already_watched[already_watched > 0].index
    mean_ratings = mean_ratings.drop(labels=already_watched, errors='ignore')

    return mean_ratings.head(n).index.tolist()