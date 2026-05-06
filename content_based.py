import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

BASE_DIR = os.path.dirname(__file__)

def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "Data", "movies_metadata.csv"), low_memory=False)

    # Keep needed columns
    df = df[['title', 'overview']].dropna()

    # 🔥 IMPORTANT FIX: remove sampling
    # df = df.sample(5000, random_state=42)

    # Clean titles
    df['title'] = df['title'].str.lower().str.strip()

    return df.reset_index(drop=True)


def create_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])

    # Use cleaned titles
    indices = pd.Series(df.index, index=df['title'])

    return tfidf_matrix, indices


from difflib import get_close_matches

def recommend(movie, df, tfidf_matrix, indices):
    movie_clean = movie.lower().strip()

    # 🔥 Try exact match first
    if movie_clean in indices:
        idx = indices[movie_clean]

    else:
        # 🔥 Try partial match (VERY IMPORTANT FIX)
        matches = df[df['title'].str.contains(movie_clean, case=False, na=False)]

        if not matches.empty:
            idx = matches.index[0]   # take first match

        else:
            # 🔥 fallback suggestion
            suggestions = get_close_matches(movie_clean, indices.index, n=3, cutoff=0.5)
            
            if suggestions:
                return [f"Movie not found. Did you mean: {', '.join(suggestions)} ?"]
            else:
                return ["Movie not found"]

    # ✅ similarity logic
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    movie_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[movie_indices].tolist()