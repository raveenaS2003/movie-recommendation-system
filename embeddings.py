import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

def semantic_recommend(title, n=5):
    BASE_DIR = os.path.dirname(__file__)

    movies = pd.read_csv(os.path.join(BASE_DIR, "Data", "movies.csv"))

    movies['genres'] = movies['genres'].fillna("")
    movies['text'] = movies['title'] + " " + movies['genres']

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # ✅ FIX HERE
    embeddings = model.encode(movies['text'].tolist(), show_progress_bar=False)

    if title not in movies['title'].values:
        return ["Movie not found"]

    idx = movies[movies['title'] == title].index[0]

    sim_scores = cosine_similarity([embeddings[idx]], embeddings)[0]

    similar_indices = sim_scores.argsort()[-(n+1):-1][::-1]

    return movies.iloc[similar_indices]['title'].tolist()