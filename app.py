import streamlit as st

st.title("🎬 Movie Recommendation System")

option = st.sidebar.selectbox(
    "Choose Method",
    ["Collaborative", "Content-Based", "Semantic"]
)

# ------------------ COLLABORATIVE ------------------
if option == "Collaborative":
    from collaborative import recommend_movies

    st.header("👥 Collaborative Filtering")

    # FIX: step=1 and int() cast — number_input returns float by default
    user_id = int(st.number_input("Enter User ID", min_value=1, step=1))

    if st.button("Recommend"):
        with st.spinner("Loading..."):
            try:
                recs = recommend_movies(user_id)
                if not recs:
                    st.warning("No recommendations found.")
                else:
                    for r in recs:
                        st.write("🎥", r)
            except Exception as e:
                st.error(f"Error: {e}")

# ------------------ CONTENT ------------------
elif option == "Content-Based":
    from content_based import load_data, create_model, recommend

    st.header("📄 Content-Based Filtering")

    # FIX: cache so model doesn't reload on every interaction
    @st.cache_data
    def get_content_model():
        df = load_data()
        tfidf_matrix, indices = create_model(df)
        return df, tfidf_matrix, indices

    movie = st.text_input("Enter Movie Title")

    if st.button("Recommend"):
        with st.spinner("Processing..."):
            try:
                df, tfidf_matrix, indices = get_content_model()
                recs = recommend(movie, df, tfidf_matrix, indices)
                if not recs or recs == ["Movie not found"]:
                    st.warning("Movie not found. Check the title spelling.")
                else:
                    for r in recs:
                        st.write("🎬", r)
            except Exception as e:
                st.error(f"Error: {e}")

# ------------------ SEMANTIC ------------------
elif option == "Semantic":
    from embeddings import semantic_recommend

    st.header("🧠 Semantic Recommendation")

    # FIX: cache so the sentence-transformer model doesn't reload every time
    @st.cache_data
    def get_semantic_recs(movie_title):
        return semantic_recommend(movie_title)

    movie = st.text_input("Enter Movie Title")

    if st.button("Recommend"):
        with st.spinner("Understanding meaning..."):
            try:
                recs = get_semantic_recs(movie)
                if not recs or recs == ["Movie not found"]:
                    st.warning("Movie not found. Check the title spelling.")
                else:
                    for r in recs:
                        st.write("✨", r)
            except Exception as e:
                st.error(f"Error: {e}")