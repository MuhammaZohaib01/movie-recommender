import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")
movies['combined'] = movies['title'] + " " + movies['genres']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_movie(title):
    try:
        idx = movies[movies['title'].str.contains(title, case=False)].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        movie_indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[movie_indices]
    except:
        return ["Movie not found. Please try a different name."]

st.title("🎬 Movie Recommendation System")
st.write("Enter the name of a movie you like, and we'll recommend similar ones.")

movie_input = st.text_input("Enter a movie title:")

if st.button("Get Recommendations"):
    results = recommend_movie(movie_input)
    st.subheader("Recommended Movies:")
    for movie in results:
        st.write("➡️", movie)
