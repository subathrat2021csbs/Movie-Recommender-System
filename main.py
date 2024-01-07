import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests

# Assuming you read your dataset properly
movies = pd.read_csv("movie.csv")

# Fill missing values in 'overview' column
movies['overview'] = movies['overview'].fillna('')

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['overview'])

# Similarity matrix using linear kernel
similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to fetch movie poster
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data.get('poster_path', '')
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

st.header("Movie Recommender System")

# Movie selection dropdown
select_value = st.selectbox("Select movie from dropdown", movies['title'].values)

# Function to recommend movies
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
    recommend_movie = []
    recommend_poster = []
    for i in distance[1:6]:
        movies_id = movies.iloc[i[0]].id
        recommend_movie.append(movies.iloc[i[0]].title)
        recommend_poster.append(fetch_poster(movies_id))
    return recommend_movie, recommend_poster

# Show recommendations on button click
if st.button("Show Recommendations"):
    movie_name, movie_poster = recommend(select_value)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(movie_name[0])
        st.image(movie_poster[0], width=100)
    with col2:
        st.text(movie_name[1])
        st.image(movie_poster[1], width=100)
    with col3:
        st.text(movie_name[2])
        st.image(movie_poster[2], width=100)
    with col4:
        st.text(movie_name[3])
        st.image(movie_poster[3], width=100)
    with col5:
        st.text(movie_name[4])
        st.image(movie_poster[4], width=100)

# Additional Features
st.header("Additional Features")

# Trending Movies
trending_movies = st.checkbox("Show Trending Movies")
if trending_movies:
    # You can fetch trending movies from an external API or use your own logic
    trending_movies_list = ["Trending Movie 1", "Trending Movie 2", "Trending Movie 3"]
    st.write("Trending Movies:")
    for trend_movie in trending_movies_list:
        st.text(trend_movie)

# Upcoming Releases
upcoming_releases = st.checkbox("Show Upcoming Releases")
if upcoming_releases:
    # You can fetch upcoming releases from an external API or use your own logic
    upcoming_releases_list = ["Upcoming Movie 1", "Upcoming Movie 2", "Upcoming Movie 3"]
    st.write("Upcoming Releases:")
    for upcoming_movie in upcoming_releases_list:
        st.text(upcoming_movie)

# User Feedback/Reviews
user_reviews = st.checkbox("Show User Feedback/Reviews")
if user_reviews:
    # You can implement a feedback/review system and display user feedback here
    user_feedback_list = ["User Feedback 1", "User Feedback 2", "User Feedback 3"]
    st.write("User Feedback/Reviews:")
    for user_feedback in user_feedback_list:
        st.text(user_feedback)
