import streamlit as st
import pickle
import requests



cosine_similarity = pickle.load(open("cosine_similarity.pkl","rb"))
new_df = pickle.load(open("data.pkl","rb"))
encoder = pickle.load(open("encoder.pkl","rb"))
kmeans = pickle.load(open("kmeans.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

def get_genres_for_movie(movie_id):
    row = new_df[new_df['Movie_id'] == movie_id]
    if not row.empty:
        genres = row[['Genre 1', 'Genre 2', 'Genre 3',"Original_language"]].values[0].tolist()
        return genres
    else:
        return None  # Return None if the movie ID is not found

def find_similar_movies(new_data_point):
    new_data_encoded = encoder.transform([new_data_point])
    cluster_label = kmeans.predict(new_data_encoded)[0]
    similar_movies = new_df[new_df['Cluster'] == cluster_label]
    return similar_movies  # Returns a subset of Data set where the Cluster is same as data point


TMDB_API_KEY = 'b2b3610b2d734655e8f50a74599f5a62'  # Replace with your TMDB API key
BASE_URL = 'https://api.themoviedb.org/3'
POSTER_BASE_URL = 'https://image.tmdb.org/t/p/w500'

def fetch_and_show_movie_poster(movie_id,movie_name,col):
    url = f"{BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        movie_data = response.json()
        poster_path = movie_data.get('poster_path')
        
        if poster_path:
            poster_url = f"{POSTER_BASE_URL}/{poster_path}"
            col.image(poster_url, caption=movie_name, width=100)
        else:
            print("No poster available for this movie.")
    else:
        print("Error fetching movie data.")





def recomed(movie_name):
    movie_id = new_df[new_df["Movie_key"] == movie_name]["Movie_id"].values[0]
    print(movie_id)
    data = find_similar_movies(get_genres_for_movie(movie_id))
    
    movie = vectorizer.transform(new_df[new_df["Movie_id"] == movie_id]["tags"]).toarray()
    matrix = []
    for index, value in data["tags"].items():
        others = vectorizer.transform([value]).toarray()
        cosine = cosine_similarity(movie,others)
        matrix.append((cosine,index))
    sorted_list = sorted(matrix, key=lambda x: x[0][0][0], reverse=True)
    cols = st.columns(6)
    for i in range(1,7):
        fetch_and_show_movie_poster(new_df["Movie_id"].iloc[sorted_list[i][1]],new_df["Movie_name"].iloc[sorted_list[i][1]],cols[i - 1])

st.title("Welcome to Movie Recomendation system")

selected_movies = st.selectbox("Search Your Movie",new_df["Movie_key"].values)

if st.button("Recomend"):
    recomed(selected_movies)