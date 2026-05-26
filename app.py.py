# import required libraries
import pandas as pd
import streamlit as st
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Downlaod stopwards
nltk.download("stopwords")

# Stremlit title
st.title("Movie Recommendation System")

# load dataset
data = pd.read_csv(r"D:\TuteDude\Assignment20\tmdb_5000_movies.csv")
print(data.head())

# Select Required columns
data = data[["title","overview"]]

# Handle Missing Values
data["overview"] = data["overview"].fillna("")

# Stopwords
stop_words = set(stopwords.words("english"))

# Cleaning function
def clean_text(text):
    
    # Convert to lowercase
    text = text.lower()

    # Remove Special Characters
    text = re.sub(r"[^a-zA-Z0-9 ]", " ",text)

    # Split sentence into words
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Join Words
    return " ".join(words)

# Apply preprocessing
data["clean_text"] = data["overview"].apply(clean_text)

# IF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

# Convert text into numerical vectors
tfidf_matrix = vectorizer.fit_transform(data["clean_text"])

# Similar Matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Recommendation function 
def recommend(movie_name,top_n=5):

    # find selected movie index
    movie_index = data[data["title"]==movie_name].index[0]

    # Get Similarity Scores
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    # Sort movies based on similarity score
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x:x[1],
        reverse=True
    )

    # Remove selected movie itself
    similarity_scores = similarity_scores[1:top_n+1]

    # Store Recommended movie names
    recommend_data =[]

    for movie in similarity_scores:
        index = movie[0]
        recommend_data.append(
            data.iloc[index]["title"]
        )
    return recommend_data

# Dropdown
selected_movie = st.selectbox(
    "Select a Movie",
    data["title"].values
)

# button
if st.button("Recommend"):

    # Generate Recommendations
    recommendations = recommend(selected_movie)

    # Display output
    st.subheader("Recommended Movies")
    for movie in recommendations:
        st.write(movie)
