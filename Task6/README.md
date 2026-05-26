This is a simple Movie Recommendation System made using Python and Streamlit.
The project recommends movies based on movie overview text using TF-IDF and Cosine Similarity.

Dataset Used

Dataset: TMDB 5000 Movies Dataset download from kaggle.com

CSV file used:
tmdb_5000_movies.csv

Libraries Used
pandas
streamlit
nltk
scikit-learn
re

Install required libraries using:

pip install pandas streamlit nltk scikit-learn
Project Working
Load movie dataset
Select only title and overview columns
Clean the text data
Remove stopwords
Convert text into vectors using TF-IDF
Calculate similarity using cosine similarity
Recommend similar movies
Features
Simple UI using Streamlit
Dropdown for selecting movies
Gives top recommended movies
Text preprocessing included
Uses machine learning concepts
How to Run
Open terminal and run:
streamlit run Task6.py