# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import hashlib

# Load data
df = pd.read_excel("SmartTourMalaysia_Attractions.xlsx")
df['combined_features'] = df['state'] + " " + df['category'] + " " + df['description']

# Vectorize and compute similarities
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Dummy user database
users = {}

# Tabs
st.set_page_config(page_title="SmartTour Malaysia", layout="wide")
tabs = st.tabs(["Home", "Recommendation", "Rate Destinations"])

# User auth functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup(username, password):
    if username in users:
        return False
    users[username] = hash_password(password)
    return True

def login(username, password):
    return users.get(username) == hash_password(password)

# Home Tab
with tabs[0]:
    st.title("SmartTour Malaysia")
    st.write("Welcome to your personalized AI travel guide.")

    st.subheader("Log In")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Log In"):
        if login(username, password):
            st.success("Logged in successfully!")
        else:
            st.error("Invalid credentials.")

    st.subheader("Sign Up")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type='password')
    if st.button("Sign Up"):
        if signup(new_username, new_password):
            st.success("Account created!")
        else:
            st.error("Username already exists.")

# Recommendation Tab
with tabs[1]:
    st.header("Find Your Destination")
    selected_state = st.selectbox("Choose a state", df['state'].unique())
    selected_category = st.selectbox("Choose a category", df['category'].unique())

    query = selected_state + " " + selected_category
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:5]

    st.subheader("Recommended Destinations")
    for i in top_indices:
        st.markdown(f"**{df.iloc[i]['name']}**")
        st.write(f"State: {df.iloc[i]['state']} | Category: {df.iloc[i]['category']}")
        st.write(f"Description: {df.iloc[i]['description']}")
        st.write(f"Rating: {df.iloc[i]['avg_rating']}")
        st.markdown("---")

# Rating Tab
with tabs[2]:
    st.header("Rate Destinations")
    selected_dest = st.selectbox("Choose a destination", df['name'])
    rating = st.slider("Your Rating", 1, 5, 3)
    if st.button("Submit Rating"):
        st.success(f"Thank you for rating {selected_dest} with {rating} stars!")
