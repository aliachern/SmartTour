import streamlit as st
import pandas as pd
import os
import json
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === File paths
DATASET_PATH = "/Users/aliachern/Documents/FYP/SmartTourMalaysia_Attractions.xlsx"
RATINGS_PATH = "ratings.csv"
USERS_PATH = "users.json"

# === Load Dataset ===
df = pd.read_excel(DATASET_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df['combined_features'] = df['category'].fillna('') + " " + df['state'].fillna('')

# === Load Ratings File ===
if os.path.exists(RATINGS_PATH):
    ratings_df = pd.read_csv(RATINGS_PATH)
else:
    ratings_df = pd.DataFrame(columns=['user_id', 'place_name', 'rating'])

# === Load Users File ===
if os.path.exists(USERS_PATH):
    with open(USERS_PATH, 'r') as f:
        USERS = json.load(f)
else:
    USERS = {"admin": "admin"}
    with open(USERS_PATH, 'w') as f:
        json.dump(USERS, f)

# === Train SVD Model ===
def train_svd(ratings_df):
    if ratings_df.empty:
        return None
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'place_name', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    return model

model_svd = train_svd(ratings_df)

# === TF-IDF Model ===
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# === Streamlit Interface Theming ===
st.markdown("""
    <style>
    .main { background-color: #f2f6fc; }
    .stApp {
        background-color: #ffffff;
        color: #0d1b2a;
    }
    .css-1d391kg { background-color: #1f3c88 !important; color: white !important; }
    .stButton > button {
        background-color: #ff4b5c;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# === Login / Signup System ===
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

auth_tab = st.sidebar.radio("🔐 Authentication", ["Login", "Sign Up"])

if not st.session_state.logged_in:
    if auth_tab == "Login":
        st.sidebar.title("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Logged in successfully!")
            else:
                st.error("❌ Invalid username or password")
        st.stop()
    else:
        st.sidebar.title("Sign Up")
        new_user = st.sidebar.text_input("Choose Username")
        new_pass = st.sidebar.text_input("Choose Password", type="password")
        if st.sidebar.button("Register"):
            if new_user in USERS:
                st.error("⚠️ Username already exists.")
            else:
                USERS[new_user] = new_pass
                with open(USERS_PATH, 'w') as f:
                    json.dump(USERS, f)
                st.success("✅ Registration successful! Please log in.")
        st.stop()

# === Main App UI ===
st.title("🌴 SmartTour Malaysia: Personalized Travel Recommender")

tabs = st.tabs(["🎯 Content-Based", "🧠 Collaborative (SVD)", "📝 Submit Rating"])

# --- Tab 1: Content-Based ---
with tabs[0]:
    st.subheader("🎯 Content-Based Recommendation")
    cat = st.selectbox("Choose a category", df['category'].dropna().unique())
    state = st.selectbox("Choose a state", df['state'].dropna().unique())
    if st.button("Get Recommendations"):
        user_input = cat + " " + state
        user_vec = vectorizer.transform([user_input])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        top_indices = sims.argsort()[-5:][::-1]
        result = df.iloc[top_indices][['place_name', 'category', 'state', 'avg_rating']]
        st.write(result.reset_index(drop=True))

# --- Tab 2: Collaborative Filtering (SVD) ---
with tabs[1]:
    st.subheader("🧠 Collaborative Filtering Recommendations")
    if model_svd:
        rated = ratings_df[ratings_df['user_id'] == st.session_state.username]['place_name'].tolist()
        all_places = df['place_name'].unique()
        predictions = [model_svd.predict(st.session_state.username, place) for place in all_places if place not in rated]
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_places = [pred.iid for pred in predictions[:5]]
        st.write(df[df['place_name'].isin(top_places)][['place_name', 'category', 'state', 'avg_rating']])
    else:
        st.warning("❗ Not enough ratings to train model. Add more first.")

# --- Tab 3: Submit Rating ---
with tabs[2]:
    st.subheader("📝 Rate an Attraction")
    place = st.selectbox("Select a place", df['place_name'].unique())
    rating = st.slider("Your rating (1–5)", 1, 5, 3)
    if st.button("Submit Rating"):
        new_entry = pd.DataFrame([{
            'user_id': st.session_state.username,
            'place_name': place,
            'rating': rating
        }])
        ratings_df = pd.concat([ratings_df, new_entry], ignore_index=True)
        ratings_df.to_csv(RATINGS_PATH, index=False)
        model_svd = train_svd(ratings_df)
        st.success(f"⭐ You rated {place} with {rating} stars!")

