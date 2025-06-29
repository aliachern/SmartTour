
import streamlit as st
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# === Configs ===
DATASET_PATH = "SmartTourMalaysia_Cleaned_Latest.xlsx"
RATINGS_PATH = "ratings.csv"
USERS_PATH = "users.json"

# === Load data ===
df = pd.read_excel(DATASET_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df['combined_features'] = df['category'].fillna('') + " " + df['state'].fillna('')

if os.path.exists(RATINGS_PATH):
    ratings_df = pd.read_csv(RATINGS_PATH)
else:
    ratings_df = pd.DataFrame(columns=['user_id', 'place_name', 'rating'])

if os.path.exists(USERS_PATH):
    with open(USERS_PATH, 'r') as f:
        USERS = json.load(f)
else:
    USERS = {"admin": "admin"}
    with open(USERS_PATH, 'w') as f:
        json.dump(USERS, f)

# === TF-IDF Vectorizer ===
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# === Build user-item matrix for collaborative filtering ===
def build_similarity(ratings_df):
    user_item = ratings_df.pivot_table(index='user_id', columns='place_name', values='rating').fillna(0)
    return pd.DataFrame(cosine_similarity(user_item), index=user_item.index, columns=user_item.index), user_item

similarity_matrix, user_item_matrix = build_similarity(ratings_df)

# === Collaborative filtering recommender ===
def recommend_places_cf(user_id, df_places, ratings_df, top_n=5):
    if user_id not in similarity_matrix.index:
        return pd.DataFrame(columns=['name', 'category', 'state', 'avg_rating'])

    similar_users = similarity_matrix[user_id].drop(user_id).sort_values(ascending=False)
    unrated_places = df_places[~df_places['name'].isin(ratings_df[ratings_df['user_id'] == user_id]['place_name'])]

    scores = {}
    for place in unrated_places['name'].unique():
        total_score, total_sim = 0, 0
        for sim_user in similar_users.index:
            sim_rating = ratings_df[(ratings_df['user_id'] == sim_user) & (ratings_df['place_name'] == place)]
            if not sim_rating.empty:
                score = sim_rating['rating'].values[0]
                sim = similar_users[sim_user]
                total_score += score * sim
                total_sim += sim
        if total_sim > 0:
            scores[place] = total_score / total_sim

    top_places = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_names = [x[0] for x in top_places]
    return df_places[df_places['name'].isin(top_names)][['name', 'category', 'state', 'avg_rating']]

# === Login system ===
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

# === Main UI ===
st.title("🌴 SmartTour Malaysia")

tabs = st.tabs(["🎯 Content-Based", "🤝 Collaborative Filtering", "📝 Submit Rating"])

with tabs[0]:
    st.subheader("🎯 Content-Based Recommendations")
    cat = st.selectbox("Choose a category", df['category'].dropna().unique())
    state = st.selectbox("Choose a state", df['state'].dropna().unique())
    if st.button("Get Recommendations"):
        user_input = cat + " " + state
        user_vec = vectorizer.transform([user_input])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        top_indices = sims.argsort()[-5:][::-1]
        result = df.iloc[top_indices][['name', 'category', 'state', 'avg_rating']]
        st.write(result.reset_index(drop=True))

with tabs[1]:
    st.subheader("🤝 Collaborative Filtering")
    if not ratings_df.empty:
        recs = recommend_places_cf(st.session_state.username, df, ratings_df)
        st.write(recs.reset_index(drop=True))
    else:
        st.info("Please add some ratings first.")

with tabs[2]:
    st.subheader("📝 Submit Rating")
    place = st.selectbox("Select a place", df['name'].unique())
    rating = st.slider("Your rating", 1, 5, 3)
    if st.button("Submit Rating"):
        new_entry = pd.DataFrame([{
            'user_id': st.session_state.username,
            'place_name': place,
            'rating': rating
        }])
        ratings_df = pd.concat([ratings_df, new_entry], ignore_index=True)
        ratings_df.to_csv(RATINGS_PATH, index=False)
        similarity_matrix, user_item_matrix = build_similarity(ratings_df)
        st.success(f"⭐ You rated {place} with {rating} stars!")
