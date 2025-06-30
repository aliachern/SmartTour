import streamlit as st
import pandas as pd
from model.recommender import preprocess_data, build_cosine_model, build_svd_model
from utils.auth import create_user_table, add_user, login_user

st.set_page_config(page_title="SmartTour Malaysia", layout="wide")
st.title("SmartTour Malaysia 🇲🇾")

create_user_table()

menu = ["Login", "Sign Up"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Sign Up":
    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        add_user(new_user, new_password)
        st.success("Account created successfully!")

elif choice == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(username, password):
            st.success(f"Welcome {username}")
            tab1, tab2, tab3 = st.tabs(["🏠 Home", "🎯 Recommendations", "⭐ Ratings"])

            with tab1:
                st.markdown("Welcome to SmartTour Malaysia! Select your preferences and get tailored travel suggestions.")

            with tab2:
                st.markdown("### Get Travel Recommendations")
                state = st.selectbox("Select State", ['kuala lumpur', 'kedah', 'melaka'])
                category = st.selectbox("Select Category", ['beach', 'historical', 'nature', 'adventure', 'city'])

                df = preprocess_data("data/SmartTourMalaysia_Attractions.xlsx")
                filtered = df[(df['state'] == state) & (df['category'] == category)]

                if filtered.empty:
                    st.warning("No results found. Try different filters.")
                else:
                    st.dataframe(filtered[['place_name', 'category', 'avg_rating']].sort_values(by='avg_rating', ascending=False))

            with tab3:
                st.markdown("Rate your visited destinations (future enhancement)")
        else:
            st.error("Incorrect Username or Password")

