# vagabond_combined_app.py

import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1️⃣ Load Dataset and Precomputed Similarity
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data set of famous India tourist places along with there images.csv")
    # Create features column if not already present
    if "features" not in df.columns:
        df["features"] = df["Type"] + " " + df["Significance"] + " " + df["State"] + " " + df["City"] + " " + df["Best Time to visit"]
    return df

@st.cache_data
def load_similarity():
    # Load precomputed similarity matrix for existing places
    with open("cosine_sim.pkl", "rb") as f:
        cosine_sim = pickle.load(f)
    return cosine_sim

df = load_data()
cosine_sim = load_similarity()

# -------------------------------
# 2️⃣ TF-IDF Vectorizer
# -------------------------------
# Fit TF-IDF directly (no caching)
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["features"])

# -------------------------------
# 3️⃣ Recommendation Functions
# -------------------------------
def recommend_place(place_name, top_n=5):
    """Existing place selection"""
    if place_name not in df["Name"].values:
        return pd.DataFrame()
    idx = df[df["Name"] == place_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    indices = [i[0] for i in sim_scores]
    return df.iloc[indices][["Name", "City", "State", "Best Time to visit", "Image URL"]]

def recommend_from_input(user_input, top_n=5):
    """User input-driven recommendations"""
    input_vec = vectorizer.transform([user_input])
    cosine_sim_input = cosine_similarity(input_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim_input.argsort()[::-1][:top_n]
    return df.iloc[top_indices][["Name", "City", "State", "Best Time to visit", "Image URL"]]

def display_recommendations(recommendations):
    if recommendations.empty:
        st.write("No recommendations found.")
    else:
        for i, row in recommendations.iterrows():
            st.subheader(row["Name"])
            st.write(f"{row['City']}, {row['State']} — Best Time: {row['Best Time to visit']}")
            if pd.notna(row["Image URL"]):
                st.image(row["Image URL"], width=400)

# -------------------------------
# 4️⃣ Streamlit UI
# -------------------------------
st.title("🌏 Vagabond Travel Recommendations")
st.write("Choose how you want recommendations:")

option = st.radio(
    "Recommendation Mode",
    ("Select Existing Place", "Type Features")
)

if option == "Select Existing Place":
    place = st.selectbox("Choose a place:", df["Name"].values)
    if st.button("Show Recommendations"):
        recommendations = recommend_place(place, top_n=5)
        display_recommendations(recommendations)

elif option == "Type Features":
    user_input = st.text_input("Enter place features (e.g., 'Temple Religious Morning'):")
    if st.button("Get Recommendations") and user_input.strip() != "":
        recommendations = recommend_from_input(user_input, top_n=5)
        display_recommendations(recommendations)
