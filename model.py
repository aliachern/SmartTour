import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

def preprocess_data(filepath):
    df = pd.read_excel(filepath)
    df = df.dropna(subset=['avg_rating', 'category', 'state'])
    df['category'] = df['category'].str.strip().str.lower()
    df['state'] = df['state'].str.strip().str.lower()
    return df

def build_cosine_model(df):
    pivot = df.pivot_table(index='user_id', columns='place_name', values='avg_rating').fillna(0)
    similarity = cosine_similarity(pivot)
    return similarity, pivot

def build_svd_model(df, n_components=10):
    pivot = df.pivot_table(index='user_id', columns='place_name', values='avg_rating').fillna(0)
    svd = TruncatedSVD(n_components=n_components)
    matrix = svd.fit_transform(pivot)
    return matrix, pivot.columns, svd
