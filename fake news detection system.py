import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from newspaper import Article

# --- CLEANING ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join(text.split())
    return text

# --- DATA LOAD ---
@st.cache_data
def load_data():
    fake_df = pd.read_csv("C:/Users/SAMSUNG/OneDrive/Documents/Information Retrieval Labs/IR Project/Fake.csv")
    true_df = pd.read_csv("C:/Users/SAMSUNG/OneDrive/Documents/Information Retrieval Labs/IR Project/True.csv")
    fake_df['label'] = 'FAKE'
    true_df['label'] = 'TRUE'
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df['text'] = df['text'].apply(clean_text)
    df.dropna(subset=['text'], inplace=True)
    return df

# --- VECTORIZE & MODEL ---
from sklearn.metrics import accuracy_score

def prepare_model(df):
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vect, y_train)

    y_pred = model.predict(X_test_vect)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")  # This will print in terminal
    return model, vectorizer, X_train_vect, y_train, df, accuracy


def compute_centroids(X_train_vect, y_train):
    true_centroid = np.squeeze(np.asarray(X_train_vect[(y_train == 'TRUE').values].mean(axis=0)))
    fake_centroid = np.squeeze(np.asarray(X_train_vect[(y_train == 'FAKE').values].mean(axis=0)))
    return true_centroid, fake_centroid

def predict_article(text, model, vectorizer):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    return model.predict(vect)[0]

def predict_rochio(text, vectorizer, true_centroid, fake_centroid):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    sim_true = cosine_similarity(vect, true_centroid.reshape(1, -1))[0][0]
    sim_fake = cosine_similarity(vect, fake_centroid.reshape(1, -1))[0][0]
    return "TRUE" if sim_true > sim_fake else "FAKE"

def fetch_article_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error: {e}"

#STREAMLIT FRONTEND
st.title("📰 Fake News Detection System")

df = load_data()
model, vectorizer, X_train_vect, y_train, df, accuracy = prepare_model(df)
st.sidebar.markdown(f"📊 **Model Accuracy:** `{accuracy:.2%}`")

true_centroid, fake_centroid = compute_centroids(X_train_vect, y_train)

option = st.sidebar.radio("Choose an action:", ["Classify Text", "Classify from URL", "Search News"])

if option == "Classify Text":
    user_text = st.text_area("Paste a news article here:", height=200)
    if st.button("Classify"):
        if user_text.strip():
            pred = predict_article(user_text, model, vectorizer)
            rocchio = predict_rochio(user_text, vectorizer, true_centroid, fake_centroid)
            st.success(f"🔍 Model Prediction: **{pred}**")
            st.info(f"📐 Rocchio Prediction: **{rocchio}**")
        else:
            st.warning("Please enter some article text.")

elif option == "Classify from URL":
    url = st.text_input("Enter the article URL:")
    if st.button("Fetch and Classify"):
        article_text = fetch_article_from_url(url)
        if "Error:" not in article_text:
            st.text_area("Extracted Article Text:", article_text, height=150)
            pred = predict_article(article_text, model, vectorizer)
            rocchio = predict_rochio(article_text, vectorizer, true_centroid, fake_centroid)
            st.success(f"🔍 Model Prediction: **{pred}**")
            st.info(f"📐 Rocchio Prediction: **{rocchio}**")
        else:
            st.error(article_text)

elif option == "Search News":
    keyword = st.text_input("Enter a keyword to search in dataset:")
    if st.button("Search"):
        keyword = keyword.lower()
        results = df[df['text'].str.contains(keyword)]
        if results.empty:
            st.warning("No articles found with that keyword.")
        else:
            st.write(f"🔎 Found {len(results)} articles:")
            for _, row in results.head().iterrows():
                st.markdown(f"**[{row['label']}]** {row.get('title', 'No Title')}")
                st.text(row['text'][:300] + '...')

