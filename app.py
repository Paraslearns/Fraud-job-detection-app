

    
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load your trained model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

st.title("🚩 Fraud Job Postings Detector")

# 📁 File upload section
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# 👉 Run this only when user uploads a file
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Combine text features (same as training)
    df["text"] = (
        df["title"].fillna("") + " " +
        df["location"].fillna("") + " " +
        df["department"].fillna("") + " " +
        df["company_profile"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["requirements"].fillna("") + " " +
        df["benefits"].fillna("")
    )

    # Transform and predict
    X = vectorizer.transform(df["text"])
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    df["fraudulent_pred"] = pred
    df["fraud_probability"] = prob

    # 📊 Show predictions
    st.subheader("Prediction Results")
    st.dataframe(df[["title", "location", "fraudulent_pred", "fraud_probability"]])

    # 📈 Show histogram
    st.subheader("Histogram of Fraud Probabilities")
    st.bar_chart(np.histogram(prob, bins=10)[0])

    # 🥧 Pie chart
    st.subheader("Pie Chart: Genuine vs Fraudulent")
    counts = df["fraudulent_pred"].value_counts()
    st.write(counts)
    fig, ax = plt.subplots()
    ax.pie(counts, labels=["Genuine", "Fraudulent"], autopct="%1.1f%%")
    st.pyplot(fig)

    # 🔍 Top 10 most suspicious
    st.subheader("Top 10 Most Suspicious Jobs")
    top10 = df.sort_values(by="fraud_probability", ascending=False).head(10)
    st.table(top10[["title", "location", "fraud_probability"]])
