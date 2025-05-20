import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="🕵️‍♂️ Fraud Detection in Internship Applications", layout="wide")

@st.cache_data
def load_data():
    # Replace 'data.csv' with your dataset path
    df = pd.read_csv("data.csv", parse_dates=["submission_time"])
    return df

def detect_anomalies(df):
    # Select relevant numeric features for anomaly detection
    feature_cols = ['cgpa', 'resume_quality_score', 'previous_applications']  # Example features
    features = df[feature_cols].fillna(0)
    
    # Isolation Forest
    iforest = IsolationForest(contamination=0.05, random_state=42)
    df['iforest_anomaly'] = iforest.fit_predict(features)
    df['iforest_anomaly'] = df['iforest_anomaly'].map({1:0, -1:1})  # 1 = anomaly
    
    # K-Means Clustering (example)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['kmeans_cluster'] = kmeans.fit_predict(features)
    
    return df, features, df['iforest_anomaly'].values, kmeans

def plot_pca(features, anomaly_labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=anomaly_labels, palette=["blue", "red"])
    plt.title("PCA Visualization: Blue=Normal, Red=Anomaly")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    st.pyplot(plt)

# Main app

st.title("🕵️‍♂️ Fraud Detection in Internship Applications")
st.write("This interactive dashboard detects anomalies in internship application data using Isolation Forest and K-Means Clustering.")

df = load_data()

st.header("📄 Raw Dataset")
st.dataframe(df)

df, features, anomaly_labels, kmeans_model = detect_anomalies(df)

# Summary
st.header("🚨 Anomaly Detection Summary")
st.write(f"Total Applications: {len(df)}")
st.write(f"Detected Anomalies (Isolation Forest): {anomaly_labels.sum()}")
st.write(f"Clusters found by K-Means: {kmeans_model.n_clusters}")

# PCA plot
st.header("📉 PCA Visualization of Anomalies")
plot_pca(features, anomaly_labels)

# Show suspicious filtered data
st.header("Filtered Suspicious Applications")
suspicious_df = df[df['iforest_anomaly'] == 1]
st.dataframe(suspicious_df)

# Download suspicious applications as CSV
st.header("📥 Download Alerts")
csv = suspicious_df.to_csv(index=False).encode('utf-8')
st.download_button(label="Download Suspicious Applications CSV", data=csv, file_name='suspicious_applications.csv', mime='text/csv')
