# app.py -- AI & ML Analyzer for ThingSpeak and uploaded files
# Author: Anbu Sivam B | Version 1.0

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="AI & ML ThingSpeak Analyzer", layout="wide")

# ------------------------ Sidebar ------------------------
st.sidebar.title("🔑 Data Input Options")

mode = st.sidebar.radio("Select Data Source:", ["ThingSpeak API", "Upload File"])

# ------------------------ Fetch from ThingSpeak ------------------------
def fetch_thingspeak_data(api_key, channel_id):
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={api_key}&results=100"
    response = requests.get(url)
    if response.status_code == 200:
        feeds = response.json().get('feeds', [])
        df = pd.DataFrame(feeds)
        df = df.drop(columns=['entry_id'], errors='ignore')
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.set_index('created_at')
        df = df.apply(pd.to_numeric, errors='ignore')
        return df
    else:
        st.error("⚠️ Failed to fetch data. Check your API Key or Channel ID.")
        return None

# ------------------------ File Upload ------------------------
def load_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            return pd.read_csv(uploaded_file, sep='\t')
        else:
            st.error("Unsupported file format! Use CSV, XLSX, or TXT.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# ------------------------ Main Logic ------------------------
st.title("🤖 AI & ML Data Analyzer Dashboard")
st.markdown("Analyze **ThingSpeak IoT Data** or **Uploaded Sensor Files** with Machine Learning insights.")

df = None

if mode == "ThingSpeak API":
    api_key = st.sidebar.text_input("Enter ThingSpeak Read API Key:")
    channel_id = st.sidebar.text_input("Enter ThingSpeak Channel ID:")
    if st.sidebar.button("Fetch Data"):
        df = fetch_thingspeak_data(api_key, channel_id)
elif mode == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV/XLSX/TXT file", type=['csv', 'xlsx', 'txt'])
    if uploaded_file is not None:
        df = load_uploaded_file(uploaded_file)

# ------------------------ Data Display ------------------------
if df is not None and not df.empty:
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    st.subheader("📈 Data Summary")
    st.write(df.describe())

    # ------------------------ Visualization ------------------------
    st.subheader("📉 Sensor Data Trends")
    for col in df.columns:
        if col.startswith("field"):
            st.line_chart(df[col])

    # ------------------------ Correlation ------------------------
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        st.subheader("📊 Correlation Heatmap")
        plt.figure(figsize=(8,6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt.gcf())

    # ------------------------ ML: Clustering ------------------------
    st.subheader("🧠 ML Analysis: K-Means Clustering")
    try:
        scaled_data = StandardScaler().fit_transform(numeric_df.dropna())
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        df['Cluster'] = clusters
        st.write(df[['Cluster']].head())

        st.markdown("**Cluster Distribution:**")
        st.bar_chart(df['Cluster'].value_counts())

    except Exception as e:
        st.warning(f"K-Means skipped: {e}")

    # ------------------------ ML: Anomaly Detection ------------------------
    st.subheader("🚨 Anomaly Detection (Isolation Forest)")
    try:
        iso = IsolationForest(contamination=0.1, random_state=42)
        preds = iso.fit_predict(numeric_df.fillna(0))
        df['Anomaly'] = np.where(preds == -1, 1, 0)
        st.write(df[['Anomaly']].head())

        st.markdown("**Detected Anomalies:**")
        anomaly_count = df['Anomaly'].sum()
        st.metric("⚠️ Total Anomalies", anomaly_count)
        st.line_chart(df['Anomaly'])
    except Exception as e:
        st.warning(f"Anomaly Detection skipped: {e}")

else:
    st.info("👈 Select a data source to begin analysis.")
