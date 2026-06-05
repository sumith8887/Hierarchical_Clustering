import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Page Config
st.set_page_config(page_title="Hierarchical Clustering", layout="wide")

st.title("🛍️ Mall Customer Segmentation using Hierarchical Clustering")

# Load Dataset
df = pd.read_csv("data/Mall_Customers.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Features for Clustering
features = ["Annual Income (k$)", "Spending Score (1-100)"]

X = df[features]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Dendrogram
# ---------------------------
st.subheader("Dendrogram")

linked = linkage(X_scaled, method='ward')

fig, ax = plt.subplots(figsize=(12, 6))
dendrogram(linked, ax=ax)
ax.set_title("Dendrogram")
ax.set_xlabel("Customers")
ax.set_ylabel("Euclidean Distance")

st.pyplot(fig)

# ---------------------------
# Select Clusters
# ---------------------------
n_clusters = st.slider(
    "Select Number of Clusters",
    min_value=2,
    max_value=10,
    value=5
)

# Hierarchical Clustering Model
hc = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage='ward'
)

clusters = hc.fit_predict(X_scaled)

df["Cluster"] = clusters

# ---------------------------
# Clustered Dataset
# ---------------------------
st.subheader("Clustered Dataset")
st.dataframe(df.head())

# ---------------------------
# Cluster Distribution
# ---------------------------
st.subheader("Cluster Distribution")
st.write(df["Cluster"].value_counts().sort_index())

# ---------------------------
# Visualization
# ---------------------------
st.subheader("Cluster Visualization")

fig2, ax2 = plt.subplots(figsize=(8, 6))

scatter = ax2.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=80
)

ax2.set_xlabel("Annual Income (k$)")
ax2.set_ylabel("Spending Score (1-100)")
ax2.set_title("Hierarchical Clustering Results")

plt.colorbar(scatter)

st.pyplot(fig2)

# ---------------------------
# Cluster Statistics
# ---------------------------
st.subheader("Cluster Statistics")

cluster_stats = df.groupby("Cluster")[features].mean()

st.dataframe(cluster_stats)

st.success("Hierarchical Clustering Completed Successfully!")