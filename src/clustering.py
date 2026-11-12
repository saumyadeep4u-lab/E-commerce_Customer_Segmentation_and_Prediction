
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("plots", exist_ok=True)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. SCALE FEATURES
def scale_features(rfm):
    features = rfm[['Recency', 'Frequency', 'Monetary', 'ReturnRatio']].copy()
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    return scaler.fit_transform(features)

# 2. ELBOW METHOD (KMeans)
def elbow_method(data):
    inertia = []
    k_range = range(1, 10)

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(data)
        inertia.append(model.inertia_)

    plt.plot(k_range, inertia, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.savefig("plots/elbow_method.png", dpi=300, bbox_inches='tight')
    plt.show()

# 3. KMEANS TUNING
def tune_kmeans(data, show_plot=True):
    best_score = -1
    best_k = 2
    best_labels = None

    scores = []
    k_values = []

    for k in range(2, 10):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)

        scores.append(score)
        k_values.append(k)

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    print(f"KMeans → Best K: {best_k}, Score: {best_score:.4f}")

    # K vs Silhouette graph
    if show_plot:
        plt.figure(figsize=(6, 4))
        plt.plot(k_values, scores, marker='o')
        plt.title("K vs Silhouette Score (KMeans)")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.savefig("plots/kmeans_silhouette.png", dpi=300, bbox_inches='tight')
        plt.show()

    return best_labels, best_score

#  4. DBSCAN TUNING
def tune_dbscan(data, show_plot=True):
    best_score = -1
    best_labels = None

    eps_values = []
    scores = []

    for eps in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        for min_samples in [3, 5, 10]:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(data)

            if len(set(labels)) <= 2 or -1 in labels:
                continue

            score = silhouette_score(data, labels)

            eps_values.append(eps)   
            scores.append(score) 

            if score > best_score:
                best_score = score
                best_labels = labels
                

    print(f"DBSCAN → Score: {best_score:.4f}")

    if show_plot and len(eps_values) > 0:
        plt.figure(figsize=(6, 4))
        plt.plot(eps_values, scores, marker='o')
        plt.title("DBSCAN: eps vs Silhouette Score")
        plt.xlabel("eps")
        plt.ylabel("Silhouette Score")
        plt.savefig("plots/dbscan_silhouette.png", dpi=300, bbox_inches='tight')
        plt.show()

    return best_labels, best_score

# 5. HIERARCHICAL TUNING
def tune_hierarchical(data, show_plot=True):
    best_score = -1
    best_labels = None

    scores = []
    k_values = []

    for k in range(2, 10):
        try:
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(data)

            
            if len(set(labels)) < 2:
                continue

            score = silhouette_score(data, labels)

            print(f"K={k}, Score={score:.4f}")  # DEBUG

            scores.append(score)
            k_values.append(k)

            if score > best_score:
                best_score = score
                best_labels = labels

        except Exception as e:
            print(f"Skipping k={k} due to error: {e}")
            continue

    print(f"Hierarchical → Best Score: {best_score:.4f}")

    
    if show_plot and len(k_values) > 0:
        plt.figure(figsize=(6, 4))
        plt.plot(k_values, scores, marker='o')
        plt.title("Hierarchical: K vs Silhouette Score")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.savefig("plots/hierarchical_silhouette.png", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print(" No valid clusters found for plotting")

    return best_labels, best_score

# 6. DENDROGRAM
def plot_dendrogram(data):
    linked = linkage(data, method='ward')
    dendrogram(linked)
    plt.title("Hierarchical Dendrogram")
    plt.savefig("plots/dendrogram.png", dpi=300, bbox_inches='tight')
    plt.show()

# 7. CLUSTER SUMMARY
def cluster_summary(rfm):
    summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'ReturnRatio': 'mean',
        'CustomerID': 'count'
    })

    print("\nCluster Summary:")
    print(summary)
    return summary

# 8. BUSINESS SEGMENTATION (FINAL OUTPUT)
def business_segmentation(rfm):

    r_high = rfm['Recency'].quantile(0.75)
    r_low = rfm['Recency'].quantile(0.25)

    f_high = rfm['Frequency'].quantile(0.75)
    m_high = rfm['Monetary'].quantile(0.75)

    ret_high = rfm['ReturnRatio'].quantile(0.75)

    def segment(row):

        if row['ReturnRatio'] > ret_high:
            return "High Return Customers"

        elif row['Recency'] > r_high:
            return "Churn Risk Customers"

        elif row['Frequency'] > f_high and row['Monetary'] > m_high:
            return "Champions"

        elif row['Frequency'] > f_high:
            return "Loyal Customers"

        elif row['Recency'] < r_low:
            return "Promising Customers"

        else:
            return "Occasional Customers"

    rfm['Segment'] = rfm.apply(segment, axis=1)


    return rfm

# 9. VISUALIZATION
def plot_segments(rfm):
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Segment')
    plt.savefig("plots/segments.png", dpi=300, bbox_inches='tight')
    plt.show()

# SILHOUETTE SCORE PLOT
def plot_silhouette_scores(scores_df):
    plt.figure(figsize=(6, 4))
    plt.bar(scores_df["Model"], scores_df["Silhouette Score"])
    plt.title("Model Comparison (Silhouette Score)")
    plt.xlabel("Model")
    plt.ylabel("Silhouette Score")
    plt.savefig("plots/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def run_pipeline(rfm, show_plots=True):

    # Step 1: Scaling
    data = scale_features(rfm)

    # Step 2: Elbow Method
    print("\n--- Elbow Method ---")
    elbow_method(data)

    # Step 3: Dendrogram
    print("\n--- Dendrogram ---")
    plot_dendrogram(data)

    # Step 4: Model tuning
    print("\n--- Model Tuning ---")
    k_labels, k_score = tune_kmeans(data, show_plot=show_plots)
    d_labels, d_score = tune_dbscan(data, show_plot=show_plots)
    h_labels, h_score = tune_hierarchical(data, show_plot=show_plots)

    # Step 5: Model comparison
    scores = pd.DataFrame({
        "Model": ["KMeans", "DBSCAN", "Hierarchical"],
        "Silhouette Score": [k_score, d_score, h_score]
    })


    # Silhouette graph
    if show_plots:
        print("\n--- Silhouette Score Comparison ---")
        plot_silhouette_scores(scores)

    # Step 6: Select best model
    best_model = scores.loc[scores["Silhouette Score"].idxmax(), "Model"]
    

    # Step 7: Assign clusters
    if best_model == "KMeans":
        rfm['Cluster'] = k_labels
    elif best_model == "Hierarchical":
        rfm['Cluster'] = h_labels
    else:
        rfm['Cluster'] = d_labels

    # Step 8: Cluster Summary
    summary = cluster_summary(rfm)

    # Step 9: Business Segmentation
    rfm = business_segmentation(rfm)

    # Step 10: Visualization
    plot_segments(rfm)

    return rfm, scores, best_model, summary
