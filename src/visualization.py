import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import os
os.makedirs("plots", exist_ok=True)

# Ensure plots show when running .py files
pio.renderers.default = "browser"


#  RFM Distributions (EDA)
def plot_rfm_distributions(rfm, save = True):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    sns.histplot(rfm['Monetary'], bins=50)
    plt.title("Monetary Distribution")

    plt.subplot(1, 3, 2)
    sns.histplot(rfm['Frequency'], bins=50)
    plt.title("Frequency Distribution")

    plt.subplot(1, 3, 3)
    sns.histplot(rfm['Recency'], bins=50)
    plt.title("Recency Distribution")

    plt.tight_layout()
    if save:
        plt.savefig("plots/rfm_distributions.png", dpi=300, bbox_inches='tight')

    plt.show()


#  RFM Segment Count (Bar Chart)
def plot_rfm_segments(rfm, save = True):
    segment_count = rfm['RFM_Segment_Label'].value_counts().reset_index()
    segment_count.columns = ['Segment', 'Count']

    plt.figure(figsize=(6, 4))
    sns.barplot(data=segment_count, x='Segment', y='Count')
    plt.title("Customer Count by RFM Segment")
    plt.xticks(rotation=30)

    if save:
        plt.savefig("plots/rfm_segments.png", dpi=300, bbox_inches='tight')
    plt.show()


#  Cluster Visualization (2D)
def plot_cluster_scatter(rfm, save = True):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=rfm,
        x='Recency',
        y='Monetary',
        hue='Cluster',
        palette='viridis'
    )
    plt.title("Customer Segmentation (RFM + Clusters)")
    if save:
        plt.savefig("plots/cluster_scatter.png", dpi=300, bbox_inches='tight')

    plt.show()


# 3D Cluster Visualization (Plotly)
def plot_3d_clusters(rfm, save = True):
    fig = px.scatter_3d(
        rfm,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Cluster',
        title='3D Customer Segmentation'
    )
    if save:
        fig.write_image("plots/cluster_3d.png")
    fig.show()


#  Return Behavior Visualization
def plot_return_behavior(rfm, save = True):
    plt.figure(figsize=(6, 4))
    sns.histplot(rfm['ReturnRatio'], bins=50)
    plt.title("Return Ratio Distribution")
    if save:
        plt.savefig("plots/return_ratio.png", dpi=300, bbox_inches='tight')
    plt.show()

# Customer Distribution by Business Segments

def plot_detailed_segments(rfm, save=True):
    plt.figure(figsize=(9, 5))

    order = rfm['RFM_Customer_Segments'].value_counts().index

    sns.countplot(
        data=rfm,
        x='RFM_Customer_Segments',
        order=order,
        hue='RFM_Customer_Segments',
        palette='viridis',
        legend=False
    )

    plt.xticks(rotation=30)
    plt.title("Customer Distribution by Business Segments")
    plt.xlabel("")
    plt.ylabel("Number of Customers")

    if save:
        plt.savefig("plots/rfm_detailed_segments.png", dpi=300, bbox_inches='tight')

    plt.show()

# Customer Segmentation (RFM-Based)

def plot_rfm_segment_scatter(rfm, save=True):
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=rfm,
        x='Recency',
        y='Monetary',
        hue='RFM_Customer_Segments',
        palette='tab10',
        alpha=0.7
    )

    plt.title("Customer Segmentation (RFM-Based)")
    plt.xlabel("Recency")
    plt.ylabel("Monetary Value")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    if save:
        plt.savefig("plots/rfm_segment_scatter.png", dpi=300, bbox_inches='tight')

    plt.show()    