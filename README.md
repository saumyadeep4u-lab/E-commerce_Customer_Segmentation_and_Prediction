# 🛒 E-Commerce Customer Segmentation & Churn Prediction

An end-to-end Machine Learning project that groups e-commerce customers into distinct behavioral profiles using **RFM Analysis** and unsupervised clustering (**K-Means & Hierarchical Clustering**), and trains a supervised classification model to predict **customer churn** in real time through a deployed Streamlit web application.

🌐 **Live Demo:** [Try the Streamlit Churn Prediction App Here](https://YOUR-STREAMLIT-LINK.streamlit.app)

---

## 🚀 Key Features
- 📊 **RFM Feature Engineering:** Evaluates customer value across Recency, Frequency, and Monetary metrics, alongside product return ratios.
- 🧩 **Unsupervised Customer Segmentation:** Implements K-Means and Hierarchical clustering validated via Elbow Method, Silhouette Scores, and Dendrograms.
- 🔮 **Supervised Churn Prediction:** Predicts customer churn probability using trained classification pipelines (`churn_model.pkl`).
- 📁 **Single & Bulk Prediction Support:** Interactive Streamlit UI (`app.py`) supporting both single-customer risk checks and bulk CSV uploads.
- 📈 **Executive BI Reporting:** Includes an interactive **Power BI Dashboard** and comprehensive **Executive Presentation (PPT)** for business stakeholders.

---

## 🔄 End-to-End Workflow

### 1. Exploratory Data Analysis & Preprocessing (`notebooks/`)
- Cleaning raw transactional e-commerce records (`data.csv`) and handling missing customer IDs (`known.csv` vs. `unknown.csv`).
- Analyzing country-wise sales distributions, order patterns, and product return ratios.

### 2. RFM Analysis & Behavioral Clustering (`Clustering Pipeline.ipynb` & `src/`)
- Computing Recency, Frequency, and Monetary scores (`rfm_final.csv`).
- Determining optimal cluster counts using `elbow_method.png`, `kmeans_silhouette.png`, `hierarchical_silhouette.png`, and `dendrogram.png`.
- Profiling customer personas into actionable segments (`cluster_summary.csv`, `cluster_3d.png`).

### 3. Churn Model Training (`Prediction Pipeline.ipynb` & `src/`)
- Engineering churn indicators from customer purchase histories (`churn_dataset.csv`).
- Training, comparing (`model_comparison.png`), and serializing the best-performing churn classifier (`churn_model.pkl`).

### 4. Deployment & Testing (`app.py` & `App_Testing_Data/`)
- Real-time churn inference via Streamlit for single customer inputs (`App_Testing_Single Customer.txt`) and batch CSV processing (`App_Testing_Bulk Upload.csv`).

---

## 💼 Business Implications
- **🎯 Targeted Retention Campaigns:** Identifies high-value customers at risk of churning early so marketing teams can trigger personalized retention offers.
- **💰 Marketing ROI Optimization:** Tailors promotional spend based on distinct RFM customer personas (e.g., Champions, Loyal Customers, At-Risk, Hibernating).
- **📉 Return & Revenue Protection:** Tracks return ratios alongside purchase frequency to mitigate revenue leakage and improve customer lifetime value (CLV).
- **⚡ Automated Batch Triage:** Enables CRM and support teams to upload customer lists in bulk and instantly export prioritized churn predictions (`churn_predictions.csv`).

---

## 📂 Project Structure

```text
E-commerce_Customer_Segmentation_and_Prediction/
├── App_Testing_Data/
│   ├── Bulk_Upload/
│   │   ├── App_Testing_Bulk Upload.csv                  # Sample batch input for bulk testing
│   │   └── churn_predictions.csv                        # Exported batch churn predictions
│   └── App_Testing_Single Customer.txt                  # Sample inputs for single-customer testing
├── data/
│   ├── churn_dataset.csv                                # Processed dataset for churn modeling
│   ├── cluster_summary.csv                              # Aggregated metrics per customer cluster
│   ├── data.csv                                         # Raw e-commerce transaction dataset
│   ├── description.txt                                  # Dataset feature dictionary
│   ├── known.csv                                        # Transactions with identified Customer IDs
│   ├── rfm_final.csv                                    # Engineered RFM features and cluster labels
│   └── unknown.csv                                      # Guest transactions without Customer IDs
├── notebooks/
│   ├── country_sales_distribution.png                   # Geographic sales breakdown chart
│   ├── eda.ipynb                                        # Exploratory Data Analysis notebook
│   └── preprocessing.ipynb                              # Data cleaning & filtering notebook
├── plots/
│   ├── cluster_3d.png                                   # 3D visualization of customer clusters
│   ├── dendrogram.png                                   # Hierarchical clustering dendrogram
│   ├── elbow_method.png                                 # K-Means Elbow curve for optimal K
│   ├── hierarchical_silhouette.png                      # Silhouette analysis for Hierarchical model
│   ├── kmeans_silhouette.png                            # Silhouette analysis for K-Means model
│   ├── model_comparison.png                             # Churn classification model benchmark
│   ├── return_ratio.png                                 # Return rate distribution analysis
│   ├── rfm_detailed_segments.png                        # Detailed breakdown of RFM personas
│   ├── rfm_distributions.png                            # Recency, Frequency, Monetary distributions
│   ├── rfm_segment_scatter.png                          # Scatter plot of RFM customer segments
│   ├── rfm_segments.png                                 # Overview of RFM segment sizes
│   └── segments.png                                     # Final cluster distribution chart
├── src/
│   ├── churn_model_v3.py                                # Churn prediction model architecture
│   ├── churn_training.py                                # Training pipeline for churn classifier
│   ├── clustering.py                                    # K-Means & Hierarchical clustering script
│   ├── rfm_analysis.py                                  # RFM metric calculation module
│   └── visualization.py                                 # Plotting utilities for clusters and metrics
├── (Final Dashboard Power BI) - E-commerce Cu...        # Interactive Power BI Dashboard
├── (Final Report PPT) - E-commerce Customer S...        # Executive Presentation deck
├── Clustering Pipeline.ipynb                            # End-to-end clustering workflow notebook
├── Prediction Pipeline.ipynb                            # End-to-end churn prediction notebook
├── app.py                                               # Streamlit web application entry point
├── churn_model.pkl                                      # Serialized churn prediction model
├── .gitignore                                           # Git ignore rules for virtual environments
├── requirements.txt                                     # Python dependencies for deployment
└── README.md                                            # Project documentation
```

---

## 💻 How to Run Locally

1. **Clone the repository:** `git clone [https://github.com/saumyadeep4u-lab/E-commerce_Customer_Segmentation_and_Prediction.git]`
2. **Open the project folder:** `cd E-commerce_Customer_Segmentation_and_Prediction`
3. **Install dependencies:** `pip install -r requirements.txt`
4. **Launch the Streamlit app:** `streamlit run app.py`

---

## 👨‍💻 Developer
**Saumyadeep Bardhan**
