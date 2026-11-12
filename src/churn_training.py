import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    precision_recall_curve, roc_auc_score, roc_curve,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# STEP 1: PREPARE DATA
def prepare_data(df):
    X = df.drop(columns=['CustomerID', 'Churn', 'Recency'], errors='ignore')
    y = df['Churn']
    return X, y


# STEP 2: SPLIT DATA
def split_data(X, y):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# STEP 3: MODELS
def get_models():

    models = {
        "Logistic Regression": (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(class_weight='balanced', max_iter=1000))
            ]),
            {"model__C": [0.1, 1, 10]}
        ),

        "RandomForest": (
            RandomForestClassifier(class_weight='balanced', random_state=42),
            {"n_estimators": [100, 300], "max_depth": [5, 10, None]}
        ),

        "AdaBoost": (
            AdaBoostClassifier(random_state=42),
            {"n_estimators": [50, 100]}
        ),

        "XGBoost": (
            XGBClassifier(eval_metric='logloss', random_state=42),
            {
                "n_estimators": [100, 300],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1],
                "scale_pos_weight": [1, 2, 3]
            }
        ),

        "SVC": (
            Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(probability=True, class_weight='balanced'))
            ]),
            {"model__C": [1, 10]}
        )
    }

    return models


# STEP 4: THRESHOLD OPTIMIZATION
def optimize_threshold(y_true, y_prob):

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)

    return thresholds[best_idx]

# STEP 5: TRAIN MODELS
def train_models(X_train, X_val, y_train, y_val):

    models = get_models()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    best_model = None
    best_threshold = 0
    best_score = 0
    best_model_name = ""

    results = []
    roc_data = {}

    for name, (model, params) in models.items():

        print(f"\n Training {name}...")

        grid = GridSearchCV(model, params, scoring='f1', cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)

        best_est = grid.best_estimator_

        # Validation predictions
        y_val_prob = best_est.predict_proba(X_val)[:, 1]

        threshold = optimize_threshold(y_val, y_val_prob)
        y_val_pred = (y_val_prob >= threshold).astype(int)

        # Metrics
        f1 = f1_score(y_val, y_val_pred)
        acc = accuracy_score(y_val, y_val_pred)
        roc = roc_auc_score(y_val, y_val_prob)

        print("Best Params:", grid.best_params_)
        print(f"Validation F1: {f1:.4f} | Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
        print(classification_report(y_val, y_val_pred))

        # Store results
        results.append({
            "Model": name,
            "F1_Score": round(f1, 4),
            "Accuracy": round(acc, 4),
            "ROC_AUC": round(roc, 4)
        })

        # Store ROC curve data
        fpr, tpr, _ = roc_curve(y_val, y_val_prob)
        roc_data[name] = (fpr, tpr, roc)

        # Track best model
        if f1 > best_score:
            best_score = f1
            best_model = best_est
            best_threshold = threshold
            best_model_name = name

    # MODEL COMPARISON TABLE
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="F1_Score", ascending=False).reset_index(drop=True)

    print("\n Model Comparison:")
    print(results_df[['Model', 'F1_Score', 'Accuracy', 'ROC_AUC']])

    print(f"\n Best Model: {best_model_name}")

    # ROC CURVE (ALL MODELS)
    plt.figure()

    for model_name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.2f})")

    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison (Validation Set)")
    plt.legend()
    plt.show()

    return best_model, best_threshold, best_model_name

# STEP 6: FINAL TEST EVALUATION
def evaluate(model, threshold, X_test, y_test):

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("\n FINAL TEST RESULTS")
    print(f"F1: {f1_score(y_test, y_pred):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # ROC for final model
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label="Final Model")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Final Model ROC Curve (Test Set)")
    plt.legend()
    plt.show()

    return None

# STEP 7: SAVE MODEL
def save_all(model, threshold, feature_names, X_train):

    artifacts = {
        "model": model,
        "threshold": threshold,
        "features": list(feature_names),
        "medians": X_train.median().to_dict()
    }

    with open("churn_model.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print("\n Model saved!")

# FINAL PIPELINE
def run_pipeline(df):

    X, y = prepare_data(df)
    feature_names = X.columns

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    model, threshold, model_name = train_models(X_train, X_val, y_train, y_val)

    evaluate(model, threshold, X_test, y_test)

    save_all(model, threshold, feature_names, X_train)
    medians = X_train.median().to_dict()

    final_output = generate_business_output(
    df=df,
    model=model,
    threshold=threshold,
    feature_names=feature_names,
    medians=medians
)
    print("\n SAMPLE FINAL OUTPUT:")
    print(final_output.head())

    print(f"\n FINAL BEST MODEL: {model_name}")
    print("\n PIPELINE COMPLETED!")

    return final_output


def generate_business_output(df, model, threshold, feature_names, medians):
    
    df = df.copy()

    # Prepare features
    X = df.reindex(columns=feature_names)
    X = X.fillna(medians)
    X = X.fillna(0)
    # Predict
    df['Churn_Prob'] = model.predict_proba(X)[:, 1]
    df['Churn_Pred'] = (df['Churn_Prob'] >= threshold).astype(int)

    # Risk
    def risk_category(p):
        if p >= 0.7:
            return "High Risk"
        elif p >= 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    df['Churn Risk Category'] = df['Churn_Prob'].apply(risk_category)

    # Recommended Action

    df['Recommended Action'] = np.where(
    df['Churn Risk Category'] == "High Risk",
    "Immediate retention campaign",
    np.where(
        df['Churn Risk Category'] == "Medium Risk",
        "Send targeted offers",
        "Maintain engagement"
    )
)
    # ----------------------------------
    #  RFM Scoring
    # ----------------------------------
    df['R'] = pd.qcut(df['Recency'], q=4, duplicates='drop').cat.codes.replace(-1, 0) + 1
    df['F'] = pd.qcut(df['Frequency'], q=4, duplicates='drop').cat.codes.replace(-1, 0) + 1
    df['M'] = pd.qcut(df['Monetary'], q=4, duplicates='drop').cat.codes.replace(-1, 0) + 1
    df['RFM_Score'] = (
        df['R'].astype(int) +
        df['F'].astype(int) +
        df['M'].astype(int)
    )

    #  Segment Labels
    def segment_label(score):
        if score >= 10:
            return "High Value"
        elif score >= 6:
            return "Mid Value"
        else:
            return "Low Value"

    df['RFM_Segment_Label'] = df['RFM_Score'].apply(segment_label)

    def customer_segment(score):
        if score >= 10:
            return "Loyal Customers"
        elif score >= 6:
            return "Potential Loyalists"
        else:
            return "Lost Customers"

    df['RFM_Customer_Segments'] = df['RFM_Score'].apply(customer_segment)

    #  Final Output Structure
    final_df = df[[
        'CustomerID',
        'Recency', 'Frequency', 'Monetary',
        'R', 'F', 'M',
        'RFM_Score',
        'RFM_Segment_Label',
        'RFM_Customer_Segments',
        'Churn_Pred',
        'Churn_Prob',
        'Churn Risk Category',
        'Recommended Action'
    ]]

    return final_df   
