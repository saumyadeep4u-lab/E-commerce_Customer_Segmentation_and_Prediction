import streamlit as st
import pickle
import pandas as pd

# BUSINESS THRESHOLDS 
LOW_THRESHOLD = 0.4
HIGH_THRESHOLD = 0.7

# HELPER FUNCTION 
def get_risk_and_prediction(prob):
    if prob >= HIGH_THRESHOLD:
        return "High Risk", 1
    elif prob >= LOW_THRESHOLD:
        return "Medium Risk", 1
    else:
        return "Low Risk", 0

# LOAD MODEL FILE
artifacts = pickle.load(open("churn_model.pkl", "rb"))

model = artifacts["model"]
feature_names = artifacts["features"]
medians = artifacts["medians"]


#  Import feature engineering function
from src.churn_model_v3 import create_churn_features_pro


# PAGE CONFIG
st.set_page_config(page_title="Churn Prediction App")

st.title("🛒 E-Commerce Churn Prediction")
st.write("Predict Customer Churn Based on Transaction Data")

# GLOBAL DATE SELECTOR
snapshot_input = st.date_input(
    "Select Date for Churn Prediction",
    value=pd.Timestamp.today().normalize(),
    min_value=pd.to_datetime("2010-01-01"),
    max_value=pd.to_datetime("2262-04-11")
)

snapshot_date = pd.to_datetime(snapshot_input)

st.caption("💡 Tip: Choose a date close to your dataset period for better predictions.")

# OPTION SELECTOR
option = st.radio(
    "Choose Prediction Mode:",
    ["Single Customer", "Bulk Upload (CSV)"]
)

# OPTION 1: SINGLE CUSTOMER INPUT
if option == "Single Customer":

    st.subheader("🔹 Manual Customer Input")

    st.warning("⚠️ For accurate predictions, provide aggregated customer data. Use bulk upload for best results.")

    Frequency = st.number_input("Total Purchases (Frequency)", min_value=0)
    Monetary = st.number_input("Total Spend (Monetary)", min_value=0.0)
    Tenure = st.number_input("Customer Tenure (days)", min_value=0)

    last_purchase_input = st.date_input(
        "Last Purchase Date",
        value=(snapshot_date - pd.Timedelta(days=30)),
        min_value=pd.to_datetime("1900-01-01"),
        max_value=pd.to_datetime("2262-04-11")
    )

    if st.button("Predict Churn"):

        last_purchase_ts = pd.to_datetime(last_purchase_input)

        if last_purchase_ts > snapshot_date:
            st.error("❌ Last Purchase Date cannot be after Prediction Date")
            st.stop()

        recency = (snapshot_date - last_purchase_ts).days

        if Tenure < recency:
            st.error("❌ Tenure cannot be less than days since last purchase")
            st.stop()

        input_dict = medians.copy()

        input_dict.update({
            "Frequency": Frequency,
            "Monetary": Monetary,
            "Tenure": Tenure,
            "Recency": recency
        })

        input_data = pd.DataFrame([input_dict])
        input_data = input_data.reindex(columns=feature_names).fillna(medians)

        prob = model.predict_proba(input_data)[0][1]
        risk, prediction = get_risk_and_prediction(prob)

        if prediction == 1:
            st.error(f"❌ Likely to churn (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Not likely to churn (Probability: {prob:.2f})")

        # Action
        action_map = {
             "High Risk": "Immediate retention campaign",
             "Medium Risk": "Send targeted offers",
             "Low Risk": "Maintain engagement"
           }

        st.info(f"📊 Risk Level: {risk}")
        st.success(f"🎯 Recommended Action: {action_map[risk]}")

# OPTION 2: BULK CSV UPLOAD
elif option == "Bulk Upload (CSV)":

    st.subheader("📂 Upload Transaction CSV")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.success("✅ File Uploaded!")
        st.write("Preview:")
        st.dataframe(df.head())

        if st.button("Run Churn Prediction"):

            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

            churn_df = create_churn_features_pro(
                df,
                snapshot_date=snapshot_date,
                observation_days=180,
                prediction_days=90
            )

            if churn_df.empty:
                st.error("❌ No data available after processing.")
                st.stop()

            model_input = churn_df.drop(
                columns=['CustomerID', 'Churn', 'Recency'],
                errors='ignore'
            )

            model_input = model_input[feature_names]
            model_input = model_input.fillna(medians)
            model_input = model_input.fillna(0) #safety fallback

            probs = model.predict_proba(model_input)[:, 1]
            
            churn_df['Churn_Probability'] = probs

    
            #  DISPLAY SECTION
            st.subheader("📊 Prediction Results")

            display_df = churn_df.drop(columns=["Churn"], errors="ignore")

            display_df[["Risk_Level", "Churn_Prediction"]] = display_df["Churn_Probability"].apply(
               lambda x: pd.Series(get_risk_and_prediction(x))
             )

            display_df["Recommended Action"] = display_df["Risk_Level"].map({
            "High Risk": "Immediate retention campaign",
             "Medium Risk": "Send targeted offers",
             "Low Risk": "Maintain engagement"
           })

        
            display_df = display_df.sort_values(by="Churn_Probability", ascending=False)
            st.dataframe(display_df)

            # 🔽 DOWNLOAD
            csv = display_df.to_csv(index=False)

            st.download_button(
                label="⬇️ Download Results",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

            st.success("🎯 Prediction Completed!")