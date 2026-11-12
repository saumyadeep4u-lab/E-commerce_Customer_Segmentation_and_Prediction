import pandas as pd
import datetime as dt
import numpy as np

#  1. Prepare Data 
def prepare_data(df):
    df = df.copy()

    # Ensure datetime format
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Create TotalPrice (includes returns automatically)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Create Return Flag
    df['IsReturn'] = df['Quantity'] < 0

    return df


#  2. Create RFM Features 
def create_rfm(df):

    reference_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

    # Ensure CustomerID is clean
    df['CustomerID'] = df['CustomerID'].astype(float)

    # Purchases only
    purchase_df = df[df['Quantity'] > 0]

    # 🔹 Frequency
    frequency = purchase_df.groupby('CustomerID')['InvoiceNo'].nunique()

    # 🔹 Return Count
    return_count = df[df['Quantity'] < 0].groupby('CustomerID').size()

    # 🔹 RFM Base
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'TotalPrice': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'TotalPrice': 'Monetary'
    })

    
    rfm.index = rfm.index.astype(float)

    # Add Frequency and Return Count
    rfm['Frequency'] = frequency
    rfm['ReturnCount'] = return_count

    # Fill missing
    rfm['Frequency'] = rfm['Frequency'].fillna(0)
    rfm['ReturnCount'] = rfm['ReturnCount'].fillna(0)

    return rfm


#  3. Add Return Ratio (Behavior Feature)
def add_return_features(rfm):
    # Step 1: Create ratio
    rfm['ReturnRatio'] = rfm['ReturnCount'] / rfm['Frequency']

    # Step 2: Handle division issues
    rfm['ReturnRatio'] = rfm['ReturnRatio'].replace([np.inf, -np.inf], 0).fillna(0)

    # Step 3: Cap extreme values
    rfm['ReturnRatio'] = rfm['ReturnRatio'].clip(upper=1)

    return rfm


#  4. Calculate Quantiles
def calculate_quantiles(rfm):
    return rfm[['Recency', 'Frequency', 'Monetary']].quantile([0.25, 0.50, 0.75])


#  5. RFM Scoring Logic
def rfm_score(x, column, quantiles):
    if column == 'Recency':
        if x <= quantiles[column][0.25]:
            return 4
        elif x <= quantiles[column][0.50]:
            return 3
        elif x <= quantiles[column][0.75]:
            return 2
        else:
            return 1
    else:
        if x <= quantiles[column][0.25]:
            return 1
        elif x <= quantiles[column][0.50]:
            return 2
        elif x <= quantiles[column][0.75]:
            return 3
        else:
            return 4


# 6. Apply RFM Scores
def apply_rfm_scores(rfm):
    quantiles = calculate_quantiles(rfm)

    rfm['R'] = rfm['Recency'].apply(rfm_score, args=('Recency', quantiles))
    rfm['F'] = rfm['Frequency'].apply(rfm_score, args=('Frequency', quantiles))
    rfm['M'] = rfm['Monetary'].apply(rfm_score, args=('Monetary', quantiles))

    return rfm


# 7. Create Segments
def create_segments(rfm):
    rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis=1)

    def assign_segment(score):
        if score < 5:
            return 'Low Value'
        elif score < 9:
            return 'Mid Value'
        else:
            return 'High Value'

    rfm['RFM_Segment_Label'] = rfm['RFM_Score'].apply(assign_segment)

    return rfm


#  8. Detailed Business Segments
def detailed_segments(rfm):
    rfm['RFM_Customer_Segments'] = ''

    rfm.loc[rfm['RFM_Score'] >= 9, 'RFM_Customer_Segments'] = 'Loyal Customers'
    rfm.loc[(rfm['RFM_Score'] >= 6) & (rfm['RFM_Score'] < 9), 'RFM_Customer_Segments'] = 'Potential Loyalists'
    rfm.loc[(rfm['RFM_Score'] >= 5) & (rfm['RFM_Score'] < 6), 'RFM_Customer_Segments'] = 'At Risk Customers'
    rfm.loc[(rfm['RFM_Score'] >= 4) & (rfm['RFM_Score'] < 5), 'RFM_Customer_Segments'] = "Can't Lose Them"
    rfm.loc[(rfm['RFM_Score'] >= 3) & (rfm['RFM_Score'] < 4), 'RFM_Customer_Segments'] = 'Lost Customers'

    return rfm


#  9. Full Pipeline 
def run_rfm_pipeline(df):
    df = prepare_data(df)
    rfm = create_rfm(df)
    rfm = add_return_features(rfm)
    rfm = apply_rfm_scores(rfm)
    rfm = create_segments(rfm)
    rfm = detailed_segments(rfm)

    rfm = rfm.reset_index()
    
    return rfm