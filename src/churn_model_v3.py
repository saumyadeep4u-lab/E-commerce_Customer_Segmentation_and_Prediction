import pandas as pd

def create_churn_features_pro(
    df,
    snapshot_date,
    observation_days=180,
    prediction_days=90
):
    """
    Advanced churn dataset builder with:
    - Proper time-based split (no leakage)
    - Rich feature engineering
    """

    df = df.copy()

    # 1. Preprocessing
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    snapshot_date = pd.to_datetime(snapshot_date)

    df = df[df['Quantity'] > 0]

    # 2. Define time windows
    obs_start = snapshot_date - pd.Timedelta(days=observation_days)
    pred_end = snapshot_date + pd.Timedelta(days=prediction_days)

    df_obs = df[
        (df['InvoiceDate'] >= obs_start) &
        (df['InvoiceDate'] < snapshot_date)
    ]

    print("Number of customers:", df_obs['CustomerID'].nunique())
    print("Observation window:", obs_start, "to", snapshot_date)

    df_pred = df[
        (df['InvoiceDate'] >= snapshot_date) &
        (df['InvoiceDate'] < pred_end)
    ]

    churn_data = []

    # 3. Feature Engineering
    for customer, group in df_obs.groupby('CustomerID'):

        group = group.sort_values('InvoiceDate')

        # Basic timestamps
        first_purchase = group['InvoiceDate'].min()
        last_purchase = group['InvoiceDate'].max()

        # BASIC FEATURES
        recency = (snapshot_date - last_purchase).days
        tenure = (snapshot_date - first_purchase).days
        frequency = group['InvoiceNo'].nunique()
        monetary = (group['Quantity'] * group['UnitPrice']).sum()
        recency_ratio = recency / tenure if tenure > 0 else 0


        # 1. Average Order Value
        aov = monetary / frequency if frequency > 0 else 0

        # 2. Purchase intervals
        group['gap'] = group['InvoiceDate'].diff().dt.days
        avg_gap = group['gap'].mean() if not group['gap'].isna().all() else 0
        max_gap = group['gap'].max() if not group['gap'].isna().all() else 0
        std_gap = group['gap'].std()
        std_gap = 0 if pd.isna(std_gap) else std_gap
        
        # 3. Recent activity (last 30 days)
        recent_window = snapshot_date - pd.Timedelta(days=30)
        recent_group = group[group['InvoiceDate'] >= recent_window]

        recent_freq = recent_group['InvoiceNo'].nunique()
        recent_monetary = (recent_group['Quantity'] * recent_group['UnitPrice']).sum()

        recent_freq = max(0, recent_freq)
        recent_monetary = max(0, recent_monetary)

        # 4. Activity ratio
        active_days = group['InvoiceDate'].nunique()
        activity_ratio = active_days / tenure if tenure > 0 else 0

        # 5. Purchase trend
        midpoint = first_purchase + (last_purchase - first_purchase) / 2

        first_half = group[group['InvoiceDate'] < midpoint]
        second_half = group[group['InvoiceDate'] >= midpoint]

        trend = (
            second_half['InvoiceNo'].nunique() -
            first_half['InvoiceNo'].nunique()
        )

        # 6. CLV proxy
        clv_proxy = monetary / tenure if tenure > 0 else 0

        # CHURN LABEL (FUTURE)
        future = df_pred[df_pred['CustomerID'] == customer]
        churn = 1 if future.empty else 0

        churn_data.append([
            customer,
            recency, frequency, monetary, tenure,
            aov, avg_gap, max_gap, std_gap,
            recent_freq, recent_monetary,
            activity_ratio, trend, clv_proxy,recency_ratio,
            churn
        ])

    # 4. Final DataFrame
    churn_df = pd.DataFrame(churn_data, columns=[
        'CustomerID',
        'Recency', 'Frequency', 'Monetary', 'Tenure',
        'AOV', 'AvgGap', 'MaxGap', 'StdGap',
        'RecentFreq', 'RecentMonetary',
        'ActivityRatio', 'Trend', 'CLV_Proxy','RecencyRatio',
        'Churn'
    ])

    return churn_df