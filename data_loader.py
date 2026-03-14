import pandas as pd
import numpy as np
import os

def load_all_data():
    """Load all data from data directory"""
    base = "data"
    datasets = {}

    for folder in os.listdir(base):
        path = os.path.join(base, folder)
        if os.path.isdir(path):
            files = os.listdir(path)
            datasets[folder] = {}

            for f in files:
                full_path = os.path.join(path, f)
                if f.endswith(".txt"):
                    datasets[folder][f] = pd.read_csv(full_path, sep="\t")
                elif f.endswith(".csv"):
                    datasets[folder][f] = pd.read_csv(full_path)

    return datasets


def load_jellyfish_data(lookback_days=7, forecast_days=1):
    """
    Load jellyfish data and aggregate per beach per day for forecasting.
    
    FORECASTING TASK:
    - Input: Time series of observations from previous lookback_days (7 days default)
    - Output: Binary prediction for each beach on forecast_days ahead (next day)
    - Grain: Per beach per day (not per observation)
    
    This enables:
    1. Per-beach daily predictions
    2. Temporal forecasting (predict next day based on history)
    3. Early warning systems for beach closures
    
    Args:
        lookback_days: Number of historical days to use as features (default: 7)
        forecast_days: Number of days ahead to forecast (default: 1 = next day)
    
    Returns:
        X: Feature sequences (n_samples, lookback_days, n_features_per_day)
           Shape: (number of beach-day combinations, 7, 11)
        y: Binary labels for forecast_days ahead (n_samples,)
        metadata: DataFrame with beach_id, beach_name, forecast_date, etc.
    """
    data = load_all_data()

    events = data["Citizen Science based jellyfish observations along the Israeli Mediterranean coast in 2011-2025"]["event.txt"]
    occ = data["Citizen Science based jellyfish observations along the Israeli Mediterranean coast in 2011-2025"]["occurrence.txt"]
    meas = data["Citizen Science based jellyfish observations along the Israeli Mediterranean coast in 2011-2025"]["extendedmeasurementorfact.txt"]

    # Extract beach ID and name (robust to missing/invalid locality strings)
    locality_text = events["verbatimLocality"].astype(str)
    events["beach_id"] = pd.to_numeric(
        locality_text.str.extract(r"beach:(\d+)", expand=False),
        errors="coerce",
    )
    events["beach_name"] = events["locality"]

    # Keep only rows with valid beach IDs
    events = events.dropna(subset=["beach_id"]).copy()
    events["beach_id"] = events["beach_id"].astype(np.int64)


    # Keep only rows with valid beach IDs
    events = events.dropna(subset=["beach_id"]).copy()
    events["beach_id"] = events["beach_id"].astype(np.int64)

    # Extract species from occurrenceID
    occ["species"] = occ["occurrenceID"].str.extract(r'-(.*?)_by')

    # Merge tables
    df = occ.merge(events, on="eventID", how="left")
    df = df.merge(
        meas[["occurrenceID", "measurementValue"]],
        on="occurrenceID",
        how="left"
    )

    # ========================================================================
    # PARSE DIAMETER
    # ========================================================================
    def diameter_to_float(v):
        if pd.isna(v):
            return np.nan
        text = str(v).strip()
        if not text:
            return np.nan
        lowered = text.lower()
        if lowered in {"unspecified", "unknown", "nan", "none", "na", "n/a"}:
            return np.nan
        cleaned = "".join(ch for ch in text if ch.isdigit() or ch in {".", "-"})
        if not cleaned:
            return np.nan
        try:
            if "-" in cleaned[1:]:
                parts = [part for part in cleaned.split("-") if part]
                if len(parts) >= 2:
                    return (float(parts[0]) + float(parts[1])) / 2
            return float(cleaned)
        except ValueError:
            return np.nan

    df["diameter_cm"] = df["measurementValue"].apply(diameter_to_float)

    # ========================================================================
    # PARSE DATETIME
    # ========================================================================
    df["eventDate"] = pd.to_datetime(df["eventDate"])
    dt = df["eventDate"]

    df["month"] = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["date"] = dt.dt.date

    df["sin_month"] = np.sin(2*np.pi*df["month"]/12)
    df["cos_month"] = np.cos(2*np.pi*df["month"]/12)

    # ========================================================================
    # PARSE DISTANCE FROM COAST
    # ========================================================================
    def parse_distance(x):
        if pd.isna(x):
            return np.nan
        if "distance_from_coast:" in x:
            val = x.split("distance_from_coast:")[1].split(",")[0]
            if "-" in val:
                a,b = val.replace("m","").split("-")
                return (float(a)+float(b))/2
        return np.nan

    df["distance_from_coast"] = df["locationRemarks"].apply(parse_distance)

    # ========================================================================
    # PARSE STING REPORTS
    # ========================================================================
    def sting_flag(x):
        if pd.isna(x):
            return 0
        return int("Yes" in x)

    df["sting"] = df["locationRemarks"].apply(sting_flag)

    # ========================================================================
    # ENCODE CATEGORICAL VARIABLES
    # ========================================================================
    df["species_id"] = df["species"].astype("category").cat.codes
    df["protocol_id"] = df["samplingProtocol"].astype("category").cat.codes

    # ========================================================================
    # BLOOM INTENSITY
    # ========================================================================
    df["bloom_intensity"] = df["organismQuantity"].map({
        "Few": 2,
        "Some": 10,
        "Swarm": 50
    }).fillna(df["individualCount"])

    # ========================================================================
    # AGGREGATE PER BEACH PER DAY
    # ========================================================================
    # This is the key step: convert observation-level data to daily aggregates
    
    daily_features = df.groupby(['beach_id', 'date']).agg({
        'beach_name': 'first',
        'decimalLatitude': 'mean',
        'decimalLongitude': 'mean',
        'month': 'first',
        'day_of_month': 'first',
        'sin_month': 'first',
        'cos_month': 'first',
        'distance_from_coast': 'mean',
        'species_id': 'mean',           # Average species ID observed that day
        'diameter_cm': 'mean',          # Average diameter that day
        'bloom_intensity': 'max',       # Maximum intensity that day
        'sting': 'max',                 # Any sting reports that day
        'occurrenceID': 'count'         # Number of observations that day
    }).reset_index()
    
    daily_features.rename(columns={'occurrenceID': 'observation_count'}, inplace=True)
    
    # Create binary label: jellyfish presence (1 if bloom observed, 0 otherwise)
    daily_features['jellyfish_present'] = (daily_features['bloom_intensity'] > 5).astype(int)
    
    # ========================================================================
    # CREATE SEQUENCES FOR TEMPORAL FORECASTING
    # ========================================================================
    # Each sequence: 7 days of history → predict day 8
    
    # Sort by beach and date to ensure temporal order
    daily_features = daily_features.sort_values(['beach_id', 'date']).reset_index(drop=True)
    
    # Features to use per day (11 features per day)
    feature_cols = [
        'decimalLatitude',       # Beach location
        'decimalLongitude',      # Beach location
        'month',                 # Seasonal signal
        'day_of_month',          # Within-month signal
        'sin_month',             # Sinusoidal seasonal encoding
        'cos_month',             # Sinusoidal seasonal encoding
        'distance_from_coast',   # Beach characteristic
        'species_id',            # Species composition
        'diameter_cm',           # Jellyfish size
        'observation_count',     # Observation frequency
        'sting'                  # Safety indicator
    ]
    
    X_sequences = []
    y_labels = []
    metadata_list = []
    
    unique_beaches = daily_features['beach_id'].unique()
    
    for beach_id in unique_beaches:
        beach_data = daily_features[daily_features['beach_id'] == beach_id].reset_index(drop=True)
        
        # Skip beaches with insufficient data
        if len(beach_data) < lookback_days + forecast_days:
            continue
        
        # Create sequences for this beach
        for i in range(len(beach_data) - lookback_days - forecast_days + 1):
            # ============================================================
            # HISTORICAL WINDOW (lookback_days = 7 days)
            # ============================================================
            # Extract previous 7 days of observations for this beach
            historical = beach_data.iloc[i:i+lookback_days][feature_cols].fillna(0).values.astype(np.float32)
            # Shape: (7, 11) - 7 days, 11 features per day
            
            # ============================================================
            # FUTURE LABEL (forecast_days = 1 day ahead)
            # ============================================================
            # What to predict: will there be jellyfish forecast_days from now?
            future_idx = i + lookback_days + forecast_days - 1
            future_label = beach_data.iloc[future_idx]['jellyfish_present']
            future_date = beach_data.iloc[future_idx]['date']
            beach_name = beach_data.iloc[future_idx]['beach_name']
            
            X_sequences.append(historical)
            y_labels.append(future_label)
            
            # Store metadata for interpretation
            metadata_list.append({
                'beach_id': beach_id,
                'beach_name': beach_name,
                'forecast_date': future_date,           # Date being predicted
                'latitude': beach_data.iloc[future_idx]['decimalLatitude'],
                'longitude': beach_data.iloc[future_idx]['decimalLongitude'],
                'lookback_start': beach_data.iloc[i]['date'],
                'lookback_end': beach_data.iloc[i+lookback_days-1]['date'],
                'jellyfish_observed': int(future_label)
            })
    
    # ========================================================================
    # CONVERT TO NUMPY ARRAYS
    # ========================================================================
    X = np.array(X_sequences, dtype=np.float32)  # Shape: (n_samples, 7, 11)
    y = np.array(y_labels, dtype=np.float32)     # Shape: (n_samples,)
    metadata = pd.DataFrame(metadata_list)
    
    # ========================================================================
    # PRINT SUMMARY STATISTICS
    # ========================================================================
    print(f"\n" + "="*80)
    print(f"JELLYFISH FORECASTING DATA - PER BEACH PER DAY")
    print(f"="*80)
    print(f"Lookback window:        {lookback_days} days (historical features)")
    print(f"Forecast horizon:       {forecast_days} day(s) ahead (prediction target)")
    print(f"Total sequences:        {len(X)} beach-day combinations")
    print(f"Input shape:            {X.shape} (samples, days, features_per_day)")
    print(f"Output shape:           {y.shape}")
    print(f"")
    print(f"Positive samples:       {int(y.sum())} ({y.mean()*100:.1f}%)")
    print(f"Negative samples:       {len(y) - int(y.sum())} ({(1-y.mean())*100:.1f}%)")
    print(f"")
    print(f"Unique beaches:         {len(unique_beaches)}")
    print(f"Features per day:       {len(feature_cols)}")
    print(f"Feature columns:        {feature_cols}")
    print(f"")
    print(f"Date range:             {metadata['lookback_start'].min()} to {metadata['forecast_date'].max()}")
    print(f"="*80 + "\n")
    
    return X, y, metadata
