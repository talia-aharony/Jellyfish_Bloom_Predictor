import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Beach to IMS station mapping
beach_station_map = {
    " Nahariya-Rosh Hniqra": "ROSH HANIQRA",
    " Acco-Nahariyah": "SHAVE ZIYYON",
    " Kiryat Yam-Acco": "HAIFA REFINERIES",
    " Shiqmona-Kiryat yam": "HAIFA REFINERIES",
    " Tira-Shiqmona": "HAIFA TECHNION",
    " Atlit-Tira": "EN KARMEL",
    " Dor-Atlit": "EN KARMEL",
    " Jisr a zarqa-Dor": "ZIKHRON YAAQOV",
    " Hadera-Jisr a zarqa": "HADERA PORT",
    " Michmoret-Hadera": "HADERA PORT",
    " Natanya-Michmoret": "EN HAHORESH",
    " Gaash-Natanya": "EN HAHORESH",
    " Herzlia-Gaash": "TEL AVIV COAST",
    " Tel Aviv-Herzlia": "TEL AVIV COAST",
    " Jaffa-Tel Aviv": "TEL AVIV COAST",
    " Rishon-Jaffa": "BET DAGAN MAN",
    " Palmahim-Rishon": "BET DAGAN MAN",
    " Ashdod-Palmahim": "ASHDOD PORT",
    " Ashkelon-Ashdod": "ASHDOD PORT",
    " Gaza-Ashkelon": "ASHQELON PORT"
}


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
                    datasets[folder][f] = pd.read_csv(full_path, sep="\t", low_memory=False)
                elif f.endswith(".csv"):
                    datasets[folder][f] = pd.read_csv(full_path, low_memory=False)

    return datasets


def load_and_parse_ims_weather(weather_csv_path):
    """
    Load and parse IMS weather data into hourly records (similar to citizen science structure)
    
    Args:
        weather_csv_path: Path to IMS weather CSV file
    
    Returns:
        DataFrame with parsed weather data (one row per 3-hour observation)
    """
    if not os.path.exists(weather_csv_path):
        print(f"❌ Weather file not found at {weather_csv_path}")
        return None
    
    print(f"\n📡 Loading IMS weather data from {weather_csv_path}...")
    
    try:
        weather = pd.read_csv(weather_csv_path)
        weather.columns = weather.columns.str.strip()
        
        print(f"✓ Raw weather data loaded: {len(weather)} records")
        
        # Parse datetime
        weather['DateTime'] = pd.to_datetime(weather['Date & Time (UTC)'], format='%d-%m-%Y %H:%M')
        weather['date'] = weather['DateTime'].dt.date
        weather['month'] = weather['DateTime'].dt.month
        weather['day_of_month'] = weather['DateTime'].dt.day
        weather['hour'] = weather['DateTime'].dt.hour
        
        # Seasonal encoding
        weather['sin_month'] = np.sin(2*np.pi*weather['month']/12)
        weather['cos_month'] = np.cos(2*np.pi*weather['month']/12)
        
        # Extract all weather variables with robust numeric conversion
        weather['Temperature_C'] = pd.to_numeric(weather['Temperature (°C)'], errors='coerce')
        weather['Wet_Temp_C'] = pd.to_numeric(weather['Wet temperature (°C)'], errors='coerce')
        weather['Dew_Point_C'] = pd.to_numeric(weather['Dew point temperature (°C)'], errors='coerce')
        weather['Humidity_percent'] = pd.to_numeric(weather['Relative humidity (%)'], errors='coerce')
        weather['Wind_Direction_deg'] = pd.to_numeric(weather['Wind direction (°)'], errors='coerce')
        weather['Wind_Speed_ms'] = pd.to_numeric(weather['Wind speed (m/s)'], errors='coerce')
        weather['Pressure_Station_hPa'] = pd.to_numeric(weather['Pressure at station level (hPa)'], errors='coerce')
        weather['Pressure_Sea_hPa'] = pd.to_numeric(weather['Pressure at sea level (hPa)'], errors='coerce')
        weather['Total_Clouds'] = pd.to_numeric(weather['Total clouds cover (code)'], errors='coerce')
        weather['Low_Clouds'] = pd.to_numeric(weather['Total low clouds cover (code)'], errors='coerce')
        weather['Visibility'] = pd.to_numeric(weather['Visibility (code)'], errors='coerce')
        
        # Keep station per row (file may contain multiple stations)
        if 'Station' in weather.columns:
            weather['station'] = weather['Station'].astype(str).str.strip()
        else:
            weather['station'] = "Unknown"

        unique_stations = weather['station'].dropna().unique().tolist()
        
        print(f"✓ Parsed weather data: {len(weather)} observations")
        print(f"  Unique stations: {len(unique_stations)}")
        if unique_stations:
            preview = ', '.join(unique_stations[:5])
            if len(unique_stations) > 5:
                preview += ", ..."
            print(f"  Stations sample: {preview}")
        print(f"  Date range: {weather['date'].min()} to {weather['date'].max()}")
        print(f"  Temperature: {weather['Temperature_C'].min():.1f}°C to {weather['Temperature_C'].max():.1f}°C")
        print(f"  Wind speed: {weather['Wind_Speed_ms'].min():.1f} to {weather['Wind_Speed_ms'].max():.1f} m/s")
        
        return weather
        
    except Exception as e:
        print(f"❌ Error loading weather data: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_citizen_science_data():
    """
    Load and parse citizen science jellyfish data
    
    Returns:
        DataFrame with parsed citizen science observations
    """
    print(f"\n👥 Loading citizen science data...")
    
    data = load_all_data()

    events = data["Citizen Science based jellyfish observations along the Israeli Mediterranean coast in 2011-2025"]["event.txt"]
    occ = data["Citizen Science based jellyfish observations along the Israeli Mediterranean coast in 2011-2025"]["occurrence.txt"]
    meas = data["Citizen Science based jellyfish observations along the Israeli Mediterranean coast in 2011-2025"]["extendedmeasurementorfact.txt"]

    # Extract beach ID and name
    locality_text = events["verbatimLocality"].astype(str)
    events["beach_id"] = pd.to_numeric(
        locality_text.str.extract(r"beach:(\d+)", expand=False),
        errors="coerce",
    )
    events["beach_name"] = events["locality"]

    # Keep only rows with valid beach IDs
    events = events.dropna(subset=["beach_id"]).copy()
    events["beach_id"] = events["beach_id"].astype(np.int64)

    # Exclude synthetic catch-all locality
    events = events[events["beach_name"].astype(str).str.strip().str.lower() != "other"].copy()

    print(f"✓ Loaded events: {len(events)} records from {events['beach_id'].nunique()} beaches")

    # Extract species from occurrenceID
    occ["species"] = occ["occurrenceID"].str.extract(r'-(.*?)_by')

    # Merge tables
    df = occ.merge(events, on="eventID", how="left")
    df = df.merge(
        meas[["occurrenceID", "measurementValue"]],
        on="occurrenceID",
        how="left"
    )

    print(f"✓ Merged occurrence + measurement data: {len(df)} observations")

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

    print(f"✓ Parsed all citizen science features")
    
    return df


def aggregate_citizen_by_beach_date(df):
    """
    Aggregate citizen science observations by (beach_id, date)
    
    Args:
        df: Citizen science DataFrame
    
    Returns:
        DataFrame with daily aggregates per beach
    """
    print(f"\n📊 Aggregating citizen science data by (beach_id, date)...")
    
    daily_citizen = df.groupby(['beach_id', 'date']).agg({
        'beach_name': 'first',
        'decimalLatitude': 'mean',
        'decimalLongitude': 'mean',
        'month': 'first',
        'day_of_month': 'first',
        'sin_month': 'first',
        'cos_month': 'first',
        'distance_from_coast': 'mean',
        'species_id': 'mean',
        'diameter_cm': 'mean',
        'bloom_intensity': 'max',
        'sting': 'max',
        'occurrenceID': 'count'
    }).reset_index()
    
    daily_citizen.rename(columns={'occurrenceID': 'observation_count'}, inplace=True)
    daily_citizen['jellyfish_present'] = (daily_citizen['bloom_intensity'] > 5).astype(int)
    
    print(f"✓ Aggregated to {len(daily_citizen)} beach-date combinations")
    print(f"  Unique beaches: {daily_citizen['beach_id'].nunique()}")
    print(f"  Date range: {daily_citizen['date'].min()} to {daily_citizen['date'].max()}")
    
    return daily_citizen


def aggregate_ims_by_beach_date(weather_df, beach_station_map, daily_citizen=None):
    """
    Aggregate IMS weather data by (beach_id, date) - SYMMETRIC WITH CITIZEN SCIENCE
    
    Maps each beach to its IMS station, then aggregates weather by beach_id and date
    
    Args:
        weather_df: Parsed weather DataFrame (one row per 3-hour observation)
        beach_station_map: Dictionary mapping beach names to station names
    
    Returns:
        DataFrame with daily weather aggregates per beach
    """
    print(f"\n📊 Aggregating IMS weather data by (beach_id, date)...")

    weather_df = weather_df.copy()

    def normalize_station_name(name):
        if pd.isna(name):
            return ""
        return ''.join(ch for ch in str(name).upper() if ch.isalpha())

    def normalize_beach_name(name):
        if pd.isna(name):
            return ""
        return ''.join(ch for ch in str(name).upper().strip() if ch.isalnum())

    weather_df['station'] = weather_df['station'].astype(str).str.strip()
    weather_df['station_norm'] = weather_df['station'].apply(normalize_station_name)

    station_to_beach_names = {}
    for beach_name_pattern, station in beach_station_map.items():
        st_norm = normalize_station_name(station)
        station_to_beach_names.setdefault(st_norm, []).append(beach_name_pattern.strip())

    citizen_beach_lookup = {}
    if daily_citizen is not None and {'beach_id', 'beach_name'}.issubset(daily_citizen.columns):
        unique_beaches = daily_citizen[['beach_id', 'beach_name']].dropna().drop_duplicates()
        for _, row in unique_beaches.iterrows():
            norm_name = normalize_beach_name(row['beach_name'])
            citizen_beach_lookup.setdefault(norm_name, []).append((int(row['beach_id']), row['beach_name']))

    mapped_station_rows = []
    unique_station_norms = weather_df['station_norm'].dropna().unique().tolist()
    total_target_beaches = set()

    for st_norm in unique_station_norms:
        station_rows = weather_df[weather_df['station_norm'] == st_norm]
        station_display = station_rows['station'].iloc[0] if not station_rows.empty else st_norm

        mapped_beach_names = station_to_beach_names.get(st_norm, [])
        if not mapped_beach_names:
            print(f"⚠️  Station {station_display} not found in beach_station_map")
            continue

        station_targets = []
        for mapped_beach_name in mapped_beach_names:
            beach_matches = citizen_beach_lookup.get(normalize_beach_name(mapped_beach_name), [])
            for beach_id, citizen_beach_name in beach_matches:
                station_targets.append((beach_id, citizen_beach_name))

        if not station_targets:
            print(f"⚠️  Station {station_display} mapped beaches not found in citizen data: {mapped_beach_names}")
            continue

        station_targets = list(dict.fromkeys(station_targets))
        total_target_beaches.update(station_targets)
        print(f"  Station {station_display} mapped to {len(station_targets)} beach(es)")

        for beach_id, beach_name in station_targets:
            station_beach_rows = station_rows.copy()
            station_beach_rows['beach_id'] = beach_id
            station_beach_rows['beach_name'] = beach_name
            mapped_station_rows.append(station_beach_rows)

    if not mapped_station_rows:
        print("⚠️  No IMS station rows were mapped to beaches")
        return pd.DataFrame(columns=['beach_id', 'date'])

    weather_df = pd.concat(mapped_station_rows, ignore_index=True)

    # IMS export may not include station coordinates; keep schema stable for aggregation
    if 'decimalLatitude' not in weather_df.columns:
        weather_df['decimalLatitude'] = np.nan
    if 'decimalLongitude' not in weather_df.columns:
        weather_df['decimalLongitude'] = np.nan
    
    # Aggregate by (beach_id, date) - SAME STRUCTURE AS CITIZEN SCIENCE
    daily_weather = weather_df.groupby(['beach_id', 'date']).agg({
        'beach_name': 'first',
        'station': 'first',
        'decimalLatitude': 'first',  # Weather stations have fixed coordinates - use first
        'decimalLongitude': 'first',
        'month': 'first',
        'day_of_month': 'first',
        'sin_month': 'first',
        'cos_month': 'first',
        'Temperature_C': ['min', 'mean', 'max', 'std'],
        'Wet_Temp_C': ['min', 'mean', 'max'],
        'Dew_Point_C': ['mean'],
        'Humidity_percent': ['min', 'mean', 'max'],
        'Wind_Direction_deg': ['mean'],
        'Wind_Speed_ms': ['min', 'mean', 'max', 'std'],
        'Pressure_Sea_hPa': ['min', 'mean', 'max'],
        'Total_Clouds': ['mean'],
        'Low_Clouds': ['mean'],
        'Visibility': ['mean'],
        'DateTime': 'count'  # Count observations per day
    }).reset_index()
    
    # Flatten column names
    daily_weather.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                            for col in daily_weather.columns.values]
    
    # Rename count column
    daily_weather.rename(columns={'DateTime_count': 'observation_count'}, inplace=True)

    if 'station_norm' in daily_weather.columns:
        daily_weather = daily_weather.drop(columns=['station_norm'])
    
    print(f"✓ Aggregated to {len(daily_weather)} beach-date combinations")
    print(f"  Stations mapped: {len(unique_station_norms)}")
    print(f"  Beaches covered: {len(total_target_beaches)}")
    print(f"  Date range: {daily_weather['date'].min()} to {daily_weather['date'].max()}")
    
    return daily_weather


def merge_citizen_and_weather(daily_citizen, daily_weather):
    """
    Merge citizen science and weather data on (beach_id, date)
    
    Args:
        daily_citizen: Daily citizen science DataFrame
        daily_weather: Daily weather DataFrame
    
    Returns:
        Merged DataFrame
    """
    print(f"\n🔗 Merging citizen science and weather data...")
    
    # Merge on beach_id and date
    merged = daily_citizen.merge(
        daily_weather,
        on=['beach_id', 'date'],
        how='left',
        suffixes=('_citizen', '_weather')
    )
    
    print(f"✓ Merged: {len(merged)} records")
    weather_obs_col = None
    for candidate in ['observation_count_weather', 'observation_count', 'DateTime_count']:
        if candidate in merged.columns:
            weather_obs_col = candidate
            break

    if weather_obs_col is not None:
        print(f"  Citizen-only records: {merged[merged[weather_obs_col].isna()].shape[0]}")
        print(f"  Weather-enriched records: {merged[merged[weather_obs_col].notna()].shape[0]}")
    else:
        print("  Citizen-only records: N/A (weather observation count column missing)")
        print("  Weather-enriched records: N/A (weather observation count column missing)")
    
    return merged


def create_feature_sequences(merged_data, lookback_days=7, forecast_days=1):
    """
    Create feature sequences from merged data
    
    Args:
        merged_data: Merged DataFrame with citizen + weather data
        lookback_days: Number of historical days (default 7)
        forecast_days: Days ahead to forecast (default 1)
    
    Returns:
        X: Feature sequences (n_samples, lookback_days, n_features)
        y: Binary labels (n_samples,)
        metadata: DataFrame with sequence metadata
        feature_cols: List of feature column names
    """
    print(f"\n📈 Creating feature sequences...")

    merged_data = merged_data.copy()

    if 'observation_count' not in merged_data.columns and 'observation_count_citizen' in merged_data.columns:
        merged_data['observation_count'] = merged_data['observation_count_citizen']

    # Define citizen science features
    citizen_features = [
        'decimalLatitude', 'decimalLongitude', 'month', 'day_of_month',
        'sin_month', 'cos_month', 'distance_from_coast', 'species_id',
        'diameter_cm', 'observation_count', 'sting'
    ]
    
    # Define weather features (all aggregated columns from daily_weather)
    weather_features = [col for col in merged_data.columns 
                       if any(x in col for x in ['Temperature', 'Humidity', 'Wind', 'Pressure', 
                                                  'Dew', 'Clouds', 'Visibility', 'Wet_Temp'])
                       and col not in citizen_features]
    
    # Remove the _weather suffix if present from merged columns
    weather_features = [col for col in weather_features if not col.endswith('_citizen')]
    
    all_features = citizen_features + weather_features
    
    print(f"  Citizen science features: {len(citizen_features)}")
    print(f"  Weather features: {len(weather_features)}")
    print(f"  Total features per day: {len(all_features)}")
    
    # Sort by beach and date
    merged_data = merged_data.sort_values(['beach_id', 'date']).reset_index(drop=True)
    
    X_sequences = []
    y_labels = []
    metadata_list = []
    
    unique_beaches = merged_data['beach_id'].unique()
    
    for beach_id in unique_beaches:
        beach_data = merged_data[merged_data['beach_id'] == beach_id].reset_index(drop=True)
        
        if len(beach_data) < lookback_days + forecast_days:
            continue
        
        for i in range(len(beach_data) - lookback_days - forecast_days + 1):
            # Historical window
            historical = beach_data.iloc[i:i+lookback_days][all_features].fillna(0).values.astype(np.float32)
            
            # Future label
            future_idx = i + lookback_days + forecast_days - 1
            future_label = beach_data.iloc[future_idx]['jellyfish_present']
            future_date = beach_data.iloc[future_idx]['date']
            beach_name = beach_data.iloc[future_idx]['beach_name']
            
            X_sequences.append(historical)
            y_labels.append(future_label)
            
            metadata_list.append({
                'beach_id': beach_id,
                'beach_name': beach_name,
                'forecast_date': future_date,
                'latitude': beach_data.iloc[future_idx]['decimalLatitude'],
                'longitude': beach_data.iloc[future_idx]['decimalLongitude'],
                'lookback_start': beach_data.iloc[i]['date'],
                'lookback_end': beach_data.iloc[i+lookback_days-1]['date'],
                'jellyfish_observed': int(future_label)
            })
    
    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_labels, dtype=np.float32)
    metadata = pd.DataFrame(metadata_list)
    
    return X, y, metadata, all_features


def load_integrated_data(weather_csv_path, lookback_days=7, forecast_days=1):
    """
    Master function to load and integrate all data with SYMMETRIC structure
    
    Args:
        weather_csv_path: Path to IMS weather CSV
        lookback_days: Historical window (default 7 days)
        forecast_days: Forecast horizon (default 1 day)
    
    Returns:
        X: Feature sequences
        y: Labels
        metadata: Sequence metadata
        feature_cols: List of all feature names
        daily_citizen: Daily citizen science DataFrame
        daily_weather: Daily weather DataFrame
        merged: Merged DataFrame
    """
    print("=" * 80)
    print("INTEGRATED DATA LOADING - SYMMETRIC AGGREGATION")
    print("=" * 80)
    
    # Load and aggregate citizen science data
    df_citizen = load_citizen_science_data()
    daily_citizen = aggregate_citizen_by_beach_date(df_citizen)
    
    # Load and aggregate IMS weather data (SAME STRUCTURE)
    weather_df = load_and_parse_ims_weather(weather_csv_path)
    
    if weather_df is None:
        print("❌ Failed to load weather data")
        return None
    
    daily_weather = aggregate_ims_by_beach_date(weather_df, beach_station_map, daily_citizen)
    
    # Merge on (beach_id, date)
    merged = merge_citizen_and_weather(daily_citizen, daily_weather)
    
    # Create sequences
    X, y, metadata, feature_cols = create_feature_sequences(
        merged, lookback_days, forecast_days
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("FINAL DATASET SUMMARY")
    print("=" * 80)
    print(f"Total sequences:                {len(X)}")
    print(f"Input shape:                    {X.shape} (samples, days, features)")
    print(f"Output shape:                   {y.shape}")
    print(f"")
    print(f"Total features per day:         {X.shape[2]}")
    print(f"")
    print(f"Positive samples (jellyfish):   {int(y.sum())} ({y.mean()*100:.1f}%)")
    print(f"Negative samples:               {len(y) - int(y.sum())} ({(1-y.mean())*100:.1f}%)")
    print(f"")
    print(f"Unique beaches:                 {metadata['beach_id'].nunique()}")
    print(f"Date range:                     {metadata['lookback_start'].min()} to {metadata['forecast_date'].max()}")
    print(f"=" * 80)
    
    return X, y, metadata, feature_cols, daily_citizen, daily_weather, merged


if __name__ == "__main__":
    # Example usage
    weather_path = "data/IMS/data_202603142120.csv"
    
    results = load_integrated_data(
        weather_csv_path=weather_path,
        lookback_days=7,
        forecast_days=1
    )
    
    if results:
        X, y, metadata, feature_cols, daily_citizen, daily_weather, merged = results
        
        print("\n✓ Data loaded successfully!")
        print(f"\nReady for model training:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Total features: {len(feature_cols)}")
        print(f"  Features: {feature_cols}")