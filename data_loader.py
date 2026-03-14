import pandas as pd
import numpy as np
import os

def load_all_data():

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

def load_jellyfish_data():
    data = load_all_data()

    events = data["Citizen Science based jellyfish observations along the Israeli Mediterranean coast in 2011-2025"]["event.txt"]
    occ = data["Citizen Science based jellyfish observations along the Israeli Mediterranean coast in 2011-2025"]["occurrence.txt"]
    meas = data["Citizen Science based jellyfish observations along the Israeli Mediterranean coast in 2011-2025"]["extendedmeasurementorfact.txt"]

    events["beach_id"] = events["verbatimLocality"].str.extract(r'beach:(\d+)').astype(int)

    events["beach_name"] = events["locality"]

    # Extract species from occurrenceID
    occ["species"] = occ["occurrenceID"].str.extract(r'-(.*?)_by')

    # Merge tables
    df = occ.merge(events, on="eventID", how="left")
    df = df.merge(
        meas[["occurrenceID", "measurementValue"]],
        on="occurrenceID",
        how="left"
    )

    # -----------------------------
    # Parse diameter
    # -----------------------------

    def diameter_to_float(v):
        if pd.isna(v):
            return np.nan
        text = str(v).strip()
        if not text:
            return np.nan

        lowered = text.lower()
        if lowered in {"unspecified", "unknown", "nan", "none", "na", "n/a"}:
            return np.nan

        # Keep digits, decimal point, minus sign and range separator
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

    # -----------------------------
    # Parse time
    # -----------------------------

    dt = pd.to_datetime(df["eventDate"])

    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour

    df["sin_month"] = np.sin(2*np.pi*df["month"]/12)
    df["cos_month"] = np.cos(2*np.pi*df["month"]/12)

    # -----------------------------
    # Parse distance from coast
    # -----------------------------

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

    # -----------------------------
    # Parse sting reports
    # -----------------------------

    def sting_flag(x):
        if pd.isna(x):
            return 0
        return int("Yes" in x)

    df["sting"] = df["locationRemarks"].apply(sting_flag)

    # -----------------------------
    # Encode categorical variables
    # -----------------------------

    df["species_id"] = df["species"].astype("category").cat.codes
    df["protocol_id"] = df["samplingProtocol"].astype("category").cat.codes

    # -----------------------------
    # Bloom intensity
    # -----------------------------

    df["bloom_intensity"] = df["organismQuantity"].map({
        "Few":2,
        "Some":10,
        "Swarm":50
    }).fillna(df["individualCount"])

    # -----------------------------
    # Build feature matrix
    # -----------------------------

    feature_cols = [
        "decimalLatitude",
        "decimalLongitude",
        "month",
        "day",
        "hour",
        "sin_month",
        "cos_month",
        "distance_from_coast",
        "species_id",
        "diameter_cm",
        "bloom_intensity",
        "sting",
        "protocol_id"
    ]

    X = df[feature_cols].fillna(0).values.astype(np.float32)

    # Label: presence of jellyfish bloom
    y = (df["bloom_intensity"] > 5).astype(np.float32).values

    return X, y