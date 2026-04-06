import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import re

try:
    from .weather import IMSWeatherFetcher
except ImportError:
    from weather import IMSWeatherFetcher

try:
    from .terminal_format import section, rule
except ImportError:
    from terminal_format import section, rule

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


def _extract_numeric_values(text):
    if text is None:
        return []
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    return [float(x) for x in matches]


def _parse_pubdate_to_date(value):
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


CITY_REFERENCE_COORDS = {
    "ashdod": (31.8044, 34.6553),
    "haifa": (32.7940, 34.9896),
    "tel_aviv_coast": (32.0853, 34.7818),
}

SEA_REGION_REFERENCE_COORDS = {
    "northern_coast": (32.95, 35.08),
    "central_coast": (32.10, 34.78),
    "southern_coast": (31.55, 34.60),
}

SEA_TO_ALERT_REGION = {
    "northern_coast": "north",
    "central_coast": "center",
    "southern_coast": "south",
}

# Region latitude boundaries for dynamic mapping (based on reference coords and Israeli coast geography)
REGION_LATITUDE_BOUNDARIES = {
    "northern_coast": (32.80, 90.0),      # Northern: 32.80°N and above (Haifa to Nahariya area)
    "central_coast": (31.77, 32.80),      # Central: 31.77°N to 32.80°N (Ashdod to Haifa area)
    "southern_coast": (-90.0, 31.77),     # Southern: below 31.77°N (Ashkelon, Gaza area)
}


def _haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan

    r = 6371.0
    phi1 = np.radians(float(lat1))
    phi2 = np.radians(float(lat2))
    dphi = np.radians(float(lat2) - float(lat1))
    dlambda = np.radians(float(lon2) - float(lon1))

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def get_region_from_latitude(latitude):
    """
    Determine sea region (northern/central/southern) from latitude.
    
    Uses latitude boundaries based on Israeli Mediterranean coast geography.
    
    Args:
        latitude: Beach latitude
    
    Returns:
        Region name: "northern_coast", "central_coast", or "southern_coast"
    """
    if pd.isna(latitude):
        return "central_coast"  # Default fallback
    
    lat = float(latitude)
    for region, (lat_min, lat_max) in REGION_LATITUDE_BOUNDARIES.items():
        if lat_min <= lat <= lat_max:
            return region
    
    # Fallback: if outside defined boundaries, assign to nearest
    if lat > 32.80:
        return "northern_coast"
    elif lat < 31.77:
        return "southern_coast"
    else:
        return "central_coast"


def build_beach_live_source_map(daily_citizen, available_city_sources=None):
    """Map each beach to nearest live RSS source locations (city/coast/alert region)."""
    required = {"beach_id", "beach_name", "decimalLatitude", "decimalLongitude"}
    if daily_citizen is None or not required.issubset(daily_citizen.columns):
        return pd.DataFrame(columns=[
            "beach_id", "beach_name", "decimalLatitude", "decimalLongitude",
            "city_source", "city_distance_km", "sea_source", "alert_source",
        ])

    beach_points = (
        daily_citizen[["beach_id", "beach_name", "decimalLatitude", "decimalLongitude"]]
        .dropna(subset=["beach_id", "decimalLatitude", "decimalLongitude"])
        .groupby("beach_id", as_index=False)
        .agg({
            "beach_name": "first",
            "decimalLatitude": "mean",
            "decimalLongitude": "mean",
        })
    )

    city_reference_coords = CITY_REFERENCE_COORDS.copy()
    if available_city_sources is not None:
        allowed = set(available_city_sources)
        filtered = {k: v for k, v in city_reference_coords.items() if k in allowed}
        if filtered:
            city_reference_coords = filtered

    mapped_rows = []
    for _, row in beach_points.iterrows():
        lat = float(row["decimalLatitude"])
        lon = float(row["decimalLongitude"])

        city_distances = {
            city: _haversine_km(lat, lon, coords[0], coords[1])
            for city, coords in city_reference_coords.items()
        }
        city_source = min(city_distances, key=city_distances.get)

        sea_distances = {
            coast: _haversine_km(lat, lon, coords[0], coords[1])
            for coast, coords in SEA_REGION_REFERENCE_COORDS.items()
        }
        sea_source = min(sea_distances, key=sea_distances.get)

        mapped_rows.append({
            "beach_id": int(row["beach_id"]),
            "beach_name": row["beach_name"],
            "decimalLatitude": lat,
            "decimalLongitude": lon,
            "city_source": city_source,
            "city_distance_km": float(city_distances[city_source]),
            "sea_source": sea_source,
            "alert_source": SEA_TO_ALERT_REGION.get(sea_source, "center"),
        })

    beach_source_map = pd.DataFrame(mapped_rows)
    if not beach_source_map.empty:
        print(f"✓ Built nearest-source map for {len(beach_source_map)} beaches")

    return beach_source_map


def map_live_rss_features_to_beaches(live_daily, daily_citizen):
    """Assign live RSS features to each beach-date using nearest source mapping."""
    if live_daily is None or live_daily.empty or "date" not in live_daily.columns:
        return pd.DataFrame(columns=["beach_id", "date"])

    available_city_sources = sorted({
        re.sub(r"^city_", "", c).replace("_tonight_min_temp_c", "")
        for c in live_daily.columns
        if c.startswith("city_") and c.endswith("_tonight_min_temp_c")
    })

    beach_source_map = build_beach_live_source_map(daily_citizen)
    if beach_source_map.empty:
        return pd.DataFrame(columns=["beach_id", "date"])

    mapped_frames = []
    radiation_cols = [c for c in live_daily.columns if c.startswith("radiation_")]

    for _, beach in beach_source_map.iterrows():
        city_source = beach["city_source"]
        sea_source = beach["sea_source"]
        alert_source = beach["alert_source"]

        city_feature_source = city_source
        if city_feature_source not in available_city_sources and available_city_sources:
            ref_lat, ref_lon = CITY_REFERENCE_COORDS.get(city_source, (np.nan, np.nan))
            nearest_available = min(
                available_city_sources,
                key=lambda c: _haversine_km(
                    ref_lat,
                    ref_lon,
                    CITY_REFERENCE_COORDS.get(c, (np.nan, np.nan))[0],
                    CITY_REFERENCE_COORDS.get(c, (np.nan, np.nan))[1],
                ),
            )
            city_feature_source = nearest_available

        beach_daily = live_daily[["date"]].copy()
        beach_daily["beach_id"] = int(beach["beach_id"])

        beach_daily["rss_city_source"] = city_source
        beach_daily["rss_city_feature_source"] = city_feature_source
        beach_daily["rss_city_distance_km"] = beach["city_distance_km"]
        beach_daily["rss_sea_source"] = sea_source
        beach_daily["rss_alert_source"] = alert_source

        beach_daily["rss_city_tonight_min_temp_c"] = live_daily.get(f"city_{city_feature_source}_tonight_min_temp_c")
        beach_daily["rss_city_max_temp_c"] = live_daily.get(f"city_{city_feature_source}_max_temp_c")
        beach_daily["rss_city_min_temp_c"] = live_daily.get(f"city_{city_feature_source}_min_temp_c")

        beach_daily["rss_sea_temperature_c"] = live_daily.get(f"sea_{sea_source}_temperature_c")
        beach_daily["rss_sea_wind_speed_kmh_min"] = live_daily.get(f"sea_{sea_source}_wind_speed_kmh_min")
        beach_daily["rss_sea_wind_speed_kmh_max"] = live_daily.get(f"sea_{sea_source}_wind_speed_kmh_max")
        beach_daily["rss_sea_waves_height_cm_min"] = live_daily.get(f"sea_{sea_source}_waves_height_cm_min")
        beach_daily["rss_sea_waves_height_cm_max"] = live_daily.get(f"sea_{sea_source}_waves_height_cm_max")

        beach_daily["rss_flood_alert_count"] = live_daily.get(f"flood_alert_{alert_source}_count")
        beach_daily["rss_flood_alert_active"] = live_daily.get(f"flood_alert_{alert_source}_active")

        for col in radiation_cols:
            beach_daily[f"rss_{col}"] = live_daily[col]

        mapped_frames.append(beach_daily)

    mapped_live = pd.concat(mapped_frames, ignore_index=True)
    print(f"✓ Mapped live RSS features to beach-date rows: {len(mapped_live)}")
    return mapped_live


def load_live_ims_xml_features(region=None, beach_locations_df=None):
    """Load live IMS XML forecasts aggregated by region.
    
    Args:
        region: Specific region to load (None = auto-detect from beach_locations_df, 
                "all" = load all regions)
        beach_locations_df: DataFrame with beach locations to auto-detect regions
    
    Returns:
        DataFrame with one row per date and globally aggregated weather/sea/radiation
        features from IMS XML feeds.
    """
    print("\n🌐 Loading live IMS RSS feeds...")
    
    # Determine which feed strategy to use.
    # For integrated beach data we want the actual coast RSS feeds plus the
    # flood-alert feeds, not city forecasts as a substitute for coast data.
    fetcher_region = region if region in REGION_LATITUDE_BOUNDARIES else "central_coast"
    if region is None and beach_locations_df is not None and not beach_locations_df.empty:
        required_cols = {"decimalLatitude", "beach_id"}
        if required_cols.issubset(beach_locations_df.columns):
            unique_beaches = beach_locations_df[["beach_id", "decimalLatitude"]].drop_duplicates()
            detected_regions = sorted({get_region_from_latitude(row["decimalLatitude"]) for _, row in unique_beaches.iterrows()})
            print(f"  Auto-detected coast regions from beach locations: {detected_regions}")
        else:
            print("  ⚠️  beach_locations_df missing required columns, using central_coast")

    if region == "all" or region is None:
        print("  Fetching all coast RSS feeds plus flood alerts and radiation feeds")
    else:
        print(f"  Fetching specified coast RSS feed: {fetcher_region} (plus flood alerts/radiation)")

    try:
        fetcher = IMSWeatherFetcher(region=fetcher_region)
        payload = fetcher.fetch_enriched_forecast(
            fetch_all_sea_regions=True,
            include_global_feeds=True,
        )
        all_frames = [_parse_live_ims_payload(payload)]
    except Exception as e:
        print(f"    ⚠️  Failed to fetch live RSS data: {e}")
        all_frames = []
    
    if not all_frames:
        print("⚠️  Live RSS feeds unavailable; continuing without live RSS features")
        return pd.DataFrame(columns=["date"])
    
    # Merge all region data on date
    live_daily = all_frames[0]
    for frame in all_frames[1:]:
        if not frame.empty and not live_daily.empty and 'date' in frame.columns and 'date' in live_daily.columns:
            live_daily = live_daily.merge(frame, on="date", how="outer")
    
    print(f"✓ Live RSS daily features loaded: {len(live_daily)} date rows from coast RSS feeds")
    return live_daily


def _parse_live_ims_payload(payload):
    """Parse IMS XML payload into daily features (helper for load_live_ims_xml_features)."""
    city_rows = []
    city_payload = payload.get("city_rss", {}) if payload else {}
    for city_name, city_data in city_payload.items():
        if not city_data:
            continue

        base_date = _parse_pubdate_to_date(city_data.get("pub_date"))
        city_rows.append({
            "date": base_date,
            f"city_{city_name}_tonight_min_temp_c": pd.to_numeric(city_data.get("tonight_min_temp_c"), errors="coerce"),
        })

        for fc in city_data.get("daily_forecasts", []):
            ddmm = fc.get("date_ddmm")
            if not ddmm or base_date is None:
                continue
            try:
                day, month = ddmm.split("/")
                year = base_date.year
                forecast_date = pd.Timestamp(year=year, month=int(month), day=int(day)).date()
            except Exception:
                continue

            city_rows.append({
                "date": forecast_date,
                f"city_{city_name}_max_temp_c": pd.to_numeric(fc.get("max_temp_c"), errors="coerce"),
                f"city_{city_name}_min_temp_c": pd.to_numeric(fc.get("min_temp_c"), errors="coerce"),
            })

    city_daily = pd.DataFrame(city_rows)
    if not city_daily.empty:
        value_cols = [col for col in city_daily.columns if col != "date"]
        city_daily = city_daily.groupby("date", as_index=False)[value_cols].mean(numeric_only=True)

    sea_rows = []
    sea_payload = payload.get("sea_rss", {}) if payload else {}
    for coast_region, sea_data in sea_payload.items():
        if not sea_data:
            continue

        for fc in sea_data.get("forecasts", []):
            dt = pd.to_datetime(fc.get("start_time"), errors="coerce")
            if pd.isna(dt):
                dt = pd.to_datetime(sea_data.get("pub_date"), errors="coerce")
            if pd.isna(dt):
                continue

            sea_rows.append({
                "date": dt.date(),
                f"sea_{coast_region}_temperature_c": pd.to_numeric(fc.get("temperature_c"), errors="coerce"),
                f"sea_{coast_region}_wind_speed_kmh_min": pd.to_numeric(fc.get("wind_speed_kmh_min"), errors="coerce"),
                f"sea_{coast_region}_wind_speed_kmh_max": pd.to_numeric(fc.get("wind_speed_kmh_max"), errors="coerce"),
                f"sea_{coast_region}_waves_height_cm_min": pd.to_numeric(fc.get("waves_height_cm_min"), errors="coerce"),
                f"sea_{coast_region}_waves_height_cm_max": pd.to_numeric(fc.get("waves_height_cm_max"), errors="coerce"),
            })

    sea_daily = pd.DataFrame(sea_rows)
    if not sea_daily.empty:
        value_cols = [col for col in sea_daily.columns if col != "date"]
        sea_daily = sea_daily.groupby("date", as_index=False)[value_cols].mean(numeric_only=True)

    rad_rows = []
    rad_payload = payload.get("radiation_rss") if payload else None
    if rad_payload:
        rad_date = _parse_pubdate_to_date(rad_payload.get("pub_date"))
        rad_rows.append({
            "date": rad_date,
            "radiation_city_mentions": pd.to_numeric(rad_payload.get("city_mentions"), errors="coerce"),
            "radiation_low_mentions": pd.to_numeric(rad_payload.get("low_mentions"), errors="coerce"),
            "radiation_medium_mentions": pd.to_numeric(rad_payload.get("medium_mentions"), errors="coerce"),
            "radiation_high_mentions": pd.to_numeric(rad_payload.get("high_mentions"), errors="coerce"),
            "radiation_very_high_mentions": pd.to_numeric(rad_payload.get("very_high_mentions"), errors="coerce"),
        })

    rad_daily = pd.DataFrame(rad_rows)
    if not rad_daily.empty:
        rad_daily = rad_daily.dropna(subset=["date"])

    alert_rows = []
    alerts_payload = payload.get("alerts_rss", {}) if payload else {}
    for alert_region, alert_data in alerts_payload.items():
        if not alert_data:
            continue
        alert_date = _parse_pubdate_to_date(alert_data.get("last_build_date"))
        alert_rows.append({
            "date": alert_date,
            f"flood_alert_{alert_region}_count": pd.to_numeric(alert_data.get("item_count"), errors="coerce"),
            f"flood_alert_{alert_region}_active": int(bool(alert_data.get("active"))),
        })

    alert_daily = pd.DataFrame(alert_rows)
    if not alert_daily.empty:
        value_cols = [col for col in alert_daily.columns if col != "date"]
        alert_daily = alert_daily.groupby("date", as_index=False)[value_cols].mean(numeric_only=True)
    
    # Combine all frames
    daily_frames = []
    for frame in [city_daily, sea_daily, rad_daily, alert_daily]:
        if frame is not None and not frame.empty and "date" in frame.columns:
            frame = frame.dropna(subset=["date"])
            daily_frames.append(frame)

    if not daily_frames:
        return pd.DataFrame(columns=["date"])

    live_daily = daily_frames[0]
    for frame in daily_frames[1:]:
        live_daily = live_daily.merge(frame, on="date", how="outer")

    return live_daily


# Remove the old duplicate code that was here
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


def find_all_ims_csv_files(directory="data/IMS"):
    """
    Find all IMS weather CSV files in directory, sorted by filename (newest last).
    
    Args:
        directory: Path to IMS data directory
    
    Returns:
        List of full paths to IMS CSV files, sorted (newest last)
    """
    if not os.path.isdir(directory):
        return []
    
    csv_files = [f for f in os.listdir(directory) if f.startswith("data_") and f.endswith(".csv")]
    csv_files.sort()  # Numeric sort by filename (newer files named later)
    
    return [os.path.join(directory, f) for f in csv_files]


def load_and_consolidate_ims_weather(weather_csv_path=None):
    """
    Load IMS weather data, optionally consolidating multiple CSV files.
    
    If weather_csv_path is None, finds all available IMS CSV files and loads them.
    If weight_csv_path is a single file, loads just that file.
    If weather_csv_path is a list, consolidates all files in the list.
    
    Args:
        weather_csv_path: Path(s) to IMS weather CSV file(s), or None to auto-discover all
    
    Returns:
        Consolidated weather DataFrame or None if loading fails
    """
    files_to_load = []
    
    if weather_csv_path is None:
        # Auto-discover all IMS CSV files
        files_to_load = find_all_ims_csv_files()
        if not files_to_load:
            print("❌ No IMS CSV files found in data/IMS directory")
            return None
        print(f"Auto-discovered {len(files_to_load)} IMS CSV file(s):")
        for f in files_to_load:
            print(f"  - {os.path.basename(f)}")
    elif isinstance(weather_csv_path, str):
        # Single file path
        if os.path.exists(weather_csv_path):
            files_to_load = [weather_csv_path]
        else:
            print(f"❌ Weather file not found at {weather_csv_path}")
            return None
    elif isinstance(weather_csv_path, list):
        # Multiple file paths
        files_to_load = weather_csv_path
    else:
        print(f"❌ Invalid weather_csv_path type: {type(weather_csv_path)}")
        return None
    
    if len(files_to_load) > 1:
        source_label = "data/IMS" if weather_csv_path is None else f"{len(files_to_load)} file(s)"
        print(f"\n📡 Loading IMS weather data from {source_label}...")

    dfs = []
    for fpath in files_to_load:
        df = load_and_parse_ims_weather(fpath, verbose=(len(files_to_load) == 1))
        if df is not None:
            dfs.append(df)
    
    if not dfs:
        print("❌ Failed to load any weather data files")
        return None
    
    if len(dfs) == 1:
        return dfs[0]
    
    # Consolidate multiple files: deduplicate by (station, date, hour)
    consolidated = pd.concat(dfs, ignore_index=True)
    print(f"✓ Raw weather data loaded: {len(consolidated)} records")
    
    # Remove exact duplicates
    consol_before = len(consolidated)
    consolidated = consolidated.drop_duplicates(subset=['station', 'date', 'hour'], keep='first')
    dupes_removed = consol_before - len(consolidated)
    if dupes_removed > 0:
        print(f"  Removed {dupes_removed} duplicate records")

    print(f"✓ Parsed weather data: {len(consolidated)} observations")
    unique_stations = consolidated['station'].dropna().astype(str).str.strip().replace('', np.nan).dropna().unique().tolist()
    print(f"  Unique stations: {len(unique_stations)}")
    if unique_stations:
        preview = ', '.join(unique_stations[:5])
        if len(unique_stations) > 5:
            preview += ", ..."
        print(f"  Stations sample: {preview}")

    print(f"  Date range: {consolidated['date'].min()} to {consolidated['date'].max()}")
    print(f"  Temperature: {consolidated['Temperature_C'].min():.1f}°C to {consolidated['Temperature_C'].max():.1f}°C")
    print(f"  Wind speed: {consolidated['Wind_Speed_ms'].min():.1f} to {consolidated['Wind_Speed_ms'].max():.1f} m/s")

    return consolidated


def load_and_parse_ims_weather(weather_csv_path, verbose=True):
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
    
    if verbose:
        print(f"\n📡 Loading IMS weather data from {weather_csv_path}...")
    
    try:
        weather = pd.read_csv(weather_csv_path)
        weather.columns = weather.columns.str.strip()
        
        if verbose:
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
        
        if verbose:
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


def load_integrated_data(weather_csv_path, lookback_days=7, forecast_days=1, include_live_xml=True):
    """
    Master function to load and integrate all data with SYMMETRIC structure
    
    Args:
        weather_csv_path: Path to IMS weather CSV(s). Can be:
            - None: auto-discover all IMS CSV files in data/IMS/
            - str: single CSV file path
            - list: list of CSV file paths to consolidate
        lookback_days: Historical window (default 7 days)
        forecast_days: Forecast horizon (default 1 day)
        include_live_xml: Whether to include live IMS RSS features (default True)
    
    Returns:
        X: Feature sequences
        y: Labels
        metadata: Sequence metadata
        feature_cols: List of all feature names
        daily_citizen: Daily citizen science DataFrame
        daily_weather: Daily weather DataFrame
        merged: Merged DataFrame
    """
    section("INTEGRATED DATA LOADING - SYMMETRIC AGGREGATION")
    
    # Load and aggregate citizen science data
    df_citizen = load_citizen_science_data()
    daily_citizen = aggregate_citizen_by_beach_date(df_citizen)
    
    # Load and aggregate IMS weather data (SAME STRUCTURE)
    # Now using load_and_consolidate_ims_weather which can handle multiple files
    weather_df = load_and_consolidate_ims_weather(weather_csv_path)
    
    if weather_df is None:
        print("❌ Failed to load weather data")
        return None
    
    daily_weather = aggregate_ims_by_beach_date(weather_df, beach_station_map, daily_citizen)

    if include_live_xml:
        # Now using dynamic region mapping based on beach locations, NO hardcoded region!
        live_xml_daily = load_live_ims_xml_features(
            region=None,  # Auto-detect regions from beach locations
            beach_locations_df=daily_citizen
        )
        if not live_xml_daily.empty and 'date' in live_xml_daily.columns:
            live_mapped = map_live_rss_features_to_beaches(live_xml_daily, daily_citizen)
            if not live_mapped.empty:
                daily_weather = daily_weather.merge(live_mapped, on=['beach_id', 'date'], how='left')
                print(f"✓ Daily weather enriched with nearest-mapped live RSS features")
    
    # Merge on (beach_id, date)
    merged = merge_citizen_and_weather(daily_citizen, daily_weather)
    
    # Create sequences
    X, y, metadata, feature_cols = create_feature_sequences(
        merged, lookback_days, forecast_days
    )
    
    # Print summary
    section("FINAL DATASET SUMMARY")
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
    print(rule("FINAL DATASET SUMMARY"))
    
    return X, y, metadata, feature_cols, daily_citizen, daily_weather, merged


if __name__ == "__main__":
    # Example usage
    # Option 1: Auto-discover all IMS CSV files
    weather_path = None
    
    # Option 2: Use a specific CSV file
    # weather_path = "data/IMS/data_202603142120.csv"
    
    # Option 3: Consolidate multiple specific CSV files
    # weather_path = ["data/IMS/data_202603142120.csv", "data/IMS/data_202604052350.csv"]
    
    results = load_integrated_data(
        weather_csv_path=weather_path,
        lookback_days=7,
        forecast_days=1,
        include_live_xml=True,
    )
    
    if results:
        X, y, metadata, feature_cols, daily_citizen, daily_weather, merged = results
        
        print("\n✓ Data loaded successfully!")
        print(f"\nReady for model training:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Total features: {len(feature_cols)}")
        print(f"  Features: {feature_cols}")