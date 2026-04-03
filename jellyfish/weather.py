"""
Live weather data from the Israeli Meteorological Service (IMS) RSS feeds.
"""

import re
from html import unescape
from xml.etree import ElementTree as ET

import requests


class IMSWeatherFetcher:
    """Fetch and parse IMS coastal weather forecasts from RSS XML feeds."""

    RSS_FEEDS = {
        "northern_coast": "http://www.ims.gov.il/IMSEnglish/Rss/RssNorthernCoastForecast.xml",
        "central_coast": "http://www.ims.gov.il/IMSEnglish/Rss/RssCentralCoastForecast.xml",
        "southern_coast": "http://www.ims.gov.il/IMSEnglish/Rss/RssSouthernCoastForecast.xml",
    }

    def __init__(self, region: str = "central_coast"):
        if region not in self.RSS_FEEDS:
            raise ValueError(f"Unknown region '{region}'. Choose from: {list(self.RSS_FEEDS)}")
        self.region = region
        self.rss_url = self.RSS_FEEDS[region]

    def fetch_forecast(self) -> dict | None:
        try:
            response = requests.get(self.rss_url, timeout=10)
            response.raise_for_status()
            return self._parse_rss(response.text)
        except Exception as exc:
            print(f"[IMSWeatherFetcher] Failed to fetch {self.region}: {exc}")
            return None

    def format_forecast(self, forecast_data: dict | None = None) -> str:
        if forecast_data is None:
            forecast_data = self.fetch_forecast()
        if not forecast_data:
            return "Failed to fetch forecast — check internet connection."

        lines = [
            "=" * 70,
            f"IMS WEATHER FORECAST — {forecast_data['region'].upper().replace('_', ' ')}",
            f"Last Update : {forecast_data['last_update']}",
            "=" * 70,
        ]
        for i, fc in enumerate(forecast_data["forecasts"], 1):
            lines += [
                f"\nWindow {i}: {fc['start_time']}  →  {fc['end_time']}",
                f"  Sea Temperature : {fc['temperature_c']}°C",
                f"  Wind Speed      : {fc['wind_speed_kmh_min']}–{fc['wind_speed_kmh_max']} km/h",
                f"  Wind Direction  : {fc['wind_direction']}",
                f"  Wave Height     : {fc['waves_height_cm_min']}–{fc['waves_height_cm_max']} cm",
                f"  Sea State       : {fc['sea_state']}",
            ]
        return "\n".join(lines)

    def _parse_rss(self, xml_string: str) -> dict | None:
        try:
            root = ET.fromstring(xml_string)
            desc_elem = root.find(".//description")
            if desc_elem is None or not desc_elem.text:
                return None

            desc = unescape(desc_elem.text)
            update_match = re.search(r"last update: ([\d\-\s:]+)", desc)

            return {
                "region": self.region,
                "last_update": update_match.group(1).strip() if update_match else None,
                "forecasts": self._parse_forecast_windows(desc),
            }
        except Exception as exc:
            print(f"[IMSWeatherFetcher] RSS parse error: {exc}")
            return None

    def _parse_forecast_windows(self, description: str) -> list[dict]:
        forecasts = []
        pattern = r"From ([\d\-\s:]+) to ([\d\-\s:]+)"
        starts = [m.start() for m in re.finditer(pattern, description)]

        for i, match in enumerate(re.finditer(pattern, description)):
            window_end = starts[i + 1] if i + 1 < len(starts) else len(description)
            window_text = description[match.end(): window_end]
            fc = self._extract_weather_values(window_text, match.group(1), match.group(2))
            if fc:
                forecasts.append(fc)
        return forecasts

    def _extract_weather_values(self, text: str, start_time: str, end_time: str) -> dict:
        fc = {
            "start_time": start_time.strip(),
            "end_time": end_time.strip(),
            "temperature_c": None,
            "wind_speed_kmh_min": None,
            "wind_speed_kmh_max": None,
            "wind_direction": None,
            "waves_height_cm_min": None,
            "waves_height_cm_max": None,
            "sea_state": None,
        }
        m = re.search(r"Sea temperture \(°C\): (\d+)", text)
        if m:
            fc["temperature_c"] = int(m.group(1))

        m = re.search(r"Wind speed \(km/h\): (\d+)\s+to\s+(\d+)", text)
        if m:
            fc["wind_speed_kmh_min"] = int(m.group(1))
            fc["wind_speed_kmh_max"] = int(m.group(2))

        m = re.search(r"Wind direction: ([^<\n]+)", text)
        if m:
            fc["wind_direction"] = m.group(1).strip()

        m = re.search(r"Waves height \(cm\): (\d+)\s+to\s+(\d+)", text)
        if m:
            fc["waves_height_cm_min"] = int(m.group(1))
            fc["waves_height_cm_max"] = int(m.group(2))

        m = re.search(r"Sea state: ([^<\n]+)", text)
        if m:
            fc["sea_state"] = m.group(1).strip()

        return fc
