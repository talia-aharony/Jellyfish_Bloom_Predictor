"""
Live weather data from the Israeli Meteorological Service (IMS) RSS feeds.
"""

import re
from html import unescape
from xml.etree import ElementTree as ET

import requests


class IMSWeatherFetcher:
    """Fetch and parse IMS weather forecasts/alerts from RSS XML feeds."""

    RSS_FEEDS = {
        "northern_coast": "https://ims.gov.il/sites/default/files/ims_data/rss/forecast_sea/rssForecastSea_212_en.xml",
        "central_coast": "https://ims.gov.il/sites/default/files/ims_data/rss/forecast_sea/rssForecastSea_210_en.xml",
        "southern_coast": "https://ims.gov.il/sites/default/files/ims_data/rss/forecast_sea/rssForecastSea_213_en.xml",
    }

    CITY_RSS_FEEDS = {
        "ashdod": "https://ims.gov.il/sites/default/files/ims_data/rss/forecast_city/rssForecastCity_114_en.xml",
        "haifa": "https://ims.gov.il/sites/default/files/ims_data/rss/forecast_city/rssForecastCity_115_en.xml",
        "tel_aviv_coast": "https://ims.gov.il/sites/default/files/ims_data/rss/forecast_city/rssForecastCity_402_en.xml",
    }

    RADIATION_RSS_URL = "https://ims.gov.il/sites/default/files/ims_data/rss/forecast_radiation/rssForecastRadiation_en.xml"

    FLOOD_ALERT_RSS_FEEDS = {
        "north": "https://ims.gov.il/sites/default/files/ims_data/rss/alert/rssAlert_flood_north_en.xml",
        "center": "https://ims.gov.il/sites/default/files/ims_data/rss/alert/rssAlert_flood_center_en.xml",
        "south": "https://ims.gov.il/sites/default/files/ims_data/rss/alert/rssAlert_flood_south_en.xml",
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
            return self._parse_sea_rss(response.text, region=self.region)
        except Exception as exc:
            print(f"[IMSWeatherFetcher] Failed to fetch {self.region}: {exc}")
            return None

    def fetch_enriched_forecast(
        self,
        fetch_all_sea_regions: bool = True,
        include_global_feeds: bool = True,
    ) -> dict:
        """Fetch RSS coastal forecast plus supplemental RSS feeds.

        Args:
            fetch_all_sea_regions: If True, fetch all coastal sea feeds into
                ``sea_rss``. If False, only fetch the currently selected region.
            include_global_feeds: If True, fetch city/radiation/alerts feeds.
                Set False to avoid duplicate global fetches across region loops.

        Returns:
            Dictionary containing:
            - coastal_rss: Parsed sea forecast for selected region
            - sea_rss: Parsed sea forecasts for north/center/south coasts
            - city_rss: Parsed city forecasts (Ashdod, Haifa, Tel Aviv Coast)
            - radiation_rss: Parsed UV forecast summary
            - alerts_rss: Parsed flood/flash-flood warnings by region
            - tel_aviv_coast_label: label mapping provided by user
        """
        result = {
            "coastal_rss": self.fetch_forecast(),
            "sea_rss": {},
            "city_rss": {},
            "radiation_rss": None,
            "alerts_rss": {}
        }

        sea_regions = list(self.RSS_FEEDS.items())
        if not fetch_all_sea_regions:
            sea_regions = [(self.region, self.RSS_FEEDS[self.region])]

        for coast_region, url in sea_regions:
            xml_text = self._fetch_xml(url)
            if xml_text:
                result["sea_rss"][coast_region] = self._parse_sea_rss(xml_text, region=coast_region)
                print(f"[IMSWeatherFetcher] Sea RSS loaded: {coast_region}")
            else:
                print(f"[IMSWeatherFetcher] Sea RSS fetch failed: {coast_region}")

        if include_global_feeds:
            for city_name, url in self.CITY_RSS_FEEDS.items():
                xml_text = self._fetch_xml(url)
                if not xml_text:
                    print(f"[IMSWeatherFetcher] City RSS fetch failed: {city_name}")
                    continue

                parsed_city = self._parse_city_rss(xml_text, city_name=city_name)
                result["city_rss"][city_name] = parsed_city

                if parsed_city is not None:
                    print(f"[IMSWeatherFetcher] City RSS loaded: {city_name}")
                else:
                    print(f"[IMSWeatherFetcher] City RSS parse failed: {city_name}")

            radiation_xml = self._fetch_xml(self.RADIATION_RSS_URL)
            if radiation_xml:
                result["radiation_rss"] = self._parse_radiation_rss(radiation_xml)

            for alert_region, url in self.FLOOD_ALERT_RSS_FEEDS.items():
                xml_text = self._fetch_xml(url)
                if xml_text:
                    result["alerts_rss"][alert_region] = self._parse_alert_rss(xml_text, region=alert_region)
                    print(f"[IMSWeatherFetcher] Flood alert RSS loaded: {alert_region}")
                else:
                    print(f"[IMSWeatherFetcher] Flood alert RSS fetch failed: {alert_region}")

        return result

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

    def _parse_sea_rss(self, xml_string: str, region: str) -> dict | None:
        try:
            root = ET.fromstring(xml_string)
            desc_elem = root.find("./channel/item/description")
            if desc_elem is None or not desc_elem.text:
                return None

            desc = unescape(desc_elem.text)
            update_match = re.search(r"last update:\s*([\d\-\s:]+)", desc, flags=re.IGNORECASE)
            pub_date = root.findtext("./channel/item/pubDate")
            title = root.findtext("./channel/item/title")

            return {
                "region": region,
                "title": title,
                "pub_date": pub_date,
                "last_update": update_match.group(1).strip() if update_match else None,
                "forecasts": self._parse_forecast_windows(desc),
            }
        except Exception as exc:
            print(f"[IMSWeatherFetcher] Sea RSS parse error ({region}): {exc}")
            return None

    def _fetch_xml(self, url: str) -> str | None:
        try:
            response = requests.get(url, timeout=12)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            print(f"[IMSWeatherFetcher] Failed to fetch XML {url}: {exc}")
            return None

    @staticmethod
    def _coerce_number(value):
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        text = text.replace(",", "")
        try:
            if "." in text:
                return float(text)
            return int(text)
        except ValueError:
            return value

    @staticmethod
    def _normalize_key(text: str) -> str:
        key = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower())
        return key.strip("_")

    def _parse_city_rss(self, xml_string: str, city_name: str) -> dict | None:
        try:
            root = ET.fromstring(xml_string)
            item = root.find("./channel/item")
            if item is None:
                return None

            title = item.findtext("title")
            description_html = item.findtext("description") or ""
            description = self._clean_html(description_html)
            pub_date = item.findtext("pubDate")

            update_match = re.search(r"last update:\s*([\d\-\s:]+)", description, flags=re.IGNORECASE)

            tonight_min = None
            tonight_match = re.search(r"Min temp at night:\s*([\-\d]+)", description, flags=re.IGNORECASE)
            if tonight_match:
                tonight_min = self._coerce_number(tonight_match.group(1))

            daily_forecasts = []
            daily_pattern = re.compile(
                r"(?P<date>\d{2}/\d{2})\s+\w+\s+-\s*(?P<summary>[^,]+),\s*(?P<max>\d+)\D+\-\s*(?P<min>\d+)?",
                flags=re.IGNORECASE,
            )
            for match in daily_pattern.finditer(description):
                daily_forecasts.append({
                    "date_ddmm": match.group("date"),
                    "summary": (match.group("summary") or "").strip(),
                    "max_temp_c": self._coerce_number(match.group("max")),
                    "min_temp_c": self._coerce_number(match.group("min")),
                })

            return {
                "source": "city_rss",
                "city": city_name,
                "title": title,
                "pub_date": pub_date,
                "last_update": update_match.group(1).strip() if update_match else None,
                "tonight_min_temp_c": tonight_min,
                "daily_forecasts": daily_forecasts,
                "description": description,
            }
        except Exception as exc:
            print(f"[IMSWeatherFetcher] City RSS parse error ({city_name}): {exc}")
            return None

    def _parse_radiation_rss(self, xml_string: str) -> dict | None:
        try:
            root = ET.fromstring(xml_string)
            item = root.find("./channel/item")
            if item is None:
                return None

            title = item.findtext("title")
            pub_date = item.findtext("pubDate")
            description = self._clean_html(item.findtext("description") or "")
            update_match = re.search(r"last update:\s*([\d\-\s:]+)", description, flags=re.IGNORECASE)

            city_mentions = len(re.findall(r"[A-Za-z][A-Za-z\s]+:\s*Low:", description))
            low_mentions = len(re.findall(r"\bLow:\b", description))
            medium_mentions = len(re.findall(r"\bMedium:\b", description))
            high_mentions = len(re.findall(r"\bHigh:\b", description))
            very_high_mentions = len(re.findall(r"\bVery high:\b", description, flags=re.IGNORECASE))

            return {
                "source": "radiation_rss",
                "title": title,
                "pub_date": pub_date,
                "last_update": update_match.group(1).strip() if update_match else None,
                "city_mentions": city_mentions,
                "low_mentions": low_mentions,
                "medium_mentions": medium_mentions,
                "high_mentions": high_mentions,
                "very_high_mentions": very_high_mentions,
                "description": description,
            }
        except Exception as exc:
            print(f"[IMSWeatherFetcher] Radiation RSS parse error: {exc}")
            return None

    def _parse_alert_rss(self, xml_string: str, region: str) -> dict | None:
        try:
            root = ET.fromstring(xml_string)
            channel_title = root.findtext("./channel/title")
            last_build_date = root.findtext("./channel/lastBuildDate")

            items = []
            for item in root.findall("./channel/item"):
                item_desc = self._clean_html(item.findtext("description") or "")
                items.append({
                    "title": item.findtext("title"),
                    "pub_date": item.findtext("pubDate"),
                    "description": item_desc,
                })

            return {
                "source": "alert_rss",
                "region": region,
                "channel_title": channel_title,
                "last_build_date": last_build_date,
                "item_count": len(items),
                "active": len(items) > 0,
                "items": items,
            }
        except Exception as exc:
            print(f"[IMSWeatherFetcher] Alert RSS parse error ({region}): {exc}")
            return None

    def _clean_html(self, text: str) -> str:
        text = unescape(text or "")
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

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
