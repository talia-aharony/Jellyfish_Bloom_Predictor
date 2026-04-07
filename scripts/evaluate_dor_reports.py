import re
import html as html_lib
from datetime import datetime

import pandas as pd
import requests

from jellyfish.predictor import JellyfishPredictor


def _parse_listing_html(listing_html):
    parts = listing_html.split('<div class="list_obs link white_bkg')
    rows = []
    for part in parts[1:]:
        obs_match = re.search(r'data-link="/observation/(\d+)"', part)
        date_match = re.search(
            r'<h4 class="m-0">\s*([0-9]{2}\.[0-9]{2}\.[0-9]{4})\s*</h4>', part
        )
        h4_values = re.findall(r'<h4 class="m-0">\s*([^<]+?)\s*</h4>', part)
        time_value = h4_values[1].strip() if len(h4_values) > 1 else None
        beach_match = re.search(
            r'<a href="/list/0/(\d+)" class="under">\s*(.*?)\s*</a>',
            part,
            re.S,
        )

        if obs_match and date_match and beach_match:
            rows.append(
                {
                    "observation_id": int(obs_match.group(1)),
                    "report_date": datetime.strptime(
                        date_match.group(1), "%d.%m.%Y"
                    ).date(),
                    "report_time": time_value,
                    "meduzot_beach_id": int(beach_match.group(1)),
                    "meduzot_beach_name": html_lib.unescape(
                        " ".join(beach_match.group(2).split())
                    ),
                }
            )
    return rows


def parse_dor_reports(max_pages=50, max_empty_pages=3):
    headers = {"User-Agent": "Mozilla/5.0"}

    rows = []

    base_html = requests.get(
        "https://www.meduzot.co.il/list/4",
        headers=headers,
        timeout=30,
    ).text
    rows.extend(_parse_listing_html(base_html))

    empty_pages = 0
    for page in range(2, max_pages + 1):
        page_html = requests.get(
            f"https://www.meduzot.co.il/list/4/{page}",
            headers=headers,
            timeout=30,
        ).text
        page_rows = _parse_listing_html(page_html)
        if not page_rows:
            empty_pages += 1
            if empty_pages >= max_empty_pages:
                break
            continue

        empty_pages = 0
        rows.extend(page_rows)

    reports = pd.DataFrame(rows)
    reports = reports.drop_duplicates(subset=["observation_id"])
    reports = reports.sort_values(["report_date", "report_time", "observation_id"])
    return reports


def get_beach_mapping():
    return {
        33: 1,
        31: 1,
        26: 3,
        25: 3,
        24: 4,
        23: 4,
        22: 5,
        21: 6,
        20: 7,
        19: 8,
        18: 9,
        16: 11,
        15: 11,
        12: 13,
        10: 14,
        9: 15,
        7: 16,
        5: 17,
        2: 20,
    }


def main():
    reports_df = parse_dor_reports()
    print(f"Parsed Dor reports: {len(reports_df)}")

    reports_df["model_beach_id"] = reports_df["meduzot_beach_id"].map(get_beach_mapping())

    predictor = JellyfishPredictor(device="cpu")
    predictor.load_data_cache(
        lookback_days=24,
        forecast_days=1,
        weather_csv_path=None,
        include_live_xml=False,
    )
    predictor.load_model("JellyfishNet", "models/jellyfishnet_model.pth")

    metadata = predictor.data_cache["metadata"][["beach_id", "beach_name"]].drop_duplicates()
    metadata = metadata.rename(
        columns={"beach_id": "model_beach_id", "beach_name": "model_beach_name"}
    )
    reports_df = reports_df.merge(metadata, on="model_beach_id", how="left")

    output_rows = []
    for _, row in reports_df.iterrows():
        item = row.to_dict()
        model_beach_id = row["model_beach_id"]

        if pd.isna(model_beach_id):
            item["pred_probability"] = None
            item["pred_percentage"] = None
            item["pred_yes_no"] = None
            item["pred_confidence"] = None
            item["pred_error"] = "No model beach mapping"
            item["extrapolated"] = None
            item["extrapolated_from_date"] = None
            output_rows.append(item)
            continue

        result = predictor.predict_for_beach_date(
            int(model_beach_id), row["report_date"], "JellyfishNet"
        )

        if "error" in result:
            item["pred_probability"] = None
            item["pred_percentage"] = None
            item["pred_yes_no"] = None
            item["pred_confidence"] = None
            item["pred_error"] = result["error"]
        else:
            item["pred_probability"] = float(result["probability"])
            item["pred_percentage"] = float(result["percentage"])
            item["pred_yes_no"] = result["prediction"]
            item["pred_confidence"] = result["confidence"]
            item["pred_error"] = None

        item["extrapolated"] = result.get("extrapolated")
        item["extrapolated_from_date"] = result.get("extrapolated_from_date")
        output_rows.append(item)

    output_df = pd.DataFrame(output_rows)
    output_df = output_df.sort_values(["report_date", "report_time", "observation_id"])

    output_path = "reports/final/dor/dor_report_predictions.csv"
    output_df.to_csv(output_path, index=False)

    mapped_count = int(output_df["model_beach_id"].notna().sum())
    pred_count = int(output_df["pred_yes_no"].notna().sum())
    yes_count = int((output_df["pred_yes_no"] == "Yes").sum())
    no_count = int((output_df["pred_yes_no"] == "No").sum())

    print("\n=== Summary ===")
    print(f"Total Dor reports parsed: {len(output_df)}")
    print(f"Mapped to model beaches: {mapped_count}")
    print(f"Predictions returned: {pred_count}")
    print(f"Predicted Yes: {yes_count}")
    print(f"Predicted No: {no_count}")
    print(f"Output CSV: {output_path}")

    display_cols = [
        "report_date",
        "meduzot_beach_name",
        "model_beach_id",
        "model_beach_name",
        "pred_percentage",
        "pred_yes_no",
        "pred_confidence",
        "extrapolated",
    ]
    print("\nLatest 15 results:")
    print(
        output_df.sort_values(["report_date", "report_time"], ascending=[False, False])[
            display_cols
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
