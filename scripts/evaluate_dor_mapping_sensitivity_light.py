import pandas as pd

from jellyfish.predictor import JellyfishPredictor
from scripts.evaluate_dor_reports import parse_dor_reports, get_beach_mapping


def run_predictions(reports_df, mapping, predictor, label):
    rows = []
    for _, row in reports_df.iterrows():
        model_beach_id = row['meduzot_beach_id']
        model_beach_id = mapping.get(int(model_beach_id))
        if model_beach_id is None:
            rows.append({
                'scenario': label,
                'observation_id': int(row['observation_id']),
                'meduzot_beach_id': int(row['meduzot_beach_id']),
                'model_beach_id': None,
                'pred_yes_no': None,
                'pred_percentage': None,
            })
            continue

        result = predictor.predict_for_beach_date(int(model_beach_id), row['report_date'], 'JellyfishNet')
        rows.append({
            'scenario': label,
            'observation_id': int(row['observation_id']),
            'meduzot_beach_id': int(row['meduzot_beach_id']),
            'model_beach_id': int(model_beach_id),
            'pred_yes_no': None if 'error' in result else result['prediction'],
            'pred_percentage': None if 'error' in result else float(result['percentage']),
        })
    return pd.DataFrame(rows)


def summarize(df):
    pred = df['pred_yes_no'].notna().sum()
    yes = (df['pred_yes_no'] == 'Yes').sum()
    no = (df['pred_yes_no'] == 'No').sum()
    return {
        'predictions': int(pred),
        'yes': int(yes),
        'no': int(no),
        'yes_rate_pct': float((yes / pred * 100.0) if pred else 0.0),
    }


def main():
    reports = parse_dor_reports()
    base_mapping = get_beach_mapping()

    predictor = JellyfishPredictor(device='cpu')
    predictor.load_data_cache(
        lookback_days=24,
        forecast_days=1,
        weather_csv_path=None,
        include_live_xml=False,
    )
    predictor.load_model('JellyfishNet', 'models/jellyfishnet_model.pth')

    scenarios = {
        'baseline': base_mapping,
        'hifa_south_shift': {**base_mapping, 24: 5, 23: 5, 21: 6},
        'hifa_center_shift': {**base_mapping, 24: 4, 23: 4, 21: 5},
        'hifa_north_shift': {**base_mapping, 24: 3, 23: 3, 21: 7},
        'telaviv_north': {**base_mapping, 10: 14},
        'telaviv_central': {**base_mapping, 10: 15},
        'telaviv_south': {**base_mapping, 10: 13},
    }

    all_frames = []
    summaries = []
    for name, mapping in scenarios.items():
        df = run_predictions(reports, mapping, predictor, name)
        all_frames.append(df)
        s = summarize(df)
        s['scenario'] = name
        summaries.append(s)

    all_df = pd.concat(all_frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values(['yes', 'yes_rate_pct'], ascending=[False, False])

    baseline = all_df[all_df['scenario'] == 'baseline'].set_index('observation_id')
    diffs = []
    for scenario in scenarios:
        if scenario == 'baseline':
            continue
        scen = all_df[all_df['scenario'] == scenario].set_index('observation_id')
        comp = baseline[['pred_yes_no']].join(scen[['pred_yes_no']], lsuffix='_base', rsuffix='_alt')
        changes = comp[comp['pred_yes_no_base'] != comp['pred_yes_no_alt']]
        diffs.append({
            'scenario': scenario,
            'changed_predictions': int(len(changes)),
            'changed_to_yes': int((changes['pred_yes_no_alt'] == 'Yes').sum()),
            'changed_to_no': int((changes['pred_yes_no_alt'] == 'No').sum()),
        })
    diff_df = pd.DataFrame(diffs).sort_values('changed_predictions', ascending=False)

    summary_path = 'reports/dor_mapping_sensitivity_light_summary.csv'
    diff_path = 'reports/dor_mapping_sensitivity_light_diffs.csv'
    summary_df.to_csv(summary_path, index=False)
    diff_df.to_csv(diff_path, index=False)

    print('=== Sensitivity Summary ===')
    print(summary_df.to_string(index=False))
    print('\n=== Baseline vs scenario changes ===')
    print(diff_df.to_string(index=False))
    print('\nSaved:')
    print(summary_path)
    print(diff_path)


if __name__ == '__main__':
    main()
