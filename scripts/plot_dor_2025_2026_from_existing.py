import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

src = 'reports/final/dor/dor_model_comparison_details.csv'
df = pd.read_csv(src)
df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
start_date = pd.Timestamp('2024-06-01')
end_date = pd.Timestamp('2026-12-31')
df = df[(df['report_date'] >= start_date) & (df['report_date'] <= end_date)].copy()

valid = df[(df['actual'].notna()) & (df['jellyfishnet_yes_no'].notna()) & (df['baseline_yes_no'].notna())].copy()
valid['actual'] = valid['actual'].astype(int)
valid['j_pred'] = (valid['jellyfishnet_yes_no'] == 'Yes').astype(int)
valid['b_pred'] = (valid['baseline_yes_no'] == 'Yes').astype(int)

def metrics(actual, pred):
    tp = int(((pred == 1) & (actual == 1)).sum())
    fp = int(((pred == 1) & (actual == 0)).sum())
    fn = int(((pred == 0) & (actual == 1)).sum())
    tn = int(((pred == 0) & (actual == 0)).sum())
    acc = (tp + tn) / len(actual) if len(actual) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {'n': int(len(actual)), 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

m_b = metrics(valid['actual'].values, valid['b_pred'].values)
m_j = metrics(valid['actual'].values, valid['j_pred'].values)

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Jellyfish Bloom Predictor - Evaluation on Real Sightings (Jun 2024-2026)\n(data: meduzot.co.il / Dor profile)', fontsize=14, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax0 = fig.add_subplot(gs[0, 0])
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
b_vals = [m_b['accuracy'], m_b['precision'], m_b['recall'], m_b['f1']]
j_vals = [m_j['accuracy'], m_j['precision'], m_j['recall'], m_j['f1']]
x = np.arange(len(metric_names)); w = 0.35
bars_b = ax0.bar(x - w/2, b_vals, w, label='Baseline (Logistic Regression)', color='#4C72B0', alpha=0.85)
bars_j = ax0.bar(x + w/2, j_vals, w, label='JellyfishNet', color='#DD8452', alpha=0.85)
ax0.set_ylim(0, 1.05); ax0.set_xticks(x); ax0.set_xticklabels(metric_names)
ax0.set_ylabel('Score'); ax0.set_title('Model Performance Metrics'); ax0.legend(fontsize=9); ax0.grid(axis='y', alpha=0.3)
for b in list(bars_b) + list(bars_j):
    h = b.get_height()
    ax0.text(b.get_x()+b.get_width()/2, h+0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=8)

inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.4)
def draw_cm(ax, m, title):
    cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
    ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Neg','Pos'], fontsize=9); ax.set_yticklabels(['Neg','Pos'], fontsize=9)
    ax.set_xlabel('Predicted', fontsize=8); ax.set_ylabel('Actual', fontsize=8)
    labels = [['TN','FP'],['FN','TP']]
    th = cm.max()/2 if cm.max()>0 else 0
    for r in range(2):
        for c in range(2):
            color = 'white' if cm[r,c] > th else 'black'
            ax.text(c, r, f"{labels[r][c]}\n{cm[r,c]}", ha='center', va='center', fontsize=11, fontweight='bold', color=color)

ax1 = fig.add_subplot(inner[0,0]); ax2 = fig.add_subplot(inner[0,1])
draw_cm(ax1, m_b, 'Baseline (Logistic Regression)')
draw_cm(ax2, m_j, 'JellyfishNet')

ax3 = fig.add_subplot(gs[1, 0])
tl = valid[['report_date', 'baseline_probability', 'jellyfishnet_probability', 'actual']].dropna().sort_values('report_date')
ax3.plot(tl['report_date'], tl['baseline_probability'], 'b-o', markersize=3, alpha=0.7, label='Baseline prob', linewidth=1)
ax3.plot(tl['report_date'], tl['jellyfishnet_probability'], 'r-s', markersize=3, alpha=0.7, label='JellyfishNet prob', linewidth=1)
for d, a in zip(tl['report_date'], tl['actual']):
    if int(a) == 1:
        ax3.axvline(d, color='green', alpha=0.05, linewidth=3)
ax3.axhline(0.5, color='grey', linestyle='--', linewidth=1, label='threshold=0.5')
ax3.set_ylim(-0.05, 1.05); ax3.set_ylabel('Predicted probability'); ax3.set_title('Prediction Timeline\n(green shading = actual sighting)')
ax3.legend(fontsize=8); ax3.tick_params(axis='x', rotation=30); ax3.grid(alpha=0.2)

ax4 = fig.add_subplot(gs[1, 1])
acc_rows = []
for bn, grp in valid.groupby('model_beach_name', dropna=False):
    if pd.isna(bn):
        continue
    b_acc = float((grp['b_pred'] == grp['actual']).mean())
    j_acc = float((grp['j_pred'] == grp['actual']).mean())
    acc_rows.append((str(bn), b_acc, j_acc))
acc_rows.sort(key=lambda x: x[0])
if acc_rows:
    names = [r[0].split('-')[0][:12] for r in acc_rows]
    bacc = [r[1] for r in acc_rows]
    jacc = [r[2] for r in acc_rows]
    xb = np.arange(len(names)); wb = 0.35
    ax4.bar(xb - wb/2, bacc, wb, label='Baseline (Logistic Regression)', color='#4C72B0', alpha=0.85)
    ax4.bar(xb + wb/2, jacc, wb, label='JellyfishNet', color='#DD8452', alpha=0.85)
    ax4.set_xticks(xb); ax4.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax4.set_ylim(0, 1.1); ax4.set_ylabel('Accuracy'); ax4.set_title('Per-Beach Accuracy')
    ax4.legend(fontsize=8); ax4.grid(axis='y', alpha=0.3)

out_plot = 'reports/final/dor/dor_evaluation_2024_06_2026.png'
out_summary = 'reports/final/dor/dor_model_comparison_2024_06_2026_summary.json'
plt.savefig(out_plot, dpi=130, bbox_inches='tight')
plt.close(fig)

with open(out_summary, 'w') as f:
    json.dump({'n_rows_window': int(len(df)), 'n_valid_compared': int(len(valid)), 'Baseline': m_b, 'JellyfishNet': m_j}, f, indent=2)

print('plot', out_plot)
print('summary', out_summary)
print('n_rows_window', len(df))
print('n_valid_compared', len(valid))
