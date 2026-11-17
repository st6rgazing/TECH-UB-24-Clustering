from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/model-hw-matplotlib')
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
ROOT = Path(__file__).resolve().parent
SEED = 42

def outdir():
    p = ROOT / 'results'
    p.mkdir(exist_ok=True)
    return p

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + '\n')

def check(df, n):
    assert len(df) == n
    assert not df.isna().any().any(), 'Missing values require handling'
GYM_FEATURES = ['age', 'visits_per_month', 'avg_visit_minutes', 'classes_per_month', 'months_member', 'monthly_extras_spend']

def member_decision(member, scaler, model, credit_segments):
    x = scaler.transform(pd.DataFrame([member])[GYM_FEATURES])
    segment = int(model.predict(x)[0])
    return (segment, segment in credit_segments)

def gym():
    out = outdir()
    df = pd.read_csv(ROOT / 'gym_members.csv')
    check(df, 330)
    train, test = train_test_split(df, test_size=0.3, random_state=SEED)
    scaler = StandardScaler().fit(train[GYM_FEATURES])
    xtr, xte = (scaler.transform(train[GYM_FEATURES]), scaler.transform(test[GYM_FEATURES]))
    models, spread = ({}, [])
    for k in range(1, 9):
        m = KMeans(n_clusters=k, n_init=30, random_state=SEED).fit(xtr)
        models[k] = m
        spread.append({'k': k, 'training_spread': m.inertia_, 'test_spread': -m.score(xte), 'training_spread_per_member': m.inertia_ / len(train), 'test_spread_per_member': -m.score(xte) / len(test)})
    pd.DataFrame(spread).to_csv(out / 'elbow_values.csv', index=False)
    chosen_k = 4
    model = models[chosen_k]
    centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=GYM_FEATURES)
    centers.index.name = 'segment'
    centers.to_csv(out / 'training_centroids.csv')
    train_segments = model.predict(xtr)
    training_net = np.where(train.monthly_extras_spend < 20, 5, -10)
    segment_training_net = pd.Series(training_net).groupby(train_segments).sum()
    credit_segments = segment_training_net[segment_training_net > 0].index.tolist()
    assignments = df.copy()
    assignments['segment'] = model.predict(scaler.transform(df[GYM_FEATURES]))
    assignments['split'] = np.where(df.index.isin(train.index), 'train', 'test')
    assignments['baseline_monthly_net'] = np.where(df.monthly_extras_spend < 20, 5, -10)
    assignments['v1_gets_credit'] = assignments.segment.isin(credit_segments)
    assignments['v1_monthly_net'] = np.where(assignments.v1_gets_credit, assignments.baseline_monthly_net, 0)
    assignments.to_csv(out / 'member_assignments.csv', index=False)
    segment_summary = assignments.groupby('segment').agg(member_count=('member_id', 'size'), extras_spend_mean=('monthly_extras_spend', 'mean'), members_below_20=('monthly_extras_spend', lambda s: (s < 20).sum()), baseline_monthly_net=('baseline_monthly_net', 'sum'), v1_monthly_net=('v1_monthly_net', 'sum'))
    segment_summary['gets_credit'] = segment_summary.index.isin(credit_segments)
    segment_summary['training_net_if_credited'] = segment_training_net
    segment_summary.to_csv(out / 'segment_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, field, title in zip(axes, ['training_spread', 'test_spread'], ['Training members (n=231)', 'Test members (n=99)']):
        ax.plot(range(1, 9), [r[field] for r in spread], 'o-', color='black')
        ax.axvline(chosen_k, linestyle='--', color='black')
        ax.set(xlabel='Number of segments (k)', ylabel='Sum of squared standardized distances', title=title, xticks=range(1, 9))
    fig.tight_layout()
    fig.savefig(out / 'elbow.png', dpi=180)
    plt.close(fig)
    new_member = dict(zip(GYM_FEATURES, [30, 4, 42, 1, 5, 10]))
    segment, credit = member_decision(new_member, scaler, model, credit_segments)
    save_json(out / 'summary.json', {'seed': SEED, 'train_rows': len(train), 'test_rows': len(test), 'features_used': GYM_FEATURES, 'features_excluded': ['member_id'], 'scaling': 'StandardScaler fitted on training members only', 'chosen_k': chosen_k, 'credit_segments': credit_segments, 'new_member_segment': segment, 'new_member_gets_credit': credit, 'baseline_monthly_net': int(assignments.baseline_monthly_net.sum()), 'v1_monthly_net': int(assignments.v1_monthly_net.sum()), 'monthly_improvement': int(assignments.v1_monthly_net.sum() - assignments.baseline_monthly_net.sum()), 'spread': spread})
if __name__ == '__main__':
    gym()
    print('Analysis complete. See results/.')
