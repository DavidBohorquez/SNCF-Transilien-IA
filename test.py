import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
import lightgbm as lgb
from tensorflow.keras import layers, models, callbacks, optimizers, losses
tf.random.set_seed(42)

# --- Loading ---
BASE_DIR = Path('.')
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)

x_train = pd.read_csv(BASE_DIR / 'x_train_final.csv')
x_test = pd.read_csv(BASE_DIR / 'x_test_final.csv')
y_train = pd.read_csv(BASE_DIR / 'y_train_final_j5KGWWK.csv')
y_sample = pd.read_csv(BASE_DIR / 'y_sample_final.csv')

# --- Preprocessing & Cleaning ---
same_mask = x_train[x_train.columns[0]].eq(x_train[x_train.columns[1]])
if same_mask.mean() > 0.99:
    x_train = x_train.drop(columns=x_train.columns[0])

x_train['x_train_row_id'] = x_train.iloc[:, 0]
x_test['x_test_row_id'] = x_test.iloc[:, 0]
y_train['y_train_row_id'] = y_train.iloc[:, 0]

x_train = x_train.drop(columns=x_train.columns[0])
x_test = x_test.drop(columns=x_test.columns[0])
y_train = y_train.drop(columns=y_train.columns[0])

train_full = x_train.merge(y_train, left_on='x_train_row_id', right_on='y_train_row_id', how='left')
test_full = x_test.copy()

train_full['source'] = 'train'
test_full['source'] = 'test'
train_df = pd.concat([train_full, test_full], ignore_index=True)

columns_to_inspect = ['p0q3', 'p0q4']
threshold = -1438
outlier_ok = (train_df[columns_to_inspect] > threshold).all(axis=1) | (train_df['source'] == 'test')
clean_train_df = train_df[outlier_ok].copy()

# --- Cycle removal: drop entire (train, date) trajectories where any 'gare' repeats ---
# Test rows are NEVER dropped (predictions are required for every test row).
trip_keys = ['train', 'date']
if all(k in clean_train_df.columns for k in trip_keys) and 'gare' in clean_train_df.columns:
    train_only = clean_train_df[clean_train_df['source'] == 'train']
    cycle_mask = (
        train_only.groupby(trip_keys, sort=False)['gare']
                  .transform(lambda s: s.duplicated(keep=False).any())
    )
    n_cycle_rows = int(cycle_mask.sum())
    n_cycle_trips = int(train_only.loc[cycle_mask, trip_keys].drop_duplicates().shape[0])
    n_total_trips = int(train_only[trip_keys].drop_duplicates().shape[0])
    pct = (n_cycle_trips / n_total_trips * 100) if n_total_trips else 0.0
    print(f"Cyclic trajectories removed: {n_cycle_trips} / {n_total_trips} trips "
          f"({pct:.2f}%), {n_cycle_rows} rows")
    clean_train_df = clean_train_df.drop(index=train_only.loc[cycle_mask].index).reset_index(drop=True)

cols_to_drop = ['x_train_row_id', 'x_test_row_id', 'y_train_row_id', 'y_sample_row_id', 'ID_original']
cols_to_drop = [c for c in cols_to_drop if c in clean_train_df.columns]
clean_train_df = clean_train_df.drop(columns=cols_to_drop)

# --- Feature Engineering ---
if 'date' in clean_train_df.columns:
    clean_train_df['date'] = pd.to_datetime(clean_train_df['date'], errors='coerce')
    clean_train_df['day_of_week'] = clean_train_df['date'].dt.dayofweek
    clean_train_df['day_of_year'] = clean_train_df['date'].dt.dayofyear

# --- Plots & EDA ---
def plot_comprehensive_eda(df):
    cat_cols = ['train', 'gare']
    for col in cat_cols:
        if col in df.columns:
            plt.figure(figsize=(12, 5))
            top_20 = df[col].value_counts().nlargest(20)
            sns.barplot(x=top_20.index, y=top_20.values, palette="viridis")
            plt.title(f"Top 20 most frequent: {col}")
            plt.ylabel("Number of Observations")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    if 'date' in df.columns:
        plt.figure(figsize=(14, 5))
        counts = df.groupby(df['date'].dt.date).size().sort_index()
        counts.plot(color='teal', linewidth=2)
        plt.title("Number of Observations per Date (chronological)")
        plt.xlabel("Date")
        plt.ylabel("Observations")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    numeric_cols = ['arret', 'day_of_week', 'day_of_year', 'is_weekend']
    for col in numeric_cols:
        if col in df.columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(data=df, x=col, bins=30, kde=False, color='dodgerblue')
            plt.title(f"Distribution of {col}")
            plt.ylabel("Number of Observations")
            plt.tight_layout()
            plt.show()

    delay_cols = ['p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4']
    available_delays = [c for c in delay_cols if c in df.columns]
    
    if available_delays:
        n_cols = len(available_delays)
        cols = 2
        rows = (n_cols + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4))
        axes = axes.flatten()
        
        for i, col in enumerate(available_delays):
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(data) > 0:
                lower, upper = data.quantile([0.01, 0.99])
                data_to_plot = data[(data >= lower) & (data <= upper)]
                sns.histplot(data_to_plot, bins=50, ax=axes[i], color='coral', kde=True)
                axes[i].set_title(f"Distribution of {col}\n(1st-99th percentile)")
                axes[i].set_ylabel("Observations")
            
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()

def build_digraph_from_df(df, train_col='train', gare_col='gare', date_col='date',
                          arret_col='arret', delay_col='p2q0', delay_col_s2='p0q2'):
    """Vectorized multi-weighted DiGraph build (consecutive stops only).

    Each edge a -> b (arret_b == arret_a + 1) carries:
      - count           : number of trips traversing the edge
      - mean_delay      : mean of p2q0 (train rows only)
      - mean_delay_s2   : mean of p0q2 (train rows only)
      - delay           : count * mean_delay  (total accumulated p2q0, used by all_flow)
      - delay_s2        : count * mean_delay_s2
      - mean_arret_diff : always 1.0 after the consecutive filter (kept for compatibility)
    Each node carries `visits`.
    """
    work = df.copy()
    if 'source' in work.columns:
        mask_test = work['source'] != 'train'
        work.loc[mask_test, delay_col] = np.nan
        if delay_col_s2 in work.columns:
            work.loc[mask_test, delay_col_s2] = np.nan

    sort_keys = [k for k in [train_col, date_col, arret_col] if k in work.columns]
    work = work.sort_values(by=sort_keys, kind='mergesort')

    grp = work.groupby([train_col, date_col], sort=False)
    work['next_gare'] = grp[gare_col].shift(-1)
    work['next_arret'] = grp[arret_col].shift(-1) if arret_col in work.columns else np.nan
    work['arret_diff'] = work['next_arret'] - work[arret_col] if arret_col in work.columns else np.nan

    # Keep only truly consecutive stops (prevents spurious edges across arret gaps)
    edges = work.dropna(subset=[gare_col, 'next_gare']).copy()
    if arret_col in work.columns:
        edges = edges[edges['arret_diff'] == 1.0]

    edges[gare_col] = edges[gare_col].astype(str)
    edges['next_gare'] = edges['next_gare'].astype(str)

    agg_cols = {'count': (gare_col, 'size'),
                'mean_delay': (delay_col, 'mean'),
                'mean_arret_diff': ('arret_diff', 'mean')}
    if delay_col_s2 in edges.columns:
        agg_cols['mean_delay_s2'] = (delay_col_s2, 'mean')
    agg = edges.groupby([gare_col, 'next_gare'], sort=False).agg(**agg_cols).reset_index()

    visits = work.dropna(subset=[gare_col]).assign(_g=lambda d: d[gare_col].astype(str)) \
                 .groupby('_g', sort=False).size().to_dict()

    G = nx.DiGraph()
    for node, v in visits.items():
        G.add_node(node, visits=int(v))
    for _, r in agg.iterrows():
        cnt = int(r['count'])
        md = float(r['mean_delay']) if pd.notna(r['mean_delay']) else 0.0
        md2 = float(r.get('mean_delay_s2', 0.0)) if pd.notna(r.get('mean_delay_s2', np.nan)) else 0.0
        G.add_edge(
            r[gare_col], r['next_gare'],
            count=cnt,
            mean_delay=md,
            mean_delay_s2=md2,
            delay=cnt * md,
            delay_s2=cnt * md2,
            mean_arret_diff=float(r['mean_arret_diff']) if pd.notna(r['mean_arret_diff']) else 1.0,
        )
    return G


def graph_adjacency_matrices(G):
    """Return (nodes, A_count, A_delay, A_arret) dense adjacency matrices."""
    nodes = sorted(G.nodes())
    A_count = nx.adjacency_matrix(G, nodelist=nodes, weight='count').toarray()
    A_delay = nx.adjacency_matrix(G, nodelist=nodes, weight='mean_delay').toarray()
    A_arret = nx.adjacency_matrix(G, nodelist=nodes, weight='mean_arret_diff').toarray()
    return nodes, A_count, A_delay, A_arret


def all_flow(G, target, depth=6):
    """Upstream weighted delay flow: sum of delay/total_count for all ancestors within `depth` BFS steps."""
    if target not in G or not nx.ancestors(G, target):
        return 0.0
    edges_bfs = list(nx.bfs_edges(G, source=target, depth_limit=depth, reverse=True))
    if not edges_bfs:
        return 0.0
    a = np.array([[G.edges[e[::-1]]['delay'], G.edges[e[::-1]]['count']]
                  for e in edges_bfs], dtype=float)
    total_count = a[:, 1].sum()
    return float((a[:, 0] / total_count).sum()) if total_count > 0 else 0.0


def plot_graph(G, out_path='graph_cleaned.png', max_edges=None):
    """Visualize the cleaned multi-weighted DiGraph and save to disk."""
    if G.number_of_edges() == 0:
        print("plot_graph: empty graph, skipping.")
        return

    pos = nx.spring_layout(G, seed=42, k=None)
    visits = np.array([G.nodes[n].get('visits', 1) for n in G.nodes()], dtype=float)
    node_sizes = 20 + 280 * (visits - visits.min()) / max(float(np.ptp(visits)), 1.0)

    edges = list(G.edges(data=True))
    if max_edges is not None and len(edges) > max_edges:
        edges = sorted(edges, key=lambda e: e[2].get('count', 0), reverse=True)[:max_edges]

    counts = np.array([e[2].get('count', 1) for e in edges], dtype=float)
    widths = 0.3 + 1.5 * np.log1p(counts) / max(np.log1p(counts).max(), 1.0)
    delays = np.array([e[2].get('mean_delay', 0.0) for e in edges], dtype=float)

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#1f77b4',
                           alpha=0.85, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v) for u, v, _ in edges],
        width=widths, edge_color=delays, edge_cmap=plt.cm.coolwarm,
        edge_vmin=np.nanpercentile(delays, 5) if delays.size else 0,
        edge_vmax=np.nanpercentile(delays, 95) if delays.size else 1,
        arrows=True, arrowsize=6, alpha=0.7, ax=ax,
    )
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                               norm=plt.Normalize(vmin=delays.min() if delays.size else 0,
                                                  vmax=delays.max() if delays.size else 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='mean_delay (p0q2)')
    ax.set_title(f"Cleaned SNCF DiGraph — {G.number_of_nodes()} nodes, "
                 f"{G.number_of_edges()} edges (showing {len(edges)})")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"plot_graph: saved {out_path}")

# plot_comprehensive_eda(clean_train_df)
BUILD_GRAPH = True
GRAPH_PNG = Path('graph_cleaned.png')
G = None
if BUILD_GRAPH:
    G = build_digraph_from_df(clean_train_df)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    nodes, A_count, A_delay, A_arret = graph_adjacency_matrices(G)
    print(f"Adjacency matrices: count.shape={A_count.shape} "
          f"nnz={int((A_count > 0).sum())}")
    if GRAPH_PNG.exists():
        print(f"plot_graph: {GRAPH_PNG} already exists — skipping (delete to regenerate).")
    else:
        plot_graph(G, out_path=str(GRAPH_PNG))


# --- Audit: train-vs-test distribution (I-03) ---
def audit_train_test_distribution(df, source_col='source',
                                  cats=('gare',), nums=('arret', 'p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4')):
    """Compare train vs test distributions to find the real source of the website gap."""
    tr = df[df[source_col] == 'train']
    te = df[df[source_col] == 'test']
    print(f"\n--- Distribution audit: train={len(tr)} rows, test={len(te)} rows ---")
    for c in cats:
        if c not in df.columns:
            continue
        tr_set = set(tr[c].dropna().unique())
        te_set = set(te[c].dropna().unique())
        new_in_test = te_set - tr_set
        unused_in_test = tr_set - te_set
        print(f"  [{c}] unique train={len(tr_set)} test={len(te_set)} "
              f"NEW-in-test={len(new_in_test)} unused-in-test={len(unused_in_test)}")
        if new_in_test:
            sample = list(new_in_test)[:5]
            print(f"        examples of NEW values: {sample}")
        tr_freq = tr[c].value_counts(normalize=True)
        te_freq = te[c].value_counts(normalize=True)
        common = tr_freq.index.intersection(te_freq.index)
        if len(common):
            tvd = 0.5 * float((tr_freq.reindex(common, fill_value=0) -
                               te_freq.reindex(common, fill_value=0)).abs().sum())
            print(f"        total-variation distance over shared values: {tvd:.4f}")
    for c in nums:
        if c not in df.columns:
            continue
        tr_v = pd.to_numeric(tr[c], errors='coerce').dropna()
        te_v = pd.to_numeric(te[c], errors='coerce').dropna()
        if tr_v.empty or te_v.empty:
            continue
        tq = tr_v.quantile([0.05, 0.5, 0.95]).values
        eq = te_v.quantile([0.05, 0.5, 0.95]).values
        print(f"  [{c}] train mean={tr_v.mean():.3f} std={tr_v.std():.3f} "
              f"q05/50/95={tq[0]:.2f}/{tq[1]:.2f}/{tq[2]:.2f}")
        print(f"        test  mean={te_v.mean():.3f} std={te_v.std():.3f} "
              f"q05/50/95={eq[0]:.2f}/{eq[1]:.2f}/{eq[2]:.2f}")
        print(f"        diff_mean={te_v.mean()-tr_v.mean():+.3f} "
              f"diff_std={te_v.std()-tr_v.std():+.3f}")


# --- Per-row graph features (must run BEFORE one-hot of `gare`) ---
def add_graph_features(df, G, gare_col='gare'):
    """Append per-row features derived from the cleaned DiGraph G.

    1-hop structural: gare_in_count, gare_out_count, gare_in_mean_delay, gare_out_mean_delay
    BFS flow (depths 1-8): flowavg1 ... flowavg8
    """
    in_count, out_count, in_delay, out_delay = {}, {}, {}, {}
    for n in G.nodes():
        ie = list(G.in_edges(n, data=True))
        oe = list(G.out_edges(n, data=True))
        ic = sum(e[2].get('count', 0) for e in ie)
        oc = sum(e[2].get('count', 0) for e in oe)
        in_count[n] = ic
        out_count[n] = oc
        in_delay[n] = (sum(e[2].get('mean_delay', 0.0) * e[2].get('count', 0) for e in ie) / ic) if ic else 0.0
        out_delay[n] = (sum(e[2].get('mean_delay', 0.0) * e[2].get('count', 0) for e in oe) / oc) if oc else 0.0

    gare_list = sorted(G.nodes())
    print("  Computing flowavg features (depths 1-8)...")
    flow_vals = {d: {g: all_flow(G, g, depth=d) for g in gare_list} for d in range(1, 9)}

    g_col = df[gare_col].astype(str)
    df = df.copy()
    df['gare_in_count'] = g_col.map(in_count).fillna(0).astype(float)
    df['gare_out_count'] = g_col.map(out_count).fillna(0).astype(float)
    df['gare_in_mean_delay'] = g_col.map(in_delay).fillna(0.0).astype(float)
    df['gare_out_mean_delay'] = g_col.map(out_delay).fillna(0.0).astype(float)
    for d in range(1, 9):
        df[f'flowavg{d}'] = g_col.map(flow_vals[d]).fillna(0.0).astype(float)
    return df


if G is not None and 'gare' in clean_train_df.columns:
    clean_train_df = add_graph_features(clean_train_df, G)

audit_train_test_distribution(clean_train_df)

# --- Modeling Prep ---
# Note: 'train' and 'date' are intentionally retained here so we can build
# GroupKFold keys and a time-based holdout below. They are excluded from the
# feature matrix at training time.

# Toggle feature groups without editing the logic below.
FEATURE_FLAGS = {
    'lag':        False,  # DISABLED: train has 1 row per (train,date) → lag always 0 in train;
                          # test has 2-4 rows/trip → non-zero lags. Catastrophic distribution shift.
    'trip_stats': False,  # DISABLED: same leakage pattern (trip mean ≈ target proxy on partial test trips)
    'cyclical':   True,   # sin/cos of day_of_week and day_of_year; month
}
# Feature importance audit runs unconditionally before training (see audit_feature_importance below)

# --- 3. Feature Engineering ---
if 'date' in clean_train_df.columns:
    clean_train_df['date'] = pd.to_datetime(clean_train_df['date'], errors='coerce')
    clean_train_df['day_of_week'] = clean_train_df['date'].dt.dayofweek
    clean_train_df['day_of_year'] = clean_train_df['date'].dt.dayofyear
    clean_train_df['month']       = clean_train_df['date'].dt.month
    print(f"\n[data audit] day_of_week present in dataset:")
    print(clean_train_df['day_of_week'].value_counts().sort_index().to_string())

if FEATURE_FLAGS['cyclical'] and 'day_of_week' in clean_train_df.columns:
    clean_train_df['dow_sin'] = np.sin(2 * np.pi * clean_train_df['day_of_week'] / 7)
    clean_train_df['dow_cos'] = np.cos(2 * np.pi * clean_train_df['day_of_week'] / 7)
    clean_train_df['doy_sin'] = np.sin(2 * np.pi * clean_train_df['day_of_year'] / 365)
    clean_train_df['doy_cos'] = np.cos(2 * np.pi * clean_train_df['day_of_year'] / 365)

if 'day_of_week' in clean_train_df.columns:
    clean_train_df['day_of_week'] = clean_train_df['day_of_week'].astype('category').cat.codes

# Lag and trip-stats features — computed before one-hot so gare string is still present.
# Sort by trip-arret only when needed; otherwise we'd silently reorder test rows and break
# any downstream code that assumes original x_test order (e.g. naive submission building).
if (FEATURE_FLAGS['lag'] or FEATURE_FLAGS['trip_stats']) and all(c in clean_train_df.columns for c in ['train', 'date', 'arret']):
    _trip = ['train', 'date']
    clean_train_df = clean_train_df.sort_values([*_trip, 'arret'], kind='mergesort')

    if FEATURE_FLAGS['lag']:
        for col in ['p2q0', 'p0q2']:
            if col in clean_train_df.columns:
                grp = clean_train_df.groupby(_trip, sort=False)[col]
                clean_train_df[f'lag1_{col}'] = grp.shift(1).fillna(0).astype(np.float32)
                clean_train_df[f'lag2_{col}'] = grp.shift(2).fillna(0).astype(np.float32)

    if FEATURE_FLAGS['trip_stats']:
        for col in ['p2q0', 'p0q2']:
            if col in clean_train_df.columns:
                clean_train_df[f'trip_mean_{col}'] = (
                    clean_train_df.groupby(_trip, sort=False)[col]
                    .transform('mean').astype(np.float32)
                )
        arret_min = clean_train_df.groupby(_trip, sort=False)['arret'].transform('min')
        arret_max = clean_train_df.groupby(_trip, sort=False)['arret'].transform('max')
        clean_train_df['trip_arret_count'] = (arret_max - arret_min + 1).astype(np.float32)
        rng = (arret_max - arret_min).replace(0, 1)
        clean_train_df['arret_position'] = ((clean_train_df['arret'] - arret_min) / rng).astype(np.float32)

if 'gare' in clean_train_df.columns and 'p0q0' in clean_train_df.columns:
    # LOO target encoding for gare: each training row gets the smoothed mean p0q0 of its gare
    # computed from the OTHER GroupKFold folds — no direct target leakage.
    # Test rows get the all-training-data smoothed mean (no target available).
    _alpha       = 1.0   # smoothing toward global mean (handles rare gares)
    _train_mask  = clean_train_df['source'] == 'train'
    _tr_df       = clean_train_df[_train_mask]
    _global_mean = float(_tr_df['p0q0'].mean())
    _gare_stats  = _tr_df.groupby('gare')['p0q0'].agg(['sum', 'count'])
    _enc_values  = np.full(len(clean_train_df), _global_mean, dtype=np.float32)
    _tr_pos      = np.where(_train_mask.values)[0]  # positions of train rows in clean_train_df
    _groups_enc  = _tr_df[['train', 'date']].astype(str).agg('|'.join, axis=1).values
    _gkf_enc     = GroupKFold(n_splits=5)
    for _, (fold_tr, fold_va) in enumerate(
            _gkf_enc.split(_tr_df, _tr_df['p0q0'].values, _groups_enc)):
        _fs  = _tr_df.iloc[fold_tr].groupby('gare')['p0q0'].agg(['sum', 'count'])
        _all_gares = clean_train_df['gare'].unique()
        _enc_map = {g: ((_fs.loc[g, 'sum'] + _alpha * _global_mean) /
                        (_fs.loc[g, 'count'] + _alpha)) if g in _fs.index
                    else _global_mean
                    for g in _all_gares}
        _enc_values[_tr_pos[fold_va]] = (
            _tr_df.iloc[fold_va]['gare'].map(_enc_map).fillna(_global_mean).values.astype(np.float32)
        )
    # Test rows: use all-training-data stats
    _test_pos = np.where(~_train_mask.values)[0]
    _test_enc_map = {g: ((_gare_stats.loc[g, 'sum'] + _alpha * _global_mean) /
                         (_gare_stats.loc[g, 'count'] + _alpha)) if g in _gare_stats.index
                     else _global_mean
                     for g in clean_train_df['gare'].unique()}
    _enc_values[_test_pos] = (
        clean_train_df[~_train_mask]['gare'].map(_test_enc_map).fillna(_global_mean).values.astype(np.float32)
    )
    clean_train_df['gare_delay_enc'] = _enc_values

    # Integer-encode gare for NN embedding (before dropping the raw string column)
    _gare_le = LabelEncoder()
    clean_train_df['gare_cat'] = _gare_le.fit_transform(
        clean_train_df['gare'].astype(str)
    ).astype(np.int32)
    N_GARE = int(clean_train_df['gare_cat'].max()) + 1

    # Drop raw gare string — replaced by gare_delay_enc (numeric) + gare_cat (embedding index)
    clean_train_df = clean_train_df.drop(columns=['gare'])
else:
    N_GARE = 0

# LOO target encoding for train number: each scheduled service has a characteristic delay profile.
# Train numbers are recurring (same service ID runs on multiple dates), so the encoding is
# informative even on a temporal split. Unseen numbers fall back to global mean via smoothing.
if 'train' in clean_train_df.columns and 'p0q0' in clean_train_df.columns:
    _alpha_t      = 1.0
    _train_mask_t = clean_train_df['source'] == 'train'
    _tr_t         = clean_train_df[_train_mask_t]
    _global_mean_t = float(_tr_t['p0q0'].mean())
    _train_stats  = _tr_t.groupby('train')['p0q0'].agg(['sum', 'count'])
    _enc_t        = np.full(len(clean_train_df), _global_mean_t, dtype=np.float32)
    _tr_pos_t     = np.where(_train_mask_t.values)[0]
    _groups_t     = _tr_t[['train', 'date']].astype(str).agg('|'.join, axis=1).values
    _gkf_t        = GroupKFold(n_splits=5)
    for _, (fold_tr, fold_va) in enumerate(
            _gkf_t.split(_tr_t, _tr_t['p0q0'].values, _groups_t)):
        _fs_t = _tr_t.iloc[fold_tr].groupby('train')['p0q0'].agg(['sum', 'count'])
        _enc_map_t = {k: ((_fs_t.loc[k, 'sum'] + _alpha_t * _global_mean_t) /
                          (_fs_t.loc[k, 'count'] + _alpha_t)) if k in _fs_t.index
                      else _global_mean_t
                      for k in clean_train_df['train'].unique()}
        _enc_t[_tr_pos_t[fold_va]] = (
            _tr_t.iloc[fold_va]['train'].map(_enc_map_t).fillna(_global_mean_t).values.astype(np.float32)
        )
    _test_mask_t = ~_train_mask_t
    _test_enc_map_t = {k: ((_train_stats.loc[k, 'sum'] + _alpha_t * _global_mean_t) /
                            (_train_stats.loc[k, 'count'] + _alpha_t)) if k in _train_stats.index
                       else _global_mean_t
                       for k in clean_train_df['train'].unique()}
    _enc_t[_test_mask_t.values] = (
        clean_train_df[_test_mask_t]['train'].map(_test_enc_map_t).fillna(_global_mean_t).values.astype(np.float32)
    )
    clean_train_df['train_delay_enc'] = _enc_t

# LOO target encoding for (gare_cat, day_of_week) pair.
# Cardinality: 84×5=420 combinations (~1600 rows each) — stable, low sparsity risk.
# Captures day-specific station delay patterns (e.g. Gare du Nord is worse on Fridays).
if 'gare_cat' in clean_train_df.columns and 'day_of_week' in clean_train_df.columns and 'p0q0' in clean_train_df.columns:
    _alpha_gd       = 1.0
    _train_mask_gd  = clean_train_df['source'] == 'train'
    _tr_gd          = clean_train_df[_train_mask_gd]
    _global_mean_gd = float(_tr_gd['p0q0'].mean())
    _gd_key_tr      = _tr_gd['gare_cat'].astype(str) + '_' + _tr_gd['day_of_week'].astype(str)
    _gd_stats       = _tr_gd.assign(_gd_key=_gd_key_tr).groupby('_gd_key')['p0q0'].agg(['sum', 'count'])
    _enc_gd         = np.full(len(clean_train_df), _global_mean_gd, dtype=np.float32)
    _tr_pos_gd      = np.where(_train_mask_gd.values)[0]
    _groups_gd      = _tr_gd[['train', 'date']].astype(str).agg('|'.join, axis=1).values
    _gkf_gd         = GroupKFold(n_splits=5)
    for _, (fold_tr, fold_va) in enumerate(
            _gkf_gd.split(_tr_gd, _tr_gd['p0q0'].values, _groups_gd)):
        _fold_keys = (_tr_gd.iloc[fold_tr]['gare_cat'].astype(str) + '_' +
                      _tr_gd.iloc[fold_tr]['day_of_week'].astype(str))
        _fs_gd = _tr_gd.iloc[fold_tr].assign(_gd_key=_fold_keys).groupby('_gd_key')['p0q0'].agg(['sum', 'count'])
        _val_keys_gd = (_tr_gd.iloc[fold_va]['gare_cat'].astype(str) + '_' +
                        _tr_gd.iloc[fold_va]['day_of_week'].astype(str))
        _enc_gd[_tr_pos_gd[fold_va]] = _val_keys_gd.map(
            lambda k: ((_fs_gd.loc[k, 'sum'] + _alpha_gd * _global_mean_gd) /
                       (_fs_gd.loc[k, 'count'] + _alpha_gd)) if k in _fs_gd.index
                      else _global_mean_gd
        ).values.astype(np.float32)
    _test_mask_gd = ~_train_mask_gd
    _test_keys_gd = (clean_train_df[_test_mask_gd]['gare_cat'].astype(str) + '_' +
                     clean_train_df[_test_mask_gd]['day_of_week'].astype(str))
    _enc_gd[_test_mask_gd.values] = _test_keys_gd.map(
        lambda k: ((_gd_stats.loc[k, 'sum'] + _alpha_gd * _global_mean_gd) /
                   (_gd_stats.loc[k, 'count'] + _alpha_gd)) if k in _gd_stats.index
                  else _global_mean_gd
    ).values.astype(np.float32)
    clean_train_df['gare_dow_enc'] = _enc_gd

# Daily delay environment: mean p2q0 per date across all trips.
# Captures systemic delay conditions that day — available for both train and test rows.
# Seasonally agnostic: reflects actual delay level, not a calendar-based seasonal proxy.
if 'p2q0' in clean_train_df.columns and 'date' in clean_train_df.columns:
    _daily_p2q0 = clean_train_df.groupby('date')['p2q0'].mean().astype(np.float32)
    clean_train_df['daily_p2q0_mean'] = clean_train_df['date'].map(_daily_p2q0).astype(np.float32)

# Per-station upstream delay statistics: mean and std of p2q0, p3q0, p4q0 per gare.
# Captures structural upstream delay characteristics of each station — stable across timetable changes.
# Orthogonal to gare_delay_enc (which encodes p0q0, the target) and flow features (graph-edge-based).
if 'gare_cat' in clean_train_df.columns:
    for _col in ['p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4']:
        if _col in clean_train_df.columns:
            _stats = clean_train_df.groupby('gare_cat')[_col].agg(['mean', 'std']).astype(np.float32)
            clean_train_df[f'gare_{_col}_mean'] = clean_train_df['gare_cat'].map(_stats['mean']).astype(np.float32)
            clean_train_df[f'gare_{_col}_std']  = clean_train_df['gare_cat'].map(_stats['std']).astype(np.float32)

# Per-(gare, day_of_week) mean of p2q0 — Mon-Fri station-specific upstream delay patterns.
if all(c in clean_train_df.columns for c in ['gare_cat', 'day_of_week', 'p2q0']):
    clean_train_df['gare_dow_p2q0_mean'] = (
        clean_train_df.groupby(['gare_cat', 'day_of_week'])['p2q0']
        .transform('mean').astype(np.float32)
    )

# --- 4. Model Definitions ---
def train_rf_best_params(df, features, target, test_size=0.2, random_state=42):
    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=30,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=False,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Holdout Validation MAE: {mae:.4f} minutes")
    print(f"Holdout Validation R2: {r2:.4f}")

    return {'model': model, 'mae': mae, 'r2': r2}

def _make_rf(random_state=42):
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=30,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=False,
        random_state=random_state,
        n_jobs=-1,
    )

def train_rf_oof(df, features, target, n_splits=5, random_state=42):
    """Legacy random-KFold OOF — kept for continuity with the 0.7318 baseline."""
    X_all = df[features].values
    y_all = df[target].values

    oof_preds = np.zeros_like(y_all, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all), 1):
        X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        y_tr = y_all[tr_idx]
        model = _make_rf(random_state)
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)

    overall_mae = mean_absolute_error(y_all, oof_preds)
    overall_r2 = r2_score(y_all, oof_preds)
    print(f"[random KFold] OOF MAE ({n_splits}-fold): {overall_mae:.4f} minutes")
    print(f"[random KFold] OOF R2  ({n_splits}-fold): {overall_r2:.4f}")
    return {'oof_predictions': oof_preds, 'oof_mae': overall_mae, 'oof_r2': overall_r2}


def train_rf_group_oof(df, features, target, group_cols=('train', 'date'),
                       n_splits=5, random_state=42):
    """GroupKFold by (train, date): the same trip never spans train/val.

    This mirrors the website's hidden split (unseen trips/dates) and should
    track the leaderboard much more closely than random KFold.
    """
    X_all = df[features].values
    y_all = df[target].values
    groups = df[list(group_cols)].astype(str).agg('|'.join, axis=1).values

    oof_preds = np.zeros_like(y_all, dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    fold_maes = []
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups=groups), 1):
        model = _make_rf(random_state)
        model.fit(X_all[tr_idx], y_all[tr_idx])
        preds = model.predict(X_all[val_idx])
        oof_preds[val_idx] = preds
        fold_mae = mean_absolute_error(y_all[val_idx], preds)
        fold_maes.append(fold_mae)
        print(f"[GroupKFold] fold {fold} MAE: {fold_mae:.4f} minutes")

    overall_mae = mean_absolute_error(y_all, oof_preds)
    overall_r2 = r2_score(y_all, oof_preds)
    print(f"[GroupKFold] OOF MAE ({n_splits}-fold by {group_cols}): {overall_mae:.4f} minutes "
          f"(per-fold std={np.std(fold_maes):.4f})")
    print(f"[GroupKFold] OOF R2 : {overall_r2:.4f}")
    return {'oof_predictions': oof_preds, 'oof_mae': overall_mae, 'oof_r2': overall_r2}


# --- Neural network (Keras) — active model from I-03 onward ---
NN_SCALE_COLS = [
    'arret', 'day_of_year', 'month',
    'p2q0', 'p3q0', 'p4q0', 'p0q2', 'p0q3', 'p0q4',
    'gare_in_count', 'gare_out_count', 'gare_in_mean_delay', 'gare_out_mean_delay',
    'gare_delay_enc', 'train_delay_enc', 'gare_dow_enc', 'gare_dow_p2q0_mean', 'daily_p2q0_mean',
    'gare_p2q0_mean', 'gare_p2q0_std',
    'gare_p3q0_mean', 'gare_p3q0_std', 'gare_p4q0_mean', 'gare_p4q0_std',
    'gare_p0q2_mean', 'gare_p0q2_std',
    'gare_p0q3_mean', 'gare_p0q3_std', 'gare_p0q4_mean', 'gare_p0q4_std',
    *[f'flowavg{d}' for d in range(1, 9)],
    'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos',
]

# Tune these without touching the architecture code below.
NN_CONFIG = {
    'units':       [512, 256, 128, 64],   # layer widths
    'activation':  'gelu',                # 'gelu' | 'swish' | 'relu' | 'leaky_relu'
    'dropout':     [0.30, 0.25, 0.20, 0.10],
    'l2':          1e-4,                  # weight-decay on Dense kernels
    'huber_delta': 0.8,
    'lr':          1e-3,
    'batch':       1024,
    'epochs':      150,
    'patience_es': 12,                    # EarlyStopping patience
    'patience_lr': 5,                     # ReduceLROnPlateau patience
    'lr_factor':   0.5,
    'min_lr':      5e-6,
    'residual':      True,                # add skip connections between same-width blocks
    'embedding_dim': 16,                  # gare embedding dimension (0 = disable)
}


def _scale_split(tr_df, va_df, features, scale_cols):
    """Fit StandardScaler on train only, apply to both. Returns numpy matrices."""
    cols_to_scale = [c for c in features if c in scale_cols]
    other_cols = [c for c in features if c not in cols_to_scale]
    scaler = StandardScaler()
    tr_scaled = scaler.fit_transform(tr_df[cols_to_scale].values.astype(np.float32)) if cols_to_scale else np.empty((len(tr_df), 0), dtype=np.float32)
    va_scaled = scaler.transform(va_df[cols_to_scale].values.astype(np.float32)) if cols_to_scale else np.empty((len(va_df), 0), dtype=np.float32)
    tr_other = tr_df[other_cols].values.astype(np.float32) if other_cols else np.empty((len(tr_df), 0), dtype=np.float32)
    va_other = va_df[other_cols].values.astype(np.float32) if other_cols else np.empty((len(va_df), 0), dtype=np.float32)
    X_tr = np.concatenate([tr_scaled, tr_other], axis=1)
    X_va = np.concatenate([va_scaled, va_other], axis=1)
    return X_tr, X_va, scaler, cols_to_scale, other_cols


def _build_nn(input_dim, cfg=None, n_gare=0):
    cfg = cfg or NN_CONFIG
    reg = tf.keras.regularizers.l2(cfg['l2'])
    emb_dim = cfg.get('embedding_dim', 16)

    def act(x):
        a = cfg['activation']
        if a == 'leaky_relu':
            return layers.LeakyReLU(0.1)(x)
        return layers.Activation(a)(x)

    def dense_block(x, units, drop):
        x = layers.Dense(units, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = act(x)
        x = layers.Dropout(drop)(x)
        return x

    unit_list = cfg['units']
    drop_list = cfg['dropout']

    if n_gare > 0:
        gare_inp = layers.Input(shape=(1,), name='gare_cat')
        num_inp  = layers.Input(shape=(input_dim,), name='numeric')
        emb = layers.Embedding(n_gare, emb_dim, name='gare_emb')(gare_inp)
        emb = layers.Flatten()(emb)
        x = layers.Concatenate()([num_inp, emb])
        inputs = [gare_inp, num_inp]
    else:
        num_inp = layers.Input(shape=(input_dim,))
        x = num_inp
        inputs = num_inp

    x = dense_block(x, unit_list[0], drop_list[0])
    for i in range(1, len(unit_list)):
        h = dense_block(x, unit_list[i], drop_list[i])
        if cfg['residual'] and x.shape[-1] == unit_list[i]:
            x = layers.Add()([x, h])
        else:
            x = h

    out = layers.Dense(1)(x)
    model = models.Model(inputs, out)
    model.compile(
        optimizer=optimizers.Adam(cfg['lr']),
        loss=losses.Huber(delta=cfg['huber_delta']),
        metrics=['mae'],
    )
    return model


def _fit_nn(X_tr, y_tr, X_va, y_va, cfg=None, n_gare=0, gare_tr=None, gare_va=None):
    cfg = cfg or NN_CONFIG
    model = _build_nn(X_tr.shape[1], cfg, n_gare=n_gare)
    cbs = [
        callbacks.EarlyStopping(patience=cfg['patience_es'], restore_best_weights=True,
                                monitor='val_mae', verbose=1),
        callbacks.ReduceLROnPlateau(factor=cfg['lr_factor'], patience=cfg['patience_lr'],
                                    monitor='val_mae', min_lr=cfg['min_lr'], verbose=1),
    ]
    train_in = [gare_tr, X_tr] if n_gare > 0 else X_tr
    val_in   = [gare_va, X_va] if n_gare > 0 else X_va
    model.fit(train_in, y_tr, validation_data=(val_in, y_va),
              epochs=cfg['epochs'], batch_size=cfg['batch'], verbose=2, callbacks=cbs)
    return model


def train_nn_group_oof(df, features, target, group_cols=('train', 'date'),
                      n_splits=5, scale_cols=NN_SCALE_COLS):
    X_all_df = df[features]
    y_all = df[target].values.astype(np.float32)
    groups = df[list(group_cols)].astype(str).agg('|'.join, axis=1).values
    oof_preds = np.zeros_like(y_all, dtype=np.float32)
    gkf = GroupKFold(n_splits=n_splits)
    fold_maes = []
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all_df, y_all, groups=groups), 1):
        X_tr, X_va, _, _, _ = _scale_split(X_all_df.iloc[tr_idx], X_all_df.iloc[val_idx],
                                            features, scale_cols)
        model = _fit_nn(X_tr, y_all[tr_idx], X_va, y_all[val_idx])
        preds = model.predict(X_va, verbose=0).ravel()
        oof_preds[val_idx] = preds
        fold_mae = mean_absolute_error(y_all[val_idx], preds)
        fold_maes.append(fold_mae)
        print(f"[NN GroupKFold] fold {fold} MAE: {fold_mae:.4f}")
        tf.keras.backend.clear_session()
    overall_mae = mean_absolute_error(y_all, oof_preds)
    overall_r2 = r2_score(y_all, oof_preds)
    print(f"[NN GroupKFold] OOF MAE ({n_splits}-fold): {overall_mae:.4f} "
          f"(per-fold std={np.std(fold_maes):.4f})")
    print(f"[NN GroupKFold] OOF R2 : {overall_r2:.4f}")
    return {'oof_predictions': oof_preds, 'oof_mae': overall_mae, 'oof_r2': overall_r2}


def train_nn_time_holdout(df, features, target, date_col='date', quantile=0.85,
                          scale_cols=NN_SCALE_COLS):
    if date_col not in df.columns or df[date_col].isna().all():
        return None
    cutoff = df[date_col].quantile(quantile)
    tr = df[df[date_col] < cutoff]
    ho = df[df[date_col] >= cutoff]
    if len(tr) == 0 or len(ho) == 0:
        return None
    X_tr, X_ho, _, _, _ = _scale_split(tr[features], ho[features], features, scale_cols)
    model = _fit_nn(X_tr, tr[target].values.astype(np.float32),
                    X_ho, ho[target].values.astype(np.float32))
    preds = model.predict(X_ho, verbose=0).ravel()
    mae = mean_absolute_error(ho[target].values, preds)
    r2 = r2_score(ho[target].values, preds)
    print(f"[NN time holdout] cutoff={cutoff} train={len(tr)} holdout={len(ho)}")
    print(f"[NN time holdout] MAE: {mae:.4f} | R2: {r2:.4f}")
    tf.keras.backend.clear_session()
    return {'mae': mae, 'r2': r2, 'cutoff': cutoff}


def train_nn_full_predict(train_df, test_df, features, target, scale_cols=NN_SCALE_COLS,
                          val_frac=0.1, gare_cat_col=None, n_gare=0):
    """Fit NN on all training data (with a small val split for early stopping) and predict on test_df.
    When gare_cat_col is given and n_gare > 0, uses a dual-input model with gare Embedding.
    Returns (predictions, best_val_mae).
    """
    n = len(train_df)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_val = max(int(n * val_frac), 1)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    tr = train_df.iloc[tr_idx]
    va = train_df.iloc[val_idx]
    X_tr, X_va, scaler, scaled_cols, other_cols = _scale_split(tr[features], va[features],
                                                                features, scale_cols)
    use_emb = gare_cat_col is not None and n_gare > 0
    gare_tr = tr[gare_cat_col].values.astype(np.int32)  if use_emb else None
    gare_va = va[gare_cat_col].values.astype(np.int32)  if use_emb else None
    model = _fit_nn(X_tr, tr[target].values.astype(np.float32),
                    X_va, va[target].values.astype(np.float32),
                    n_gare=n_gare if use_emb else 0,
                    gare_tr=gare_tr, gare_va=gare_va)
    best_val_mae = float(min(model.history.history.get('val_mae', [float('nan')])))
    X_te_scaled = scaler.transform(test_df[scaled_cols].values.astype(np.float32)) if scaled_cols else np.empty((len(test_df), 0), dtype=np.float32)
    X_te_other = test_df[other_cols].values.astype(np.float32) if other_cols else np.empty((len(test_df), 0), dtype=np.float32)
    X_te = np.concatenate([X_te_scaled, X_te_other], axis=1)
    if use_emb:
        gare_te = test_df[gare_cat_col].values.astype(np.int32)
        preds = model.predict([gare_te, X_te], verbose=0).ravel()
    else:
        preds = model.predict(X_te, verbose=0).ravel()
    tf.keras.backend.clear_session()
    return preds, best_val_mae


_LGBM_PARAMS = {
    'objective':         'regression_l1',
    'metric':            'mae',
    'num_leaves':        127,
    'learning_rate':     0.02,
    'min_child_samples': 40,
    'subsample':         0.85,
    'colsample_bytree':  0.90,
    'reg_alpha':         0.1,
    'reg_lambda':        3.0,
    'verbose':           -1,
}

_CATBOOST_PARAMS = {
    'loss_function':       'MAE',
    'eval_metric':         'MAE',
    'iterations':          3000,
    'learning_rate':       0.02,
    'depth':               8,
    'l2_leaf_reg':         3.0,
    'subsample':           0.85,
    'random_seed':         42,
    'early_stopping_rounds': 50,
    'verbose':             200,
    'use_best_model':      True,
    'task_type':           'CPU',
}


def train_lgbm_oof_predict(train_df, test_df, features, target='p0q0',
                            group_cols=('train', 'date'), n_splits=5, seed=42):
    """GroupKFold OOF: clean, leakage-free LGBM predictions for every training row.
    Test predictions are BAGGED across the 5 fold models (mean of fold predictions),
    NOT a full-data retrain — this guarantees train and test `lgbm_pred` distributions
    match (both come from models trained on 4/5 of the data).
    Returns (oof_predictions, test_predictions, oof_mae).
    OOF predictions are injected as `lgbm_pred` feature into the NN downstream.
    """
    params = {**_LGBM_PARAMS, 'random_state': seed}
    X_all = train_df[features]
    y_all = train_df[target].values.astype(np.float32)
    groups = train_df[list(group_cols)].astype(str).agg('|'.join, axis=1).values

    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    # Bagged test predictions: each fold's model predicts on test, then we average.
    # This keeps the distribution of `lgbm_pred` IDENTICAL between train (OOF) and test
    # (both come from models trained on 4/5 of the data), avoiding the classic stacking
    # mismatch where a full-data retrain produces sharper test preds than OOF train preds.
    test_preds_folds = np.zeros((n_splits, len(test_df)), dtype=np.float32)
    gkf = GroupKFold(n_splits=n_splits)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups=groups), 1):
        dtrain = lgb.Dataset(X_all.iloc[tr_idx], label=y_all[tr_idx])
        dval   = lgb.Dataset(X_all.iloc[val_idx], label=y_all[val_idx], reference=dtrain)
        booster = lgb.train(
            params, dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
        )
        oof_preds[val_idx]       = booster.predict(X_all.iloc[val_idx]).astype(np.float32)
        test_preds_folds[fold-1] = booster.predict(test_df[features]).astype(np.float32)
        fold_mae = mean_absolute_error(y_all[val_idx], oof_preds[val_idx])
        print(f"  [LGBM fold {fold}/{n_splits}] MAE={fold_mae:.4f}  best_iter={booster.best_iteration}")

    oof_mae = mean_absolute_error(y_all, oof_preds)
    test_preds = test_preds_folds.mean(axis=0)
    print(f"  [LGBM OOF] MAE={oof_mae:.4f}  (test preds bagged across {n_splits} folds)")

    # Temporal OOF: MAE on the most recent 20% of training dates — proxy for website performance.
    # Test set comes from a later time period, so this metric better reflects generalisation.
    _dates = train_df['date'].values
    _date_thresh = np.percentile(_dates, 80)
    _recent_mask = _dates >= _date_thresh
    temporal_oof_mae = float('nan')
    if _recent_mask.sum() > 0:
        temporal_oof_mae = mean_absolute_error(y_all[_recent_mask], oof_preds[_recent_mask])
        print(f"  [LGBM temporal OOF] MAE on most recent 20% of dates: {temporal_oof_mae:.4f}"
              f"  ({_recent_mask.sum()} rows, date >= {_date_thresh})")

    return oof_preds, test_preds, oof_mae, temporal_oof_mae


def train_catboost_oof_predict(train_df, test_df, features, target='p0q0',
                               group_cols=('train', 'date'), n_splits=5, seed=42):
    """GroupKFold OOF for CatBoost; test preds bagged across folds.
    Mirrors train_lgbm_oof_predict structure so train/test distributions stay aligned.
    """
    from catboost import CatBoostRegressor
    params = {**_CATBOOST_PARAMS, 'random_seed': seed}
    X_all = train_df[features]
    y_all = train_df[target].values.astype(np.float32)
    groups = train_df[list(group_cols)].astype(str).agg('|'.join, axis=1).values

    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    test_preds_folds = np.zeros((n_splits, len(test_df)), dtype=np.float32)
    gkf = GroupKFold(n_splits=n_splits)

    _cat_features = ['gare_cat'] if 'gare_cat' in features else None
    if _cat_features:
        params['cat_features'] = _cat_features
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups=groups), 1):
        model = CatBoostRegressor(**params)
        model.fit(
            X_all.iloc[tr_idx], y_all[tr_idx],
            eval_set=(X_all.iloc[val_idx], y_all[val_idx]),
        )
        oof_preds[val_idx]       = model.predict(X_all.iloc[val_idx]).astype(np.float32)
        test_preds_folds[fold-1] = model.predict(test_df[features]).astype(np.float32)
        fold_mae = mean_absolute_error(y_all[val_idx], oof_preds[val_idx])
        print(f"  [CatBoost fold {fold}/{n_splits}] MAE={fold_mae:.4f}")

    oof_mae = mean_absolute_error(y_all, oof_preds)
    test_preds = test_preds_folds.mean(axis=0)
    print(f"  [CatBoost OOF] MAE={oof_mae:.4f}  (test preds bagged across {n_splits} folds)")
    return oof_preds, test_preds, oof_mae


def train_rf_time_holdout(df, features, target, date_col='date', quantile=0.85,
                          random_state=42):
    """Train on the first `quantile` of dates, evaluate on the remainder.

    This is the closest local proxy to the website's hidden test set when the
    leaderboard is split chronologically. Reported alongside the two OOFs so
    we can watch the random/group/time gap directly.
    """
    if date_col not in df.columns or df[date_col].isna().all():
        print("[time holdout] no usable date column, skipping.")
        return None
    cutoff = df[date_col].quantile(quantile)
    tr = df[df[date_col] < cutoff]
    ho = df[df[date_col] >= cutoff]
    if len(tr) == 0 or len(ho) == 0:
        print(f"[time holdout] empty split at quantile={quantile}, skipping.")
        return None

    model = _make_rf(random_state)
    model.fit(tr[features].values, tr[target].values)
    preds = model.predict(ho[features].values)
    mae = mean_absolute_error(ho[target].values, preds)
    r2 = r2_score(ho[target].values, preds)
    print(f"[time holdout] cutoff={cutoff} (train={len(tr)} rows, holdout={len(ho)} rows)")
    print(f"[time holdout] MAE: {mae:.4f} minutes | R2: {r2:.4f}")
    return {'mae': mae, 'r2': r2, 'cutoff': cutoff}

def audit_feature_importance(df, features, target='p0q0', sample_frac=0.30, seed=42):
    """Two-method feature importance audit on a sample.

    Method 1: LGBM gain + split (matches the production model).
    Method 2: mutual_info_regression (non-parametric, captures non-linear signal).
    Verdict: 'DEAD' if both LGBM gain == 0 AND mutual_info < 1e-4 — flagged for manual removal.
    """
    print(f"\n--- Feature importance audit ({int(sample_frac*100)}% sample, n_features={len(features)}) ---")
    sample = df.sample(frac=sample_frac, random_state=seed)
    X = sample[features].fillna(0).values.astype(np.float32)
    y = sample[target].values.astype(np.float32)

    lgbm_audit = lgb.train(
        {'objective': 'regression_l1', 'metric': 'mae', 'verbose': -1,
         'num_leaves': 63, 'learning_rate': 0.05, 'random_state': seed},
        lgb.Dataset(X, label=y),
        num_boost_round=200,
    )
    gain  = lgbm_audit.feature_importance(importance_type='gain')
    split = lgbm_audit.feature_importance(importance_type='split')

    mi = mutual_info_regression(X, y, random_state=seed)

    rows = []
    for i, f in enumerate(features):
        verdict = 'DEAD' if (gain[i] == 0.0 and mi[i] < 1e-4) else 'KEEP'
        rows.append((f, float(gain[i]), int(split[i]), float(mi[i]), verdict))
    rows.sort(key=lambda r: -r[1])

    print(f"  {'feature':<22}{'lgbm_gain':>12}{'lgbm_split':>12}{'mutual_info':>14}   verdict")
    for f, g, s, m, v in rows:
        print(f"  {f:<22}{g:>12.2f}{s:>12d}{m:>14.4f}   {v}")
    dead = [r[0] for r in rows if r[4] == 'DEAD']
    if dead:
        print(f"  → {len(dead)} DEAD features flagged: {dead}")
    return rows


# --- 5. Execution & Submission ---
X_train_df = clean_train_df[clean_train_df['source'] == 'train'].copy()
X_test_df  = clean_train_df[clean_train_df['source'] == 'test'].copy()

non_features = {'source', 'date', 'train', 'p0q0',
                'x_train_row_id', 'x_test_row_id', 'y_train_row_id'}
features = [col for col in X_train_df.columns if col not in non_features]

# NN uses graph/numeric features + gare Embedding; no OHE gare columns (they're redundant
# with the embedding and would dominate the numeric input in a noisy way).
_GRAPH_GARE_COLS = {'gare_in_count', 'gare_out_count', 'gare_in_mean_delay', 'gare_out_mean_delay',
                    'gare_delay_enc'}  # gare_delay_enc is numeric, not an OHE column
_gare_ohe_cols   = {f for f in features
                    if f.startswith('gare_') and f not in _GRAPH_GARE_COLS and f != 'gare_cat'}
nn_features = [f for f in features if f not in _gare_ohe_cols and f != 'gare_cat']

n_flow = sum(c.startswith('flowavg') for c in features)
print(f"\nFeatures: {len(features)} total | flow={n_flow} gare_ohe={len(_gare_ohe_cols)} "
      f"nn_features={len(nn_features)} N_GARE={N_GARE}")

audit_feature_importance(X_train_df, features, target='p0q0', sample_frac=0.30)

# 2-model ensemble: LGBM + NN (gare Embedding)
# CatBoost dropped: OOF optimiser gave it weight=0.000 in I-13 and I-14 — no orthogonal signal.
LGBM_WEIGHT = 0.85
NN_WEIGHT   = 0.15

print('\n--- [LGBM] GroupKFold OOF (5 folds) + bagged test predictions ---')
print(f"  Training date range: {X_train_df['date'].min()} to {X_train_df['date'].max()}")
print(f"  Test date range:     {X_test_df['date'].min()} to {X_test_df['date'].max()}")
lgbm_oof, lgbm_test_preds, lgbm_oof_mae, lgbm_temporal_mae = train_lgbm_oof_predict(
    X_train_df, X_test_df, features, target='p0q0')
print(f"  LGBM OOF MAE: {lgbm_oof_mae:.4f}  |  Temporal OOF MAE: {lgbm_temporal_mae:.4f}")

print('\n--- [NN] Training on numeric features + gare Embedding ---')
nn_test_preds, nn_val_mae = train_nn_full_predict(
    X_train_df, X_test_df, nn_features, target='p0q0',
    gare_cat_col='gare_cat', n_gare=N_GARE)
print(f"  NN val MAE: {nn_val_mae:.4f}")

test_predictions = (LGBM_WEIGHT * lgbm_test_preds
                    + NN_WEIGHT  * nn_test_preds)
print(f"\n  Ensemble (LGBM×{LGBM_WEIGHT:.3f} + NN×{NN_WEIGHT:.3f}): "
      f"LGBM={lgbm_oof_mae:.4f} | NN={nn_val_mae:.4f}")

submission_df = pd.DataFrame({
    'row_id': x_test['x_test_row_id'].values,
    'p0q0':   test_predictions,
})
assert len(submission_df) == len(x_test), \
    f"submission rows {len(submission_df)} != x_test rows {len(x_test)}"
assert set(submission_df['row_id']) == set(x_test['x_test_row_id']), \
    "submission row_ids do not match x_test row_ids — alignment broken"
submission_df = submission_df.sort_values('row_id').reset_index(drop=True)
submission_df.to_csv('submission_test_rf.csv', index=False)
print(f"Saved submission_test_rf.csv ({len(submission_df)} rows, row_ids verified)")