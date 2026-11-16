# Collect every number the report needs into one file.
#
# The other scripts print their results to the terminal, which is fine while
# you are working but useless when you sit down to write the report. This
# script re-computes the same numbers and writes them to results.md.
#
# Run it last:  python 07_results.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # we only save files here, no window needed

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(123)
raw = pd.read_csv('AB_NYC_2019.csv')
df = pd.read_csv('clean_clustered.csv', parse_dates=['last_review'])

lines = []
def w(text=''):
    lines.append(text)

def distance(lat1, lon1, lat2, lon2):
    dx = (lon1 - lon2) * 111.0 * np.cos(np.radians(lat1))
    dy = (lat1 - lat2) * 111.0
    return np.sqrt(dx ** 2 + dy ** 2)

df['dist_to_center'] = distance(df['latitude'], df['longitude'], 40.7580, -73.9855)

# ==================================================================
w('# Result summary')
w()
w('All numbers below are printed by 03_clean.py, 04_eda.py, 05_model.py and')
w('06_cluster.py, collected here so they can be copied into the report.')
w('Random seeds are fixed, so re-running the scripts reproduces them exactly.')
w()

# ------------------------------------------------------------------
w('## 1. Data')
w()
w('| Item | Value |')
w('| --- | --- |')
w('| Raw file | AB_NYC_2019.csv |')
w('| Raw shape | %d rows x %d columns |' % raw.shape)
w('| Clean shape | %d rows x %d columns |' % (df.shape[0], df.shape[1]))
w('| Rows removed by cleaning | %d (%.1f%%) |'
  % (len(raw) - len(df), 100.0 * (len(raw) - len(df)) / len(raw)))
w('| Rows kept | %.1f%% |' % (100.0 * len(df) / len(raw)))
w('| price missing before cleaning | %d |' % raw['price'].isna().sum())
w('| price == 0 before cleaning | %d |' % (raw['price'] == 0).sum())
w('| price > 799 before cleaning | %d |' % (raw['price'] > 799).sum())
w('| reviews_per_month missing | %d |' % raw['reviews_per_month'].isna().sum())
w('| Date range of last_review | %s to %s |'
  % (raw['last_review'].min(), raw['last_review'].max()))
w()
w('Price after cleaning: mean %.2f, median %.0f, std %.2f, min %.0f, max %.0f'
  % (df['price'].mean(), df['price'].median(), df['price'].std(),
     df['price'].min(), df['price'].max()))
w()
w('Rows per borough:')
for name, n in df['neighbourhood_group'].value_counts().items():
    w('- %s: %d (%.1f%%)' % (name, n, 100.0 * n / len(df)))
w()
w('Rows per room type:')
for name, n in df['room_type'].value_counts().items():
    w('- %s: %d (%.1f%%)' % (name, n, 100.0 * n / len(df)))
w()

# ------------------------------------------------------------------
w('## 2. Exploratory analysis')
w()
w('Median price per borough:')
med_borough = df.groupby('neighbourhood_group')['price'].median().sort_values(ascending=False)
for name, v in med_borough.items():
    w('- %s: %.0f' % (name, v))
w()
w('Median price per room type:')
med_room = df.groupby('room_type')['price'].median().sort_values(ascending=False)
for name, v in med_room.items():
    w('- %s: %.0f' % (name, v))
w()
w('Correlation of each numeric feature with price:')
num_cols_all = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
                'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
corr = df[num_cols_all + ['price']].corr()['price'].drop('price').sort_values()
for name, v in corr.items():
    w('- %s: %+.4f' % (name, v))
w()

# ------------------------------------------------------------------
w('## 3. Regression models')
w()
num_cols = ['latitude', 'longitude', 'dist_to_center', 'minimum_nights',
            'number_of_reviews', 'reviews_per_month',
            'calculated_host_listings_count', 'availability_365']
cat_cols = ['room_type', 'neighbourhood_group']

X = pd.get_dummies(df[num_cols + cat_cols], columns=cat_cols)
y = df['price']
y_log = df['log_price']

X_train, X_test, y_train, y_test, ylog_train, ylog_test = train_test_split(
    X, y, y_log, test_size=0.25, random_state=123)

w('Training rows: %d, testing rows: %d, features: %d'
  % (len(X_train), len(X_test), X.shape[1]))
w()

base = y_train.median()
pred_base = np.full(len(y_test), base)

lr = LinearRegression().fit(X_train, y_train)
pred_lr = lr.predict(X_test)

lr_log = LinearRegression().fit(X_train, ylog_train)
pred_log = np.clip(np.expm1(lr_log.predict(X_test)), 0, None)

ridge = Ridge(alpha=1.0).fit(X_train, ylog_train)
pred_ridge = np.clip(np.expm1(ridge.predict(X_test)), 0, None)

rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                           random_state=123, n_jobs=-1).fit(X_train, ylog_train)
pred_rf = np.clip(np.expm1(rf.predict(X_test)), 0, None)

def score(pred):
    return (mean_absolute_error(y_test, pred),
            np.sqrt(mean_squared_error(y_test, pred)),
            r2_score(y_test, pred))

w('| Model | MAE | RMSE | R2 |')
w('| --- | --- | --- | --- |')
w('| Baseline (median = %.0f) | %.2f | %.2f | %.3f |' % ((base,) + score(pred_base)))
w('| LinearRegression, price | %.2f | %.2f | %.3f |' % score(pred_lr))
w('| LinearRegression, log_price | %.2f | %.2f | %.3f |' % score(pred_log))
w('| Ridge, log_price | %.2f | %.2f | %.3f |' % score(pred_ridge))
w('| RandomForest, log_price | %.2f | %.2f | %.3f |' % score(pred_rf))
w()

# standardised coefficients, the honest way to compare features
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
lr_std = LinearRegression().fit(X_train_s, ylog_train)
pred_std = np.clip(np.expm1(lr_std.predict(X_test_s)), 0, None)
w('Check: the standardised model has the same test MAE (%.2f).' % mean_absolute_error(y_test, pred_std))
w()
w('Standardised coefficients on log_price:')
w()
w('| Feature | Coefficient | Effect on price |')
w('| --- | --- | --- |')
coef = pd.Series(lr_std.coef_, index=X.columns).sort_values(key=abs, ascending=False)
for name, v in coef.items():
    w('| %s | %+.4f | %+.1f%% |' % (name, v, 100 * (np.exp(v) - 1)))
w()

w('RandomForest feature importance:')
w()
w('| Feature | Importance |')
w('| --- | --- |')
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
for name, v in imp.items():
    w('| %s | %.3f |' % (name, v))
w()

# ------------------------------------------------------------------
w('## 4. Clusters')
w()
w('KMeans on 7 standardised features (price NOT used for clustering).')
w('k = 6 chosen by the elbow plus silhouette score.')
w()
w('| Cluster | n | Share | Mean price | Median price | Median min nights | Median avail 365 | Main borough | Main room type | km to centre |')
w('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
for c in sorted(df['cluster'].unique()):
    part = df[df['cluster'] == c]
    w('| %d | %d | %.1f%% | %.1f | %.0f | %.0f | %.0f | %s | %s | %.1f |'
      % (c, len(part), 100.0 * len(part) / len(df),
         part['price'].mean(), part['price'].median(),
         part['minimum_nights'].median(), part['availability_365'].median(),
         part['neighbourhood_group'].value_counts().index[0],
         part['room_type'].value_counts().index[0],
         part['dist_to_center'].mean()))
w()
w('Median price of each cluster, from cheapest to most expensive:')
for c, v in df.groupby('cluster')['price'].median().sort_values().items():
    w('- cluster %d: %.0f' % (c, v))
w()

# ------------------------------------------------------------------
w('## 5. Figures')
w()
w('| File | Content |')
w('| --- | --- |')
w('| figures/fig1_price_distribution.png | price before and after the log transform |')
w('| figures/fig2_price_by_borough.png | price per borough |')
w('| figures/fig3_price_map.png | every listing on the map, coloured by price |')
w('| figures/fig4_price_by_room_type.png | price per room type |')
w('| figures/fig5_correlation.png | correlation heat map |')
w('| figures/fig6_reviews_vs_price.png | number of reviews vs price |')
w('| figures/fig7_cluster_selection.png | elbow and silhouette curves |')
w('| figures/fig8_clusters_pca.png | the six clusters in 2D (PCA) |')
w('| figures/fig9_price_by_cluster.png | price distribution of each cluster |')
w()

out = '\n'.join(lines)
with open('results.md', 'w', encoding='utf-8') as f:
    f.write(out + '\n')

print(out)
print()
print('Saved results.md (%d characters)' % len(out))