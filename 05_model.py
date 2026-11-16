# Week 4: regression models that predict the price of a listing
#
# Q1 asked in the report: which factors really decide the price?
# This script answers it with numbers.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(123)
df = pd.read_csv('clean.csv', parse_dates=['last_review'])

# ------------------------------------------------------------------
# 1. Feature engineering
#    dist_to_center: distance to Times Square, in km. A listing near the
#    centre should be more expensive, but latitude/longitude alone do not
#    tell the model that directly.
# ------------------------------------------------------------------
def distance(lat1, lon1, lat2, lon2):
    # simple equirectangular approximation, good enough for one city
    dx = (lon1 - lon2) * 111.0 * np.cos(np.radians(lat1))
    dy = (lat1 - lat2) * 111.0
    return np.sqrt(dx ** 2 + dy ** 2)

df['dist_to_center'] = distance(df['latitude'], df['longitude'], 40.7580, -73.9855)

print('Distance to Times Square: min %.2f km, median %.2f km, max %.2f km'
      % (df['dist_to_center'].min(), df['dist_to_center'].median(), df['dist_to_center'].max()))
print()

# ------------------------------------------------------------------
# 2. Build X and y
#    room_type and neighbourhood_group are text, so we one-hot them.
# ------------------------------------------------------------------
num_cols = ['latitude', 'longitude', 'dist_to_center', 'minimum_nights',
            'number_of_reviews', 'reviews_per_month',
            'calculated_host_listings_count', 'availability_365']
cat_cols = ['room_type', 'neighbourhood_group']

X = pd.get_dummies(df[num_cols + cat_cols], columns=cat_cols)
y = df['price']
y_log = df['log_price']

print('Number of features: %d' % X.shape[1])
print('Feature names:', list(X.columns))
print()

# the same split is used by every model below
X_train, X_test, y_train, y_test, ylog_train, ylog_test = train_test_split(
    X, y, y_log, test_size=0.25, random_state=123)

print('Training rows: %d, testing rows: %d' % (len(X_train), len(X_test)))
print()

# ------------------------------------------------------------------
# 3. Baseline: always predict the median price of the training set
# ------------------------------------------------------------------
baseline = y_train.median()
pred_base = np.full(len(y_test), baseline)
base_mae = mean_absolute_error(y_test, pred_base)
base_rmse = np.sqrt(mean_squared_error(y_test, pred_base))
base_r2 = r2_score(y_test, pred_base)
print('Baseline (always predict the median = %.0f)' % baseline)
print('  MAE %.2f | RMSE %.2f | R2 %.3f' % (base_mae, base_rmse, base_r2))
print()

# ------------------------------------------------------------------
# 4. Model 1: linear regression on the raw price
# ------------------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print('Model 1: LinearRegression on price')
print('  MAE %.2f | RMSE %.2f | R2 %.3f'
      % (mean_absolute_error(y_test, pred_lr),
         np.sqrt(mean_squared_error(y_test, pred_lr)),
         r2_score(y_test, pred_lr)))
print()

# ------------------------------------------------------------------
# 5. Model 2: linear regression on log_price, then convert back
#    price is right-skewed, so fitting log(price) is usually better.
#    expm1 reverses log1p, and we clip at 0 because a negative price
#    makes no sense.
# ------------------------------------------------------------------
lr_log = LinearRegression()
lr_log.fit(X_train, ylog_train)
pred_log = np.expm1(lr_log.predict(X_test))
pred_log = np.clip(pred_log, 0, None)
print('Model 2: LinearRegression on log_price')
print('  MAE %.2f | RMSE %.2f | R2 %.3f'
      % (mean_absolute_error(y_test, pred_log),
         np.sqrt(mean_squared_error(y_test, pred_log)),
         r2_score(y_test, pred_log)))
print()

# ------------------------------------------------------------------
# 6. Model 3 and 4: Ridge and Random Forest, on log_price
# ------------------------------------------------------------------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, ylog_train)
pred_ridge = np.clip(np.expm1(ridge.predict(X_test)), 0, None)

rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=123, n_jobs=-1)
rf.fit(X_train, ylog_train)
pred_rf = np.clip(np.expm1(rf.predict(X_test)), 0, None)

print('Model 3: Ridge on log_price')
print('  MAE %.2f | RMSE %.2f | R2 %.3f'
      % (mean_absolute_error(y_test, pred_ridge),
         np.sqrt(mean_squared_error(y_test, pred_ridge)),
         r2_score(y_test, pred_ridge)))
print()
print('Model 4: RandomForest on log_price')
print('  MAE %.2f | RMSE %.2f | R2 %.3f'
      % (mean_absolute_error(y_test, pred_rf),
         np.sqrt(mean_squared_error(y_test, pred_rf)),
         r2_score(y_test, pred_rf)))
print()

# ------------------------------------------------------------------
# 7. Comparison table (this goes into the report)
# ------------------------------------------------------------------
print('=' * 62)
print('%-34s %8s %8s %7s' % ('Model', 'MAE', 'RMSE', 'R2'))
print('-' * 62)
rows = [
    ('Baseline (median)', base_mae, base_rmse, base_r2),
    ('LinearRegression, price', mean_absolute_error(y_test, pred_lr),
     np.sqrt(mean_squared_error(y_test, pred_lr)), r2_score(y_test, pred_lr)),
    ('LinearRegression, log_price', mean_absolute_error(y_test, pred_log),
     np.sqrt(mean_squared_error(y_test, pred_log)), r2_score(y_test, pred_log)),
    ('Ridge, log_price', mean_absolute_error(y_test, pred_ridge),
     np.sqrt(mean_squared_error(y_test, pred_ridge)), r2_score(y_test, pred_ridge)),
    ('RandomForest, log_price', mean_absolute_error(y_test, pred_rf),
     np.sqrt(mean_squared_error(y_test, pred_rf)), r2_score(y_test, pred_rf)),
]
for name, mae, rmse, r2 in rows:
    print('%-34s %8.2f %8.2f %7.3f' % (name, mae, rmse, r2))
print('=' * 62)
print()

# ------------------------------------------------------------------
# 8. Which features matter? Two different answers.
#    latitude only takes values between 40.5 and 40.9, so a coefficient
#    like -1.86 does NOT mean "one more degree of latitude". To compare
#    the features fairly we standardise them first, so every coefficient
#    means "change of one standard deviation".
# ------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr_std = LinearRegression()
lr_std.fit(X_train_s, ylog_train)
print('After with standardised features, test MAE is %.2f (should be the same as Model 2)'
      % mean_absolute_error(y_test, np.clip(np.expm1(lr_std.predict(X_test_s)), 0, None)))
print()

print('Standardised coefficients (on log_price, sorted by |value|):')
coef = pd.Series(lr_std.coef_, index=X.columns).sort_values(key=abs, ascending=False)
for name, value in coef.items():
    print('  %-34s %+.4f' % (name, value))
print()
print('How to read this: one standard deviation more of a feature changes')
print('log_price by that amount, i.e. price by roughly (exp(value) - 1) * 100%%.')
print('  %s: exp(%.3f) - 1 = %+.1f%%'
      % (coef.index[0], coef.iloc[0], 100 * (np.exp(coef.iloc[0]) - 1)))
print('  %s: exp(%.3f) - 1 = %+.1f%%'
      % (coef.index[1], coef.iloc[1], 100 * (np.exp(coef.iloc[1]) - 1)))
print()

print('RandomForest importance (top 10):')
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
for name, value in imp.head(10).items():
    print('  %-34s %.3f' % (name, value))
print()

# ------------------------------------------------------------------
# 9. 5-fold cross validation on the training set
#    A single train/test split can be lucky or unlucky, so we also check
#    how stable the score is across 5 different folds.
# ------------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=123)
cv_mae = -cross_val_score(lr_log, X_train, ylog_train, cv=kf,
                          scoring='neg_mean_absolute_error')
print('5-fold cross validation of LinearRegression on log_price (MAE on log scale):')
print('  folds: %s' % np.round(cv_mae, 4).tolist())
print('  mean %.4f | std %.4f' % (cv_mae.mean(), cv_mae.std()))
print()

# ------------------------------------------------------------------
# 10. Save the predictions so the report can plot residuals
# ------------------------------------------------------------------
out = pd.DataFrame({
    'true_price': y_test.values,
    'pred_log_model': np.round(pred_log, 2),
    'pred_rf': np.round(pred_rf, 2),
})
out.to_csv('predictions.csv', index=False)
print('Saved predictions.csv with %d rows' % len(out))