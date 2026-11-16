# Week 2: clean the Airbnb data
# Every step prints how many rows changed, so the output can be copied
# directly into the "Data Preprocessing" section of the report.

import pandas as pd
import numpy as np

df = pd.read_csv('AB_NYC_2019.csv')
n0 = len(df)
price_mean_before = df['price'].mean()
price_median_before = df['price'].median()
price_max_before = df['price'].max()
print('Original data: %d rows, %d columns' % (df.shape[0], df.shape[1]))
print()

# ------------------------------------------------------------------
# Step 1: remove listings with price == 0
# A listing cannot really cost 0 USD per night, so these rows are
# recording errors.
# ------------------------------------------------------------------
n_before = len(df)
df = df[df['price'] > 0]
print('Step 1  remove price == 0        : -%d rows, %d left' % (n_before - len(df), len(df)))

# ------------------------------------------------------------------
# Step 2: remove the most expensive 1% of the listings
# price is very long-tailed (max is 10000), and a few extremely
# expensive listings would dominate MAE / RMSE later.
# ------------------------------------------------------------------
limit = df['price'].quantile(0.99)
n_before = len(df)
df = df[df['price'] <= limit]
print('Step 2  remove price > %-6.0f      : -%d rows, %d left' % (limit, n_before - len(df), len(df)))

# ------------------------------------------------------------------
# Step 3: reviews_per_month
# 10052 listings have never been reviewed, so pandas stores NaN.
# The real meaning is "no review yet", which is 0 reviews per month.
# ------------------------------------------------------------------
n_missing = df['reviews_per_month'].isna().sum()
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
print('Step 3  fill reviews_per_month   : %d NaN replaced by 0' % n_missing)

# ------------------------------------------------------------------
# Step 4: last_review is stored as text, convert it into a date
# ------------------------------------------------------------------
df['last_review'] = pd.to_datetime(df['last_review'])
print('Step 4  last_review -> datetime   : dtype is now %s' % df['last_review'].dtype)

# ------------------------------------------------------------------
# Step 5: add log_price
# price is right-skewed, log makes the distribution closer to normal
# ------------------------------------------------------------------
df['log_price'] = np.log1p(df['price'])
print('Step 5  add column log_price      : %d columns now' % df.shape[1])

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
df.to_csv('clean.csv', index=False)
print()
print('Saved clean.csv: %d rows, %d columns (%.1f%% of the original rows kept)'
      % (df.shape[0], df.shape[1], 100.0 * df.shape[0] / n0))
print()
print('price before / after cleaning:')
print('  mean   %.2f -> %.2f' % (price_mean_before, df['price'].mean()))
print('  median %.2f -> %.2f' % (price_median_before, df['price'].median()))
print('  max    %d -> %d' % (price_max_before, df['price'].max()))

print()
print('--- cleaning log (for the report) ---')
print('rows: %d -> %d' % (n0, len(df)))
print('columns: 16 -> %d (added log_price)' % df.shape[1])