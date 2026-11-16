# Week 3: exploratory data analysis
# Six figures, each one saved to the figures/ folder so it can be
# dropped straight into the report.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('clean.csv', parse_dates=['last_review'])
os.makedirs('figures', exist_ok=True)

# ------------------------------------------------------------------
# Figure 1: the distribution of price, before and after log
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].hist(df['price'], bins=50)
axes[0].set_title('price (raw)')
axes[0].set_xlabel('price (USD per night)')
axes[0].set_ylabel('number of listings')

axes[1].hist(df['log_price'], bins=50)
axes[1].set_title('log_price')
axes[1].set_xlabel('log(1 + price)')
axes[1].set_ylabel('number of listings')

fig.tight_layout()
fig.savefig('figures/fig1_price_distribution.png', dpi=150)
plt.show()

print('Figure 1: price is right-skewed. mean %.2f is much larger than median %.2f,'
      % (df['price'].mean(), df['price'].median()))
print('          and the raw histogram has a long tail on the right.')
print('          After log(), the distribution looks much more symmetric.')

# ------------------------------------------------------------------
# Figure 2: price by borough
# ------------------------------------------------------------------
groups = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
data = [df[df['neighbourhood_group'] == g]['price'] for g in groups]

fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot(data, tick_labels=groups)
ax.set_ylabel('price (USD per night)')
ax.set_title('Distribution of price by borough')
fig.tight_layout()
fig.savefig('figures/fig2_price_by_borough.png', dpi=150)
plt.show()

for g in groups:
    print('  %-14s median price = %6.1f   n = %d'
          % (g, df[df['neighbourhood_group'] == g]['price'].median(),
             (df['neighbourhood_group'] == g).sum()))

# ------------------------------------------------------------------
# Figure 3: the map. longitude on x, latitude on y, colour = price
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6))
sc = ax.scatter(df['longitude'], df['latitude'],
                c=df['price'], s=2, cmap='viridis', alpha=0.6)
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_title('Listings on the map, coloured by price')
fig.colorbar(sc, ax=ax, label='price (USD per night)')
fig.tight_layout()
fig.savefig('figures/fig3_price_map.png', dpi=150)
plt.show()

print('Figure 3: the expensive listings are concentrated in the middle of')
print('          Manhattan, and the cheap ones are far from the centre.')

# ------------------------------------------------------------------
# Figure 4: price by room type
# ------------------------------------------------------------------
rooms = ['Entire home/apt', 'Private room', 'Shared room']
data = [df[df['room_type'] == r]['price'] for r in rooms]

fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot(data, tick_labels=rooms)
ax.set_ylabel('price (USD per night)')
ax.set_title('Distribution of price by room type')
fig.tight_layout()
fig.savefig('figures/fig4_price_by_room_type.png', dpi=150)
plt.show()

for r in rooms:
    print('  %-16s median price = %6.1f   n = %d'
          % (r, df[df['room_type'] == r]['price'].median(),
             (df['room_type'] == r).sum()))

# ------------------------------------------------------------------
# Figure 5: correlation heatmap of the numeric columns
# ------------------------------------------------------------------
num_cols = ['price', 'log_price', 'minimum_nights', 'number_of_reviews',
            'reviews_per_month', 'calculated_host_listings_count',
            'availability_365']
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=45, ha='right')
ax.set_yticks(range(len(num_cols)))
ax.set_yticklabels(num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, '%.2f' % corr.iloc[i, j], ha='center', va='center', fontsize=7)
ax.set_title('Correlation between numeric features')
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig('figures/fig5_correlation.png', dpi=150)
plt.show()

print('Figure 5: price has almost no correlation with any other numeric column.')
print('          log_price correlates with price by %.2f, as expected.'
      % df['price'].corr(df['log_price']))

# ------------------------------------------------------------------
# Figure 6: number of reviews vs price
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df['number_of_reviews'], df['price'], s=2, alpha=0.3)
ax.set_xlabel('number of reviews')
ax.set_ylabel('price (USD per night)')
ax.set_title('Price vs number of reviews')
fig.tight_layout()
fig.savefig('figures/fig6_reviews_vs_price.png', dpi=150)
plt.show()

print('Figure 6: popular listings (many reviews) are not systematically')
print('          more expensive. The correlation is %.3f.'
      % df['price'].corr(df['number_of_reviews']))