# Week 5: clustering, the third data mining method in the report
#
# Q2 asked in the report: can the listings be grouped into a few types,
# and does each type have its own pricing logic?
#
# Note: clustering does not use the price to group the listings, otherwise
# the clusters would just be "cheap / medium / expensive" and we would
# learn nothing. Price is only used afterwards, to describe each group.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

np.random.seed(123)
df = pd.read_csv('clean.csv', parse_dates=['last_review'])

def distance(lat1, lon1, lat2, lon2):
    dx = (lon1 - lon2) * 111.0 * np.cos(np.radians(lat1))
    dy = (lat1 - lat2) * 111.0
    return np.sqrt(dx ** 2 + dy ** 2)

df['dist_to_center'] = distance(df['latitude'], df['longitude'], 40.7580, -73.9855)

# ------------------------------------------------------------------
# 1. The features used for clustering
#    Everything must be on the same scale, otherwise latitude (40.x)
#    would dominate availability_365 (0-365). StandardScaler fixes that.
# ------------------------------------------------------------------
cluster_cols = ['latitude', 'longitude', 'dist_to_center', 'availability_365',
                'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count']

X = df[cluster_cols].values
X_scaled = StandardScaler().fit_transform(X)
print('Clustering on %d features, %d listings' % (len(cluster_cols), len(X)))
print('Features:', cluster_cols)
print()

# ------------------------------------------------------------------
# 2. How many clusters? Try k from 2 to 10 and look at two things:
#    - inertia (elbow method)
#    - silhouette score (higher is better, range -1 to 1)
# ------------------------------------------------------------------
inertia = []
silhouette = []
ks = range(2, 11)

for k in ks:
    km = KMeans(n_clusters=k, random_state=123, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X_scaled, labels, sample_size=5000, random_state=123))
    print('k = %2d   inertia = %12.1f   silhouette = %.4f' % (k, km.inertia_, silhouette[-1]))

best_k = list(ks)[int(np.argmax(silhouette))]
print()
print('Best k by silhouette score: %d (score %.4f)' % (best_k, max(silhouette)))
print('But the score is almost flat after k = 6, and a larger k only splits')
print('the groups into smaller pieces that are harder to explain.')

# pick the smallest k that is already close to the best score
threshold = max(silhouette) - 0.02
chosen_k = [k for k, s in zip(ks, silhouette) if s >= threshold][0]
print('Rule: smallest k within 0.02 of the best score  ->  k = %d' % chosen_k)
print()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(list(ks), inertia, 'o-')
axes[0].set_xlabel('k')
axes[0].set_ylabel('inertia')
axes[0].set_title('Elbow method')

axes[1].plot(list(ks), silhouette, 'o-')
axes[1].axvline(chosen_k, color='red', linestyle='--', linewidth=1)
axes[1].set_xlabel('k')
axes[1].set_ylabel('silhouette score')
axes[1].set_title('Silhouette score (higher is better)')
fig.tight_layout()
fig.savefig('figures/fig7_cluster_selection.png', dpi=150)
plt.show()

# ------------------------------------------------------------------
# 3. Fit the final model
# ------------------------------------------------------------------
km = KMeans(n_clusters=chosen_k, random_state=123, n_init=10)
df['cluster'] = km.fit_predict(X_scaled)

print('Cluster sizes:')
for c, n in df['cluster'].value_counts().sort_index().items():
    print('  cluster %d: %5d listings (%.1f%%)' % (c, n, 100.0 * n / len(df)))
print()

# ------------------------------------------------------------------
# 4. Show the clusters on a 2D PCA plot
# ------------------------------------------------------------------
pca = PCA(n_components=2, random_state=123)
coords = pca.fit_transform(X_scaled)
print('PCA explains %.1f%% + %.1f%% = %.1f%% of the variance'
      % (100 * pca.explained_variance_ratio_[0],
         100 * pca.explained_variance_ratio_[1],
         100 * pca.explained_variance_ratio_.sum()))
print()

fig, ax = plt.subplots(figsize=(7, 5.5))
scatter = ax.scatter(coords[:, 0], coords[:, 1], c=df['cluster'],
                     cmap='tab10', s=2, alpha=0.5)
ax.set_xlabel('PCA component 1 (%.1f%%)' % (100 * pca.explained_variance_ratio_[0]))
ax.set_ylabel('PCA component 2 (%.1f%%)' % (100 * pca.explained_variance_ratio_[1]))
ax.set_title('Listings grouped into %d clusters' % chosen_k)
fig.colorbar(scatter, ax=ax, label='cluster', ticks=range(chosen_k))
fig.tight_layout()
fig.savefig('figures/fig8_clusters_pca.png', dpi=150)
plt.show()

# ------------------------------------------------------------------
# 5. Describe each cluster. This is the part that answers Q2:
#    what kind of listing is each cluster, and how is it priced?
# ------------------------------------------------------------------
print('=' * 100)
print('%-8s %6s %8s %8s %9s %10s %-22s %-16s' %
      ('cluster', 'n', 'price', 'median', 'min_nights', 'avail_365', 'top borough', 'top room type'))
print('-' * 100)

for c in range(chosen_k):
    part = df[df['cluster'] == c]
    top_borough = part['neighbourhood_group'].value_counts().index[0]
    top_room = part['room_type'].value_counts().index[0]
    print('%-8d %6d %8.1f %8.0f %9.1f %10.0f %-22s %-16s'
          % (c, len(part), part['price'].mean(), part['price'].median(),
             part['minimum_nights'].median(), part['availability_365'].median(),
             top_borough, top_room))
print('=' * 100)
print()

print('Distance to the centre and borough mix per cluster:')
for c in range(chosen_k):
    part = df[df['cluster'] == c]
    mix = part['neighbourhood_group'].value_counts(normalize=True)
    mix_text = ', '.join('%s %.0f%%' % (k, 100 * v) for k, v in mix.head(2).items())
    print('  cluster %d: %.2f km from centre | %s' % (c, part['dist_to_center'].mean(), mix_text))
print()

# ------------------------------------------------------------------
# 6. Does price differ between the clusters?
#    If the clusters have clearly different prices, then "type of listing"
#    is a useful way to explain the price. This is the bridge to Q1.
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
data = [df[df['cluster'] == c]['price'] for c in range(chosen_k)]
ax.boxplot(data, tick_labels=['cluster %d' % c for c in range(chosen_k)])
ax.set_ylabel('price (USD per night)')
ax.set_title('Price distribution of each cluster')
fig.tight_layout()
fig.savefig('figures/fig9_price_by_cluster.png', dpi=150)
plt.show()

print('Median price per cluster: %s'
      % [float(df[df['cluster'] == c]['price'].median()) for c in range(chosen_k)])
print()

df.to_csv('clean_clustered.csv', index=False)
print('Saved clean_clustered.csv (clean.csv + the cluster column)')