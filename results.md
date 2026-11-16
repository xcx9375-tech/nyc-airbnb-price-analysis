# Result summary

All numbers below are printed by 03_clean.py, 04_eda.py, 05_model.py and
06_cluster.py, collected here so they can be copied into the report.
Random seeds are fixed, so re-running the scripts reproduces them exactly.

## 1. Data

| Item | Value |
| --- | --- |
| Raw file | AB_NYC_2019.csv |
| Raw shape | 48895 rows x 16 columns |
| Clean shape | 48410 rows x 19 columns |
| Rows removed by cleaning | 485 (1.0%) |
| Rows kept | 99.0% |
| price missing before cleaning | 0 |
| price == 0 before cleaning | 11 |
| price > 799 before cleaning | 474 |
| reviews_per_month missing | 10052 |
| Date range of last_review | 2011-03-28 to 2019-07-08 |

Price after cleaning: mean 137.58, median 105, std 103.78, min 10, max 799

Rows per borough:
- Manhattan: 21312 (44.0%)
- Brooklyn: 19992 (41.3%)
- Queens: 5650 (11.7%)
- Bronx: 1087 (2.2%)
- Staten Island: 369 (0.8%)

Rows per room type:
- Entire home/apt: 25022 (51.7%)
- Private room: 22237 (45.9%)
- Shared room: 1151 (2.4%)

## 2. Exploratory analysis

Median price per borough:
- Manhattan: 149
- Brooklyn: 90
- Queens: 75
- Staten Island: 75
- Bronx: 65

Median price per room type:
- Entire home/apt: 160
- Private room: 70
- Shared room: 45

Correlation of each numeric feature with price:
- longitude: -0.2736
- number_of_reviews: -0.0556
- reviews_per_month: -0.0531
- minimum_nights: +0.0244
- latitude: +0.0671
- availability_365: +0.1133
- calculated_host_listings_count: +0.1513

## 3. Regression models

Training rows: 36307, testing rows: 12103, features: 16

| Model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| Baseline (median = 105) | 69.11 | 108.32 | -0.098 |
| LinearRegression, price | 51.43 | 81.69 | 0.376 |
| LinearRegression, log_price | 47.07 | 82.26 | 0.367 |
| Ridge, log_price | 47.07 | 82.26 | 0.367 |
| RandomForest, log_price | 41.96 | 73.35 | 0.497 |

Check: the standardised model has the same test MAE (47.07).

Standardised coefficients on log_price:

| Feature | Coefficient | Effect on price |
| --- | --- | --- |
| room_type_Entire home/apt | +0.1965 | +21.7% |
| dist_to_center | -0.1637 | -15.1% |
| room_type_Private room | -0.1629 | -15.0% |
| room_type_Shared room | -0.1117 | -10.6% |
| latitude | -0.1017 | -9.7% |
| availability_365 | +0.0885 | +9.3% |
| neighbourhood_group_Manhattan | +0.0740 | +7.7% |
| neighbourhood_group_Brooklyn | -0.0590 | -5.7% |
| minimum_nights | -0.0492 | -4.8% |
| neighbourhood_group_Queens | -0.0279 | -2.8% |
| number_of_reviews | -0.0210 | -2.1% |
| neighbourhood_group_Bronx | +0.0175 | +1.8% |
| neighbourhood_group_Staten Island | -0.0149 | -1.5% |
| longitude | -0.0144 | -1.4% |
| reviews_per_month | -0.0073 | -0.7% |
| calculated_host_listings_count | -0.0001 | -0.0% |

RandomForest feature importance:

| Feature | Importance |
| --- | --- |
| room_type_Entire home/apt | 0.492 |
| dist_to_center | 0.169 |
| longitude | 0.097 |
| availability_365 | 0.057 |
| latitude | 0.046 |
| reviews_per_month | 0.038 |
| minimum_nights | 0.036 |
| number_of_reviews | 0.031 |
| calculated_host_listings_count | 0.023 |
| room_type_Private room | 0.006 |
| room_type_Shared room | 0.003 |
| neighbourhood_group_Manhattan | 0.001 |
| neighbourhood_group_Brooklyn | 0.000 |
| neighbourhood_group_Queens | 0.000 |
| neighbourhood_group_Bronx | 0.000 |
| neighbourhood_group_Staten Island | 0.000 |

## 4. Clusters

KMeans on 7 standardised features (price NOT used for clustering).
k = 6 chosen by the elbow plus silhouette score.

| Cluster | n | Share | Mean price | Median price | Median min nights | Median avail 365 | Main borough | Main room type | km to centre |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 8965 | 18.5% | 169.1 | 132 | 4 | 318 | Manhattan | Entire home/apt | 5.2 |
| 1 | 19458 | 40.2% | 149.7 | 125 | 3 | 0 | Manhattan | Entire home/apt | 4.6 |
| 2 | 12916 | 26.7% | 105.0 | 80 | 2 | 9 | Brooklyn | Private room | 10.3 |
| 3 | 3287 | 6.8% | 128.6 | 107 | 2 | 203 | Brooklyn | Entire home/apt | 6.9 |
| 4 | 3225 | 6.7% | 92.8 | 70 | 2 | 166 | Queens | Private room | 15.6 |
| 5 | 559 | 1.2% | 273.9 | 252 | 29 | 317 | Manhattan | Entire home/apt | 3.8 |

Median price of each cluster, from cheapest to most expensive:
- cluster 4: 70
- cluster 2: 80
- cluster 3: 107
- cluster 1: 125
- cluster 0: 132
- cluster 5: 252

## 5. Figures

| File | Content |
| --- | --- |
| figures/fig1_price_distribution.png | price before and after the log transform |
| figures/fig2_price_by_borough.png | price per borough |
| figures/fig3_price_map.png | every listing on the map, coloured by price |
| figures/fig4_price_by_room_type.png | price per room type |
| figures/fig5_correlation.png | correlation heat map |
| figures/fig6_reviews_vs_price.png | number of reviews vs price |
| figures/fig7_cluster_selection.png | elbow and silhouette curves |
| figures/fig8_clusters_pca.png | the six clusters in 2D (PCA) |
| figures/fig9_price_by_cluster.png | price distribution of each cluster |

