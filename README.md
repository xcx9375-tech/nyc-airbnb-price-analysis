# What Really Decides an Airbnb Price?

A data mining study of 48,410 New York City listings — comparing boroughs,
room types, distance to the centre, and the type of host behind each listing.

Course: K6312 Machine Learning — Project Report

## Research questions

- **Q1** Which factors really drive the nightly price of a listing?
- **Q2** Can the listings be grouped into a few types, and does each type
  follow its own pricing logic?

## Data

`AB_NYC_2019.csv` — *New York City Airbnb Open Data*, Kaggle
(<https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data>).

- 48,895 rows × 16 columns, one row per listing
- Collected in 2019, covers all five New York boroughs

## How to run

```
python 03_clean.py      # produces clean.csv
python 04_eda.py        # produces figures/fig1 .. fig6
python 05_model.py      # regression models, produces predictions.csv
python 06_cluster.py    # clustering, produces clean_clustered.csv, fig7 .. fig9
```

Run them in this order — every script reads the output of the previous one.
Requirements: `pandas`, `numpy`, `scikit-learn`, `matplotlib`.
All random seeds are fixed (`random_state=123`), so the numbers are reproducible.

## What each script does

| File | Purpose |
| --- | --- |
| `03_clean.py` | Five cleaning steps, prints a log of how many rows each step removes |
| `04_eda.py` | Six figures: price distribution, price by borough / room type, price map, correlations, reviews vs price |
| `05_model.py` | Baseline + LinearRegression / Ridge / RandomForest, with and without `log_price`, plus 5-fold cross validation |
| `06_cluster.py` | KMeans with elbow + silhouette, PCA visualisation, and a profile of every cluster |

## Results (test set, 12,103 listings)

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Baseline (median) | 69.11 | 108.32 | −0.098 |
| LinearRegression, price | 51.43 | 81.69 | 0.376 |
| LinearRegression, log_price | 47.07 | 82.26 | 0.367 |
| Ridge, log_price | 47.07 | 82.26 | 0.367 |
| RandomForest, log_price | **41.96** | **73.35** | **0.497** |

Main findings:

1. Every model beats the median baseline by a wide margin — the features carry real signal.
2. Fitting `log(price)` lowers the MAE (51.43 → 47.07) but not the RMSE, because
   the back-transformation punishes the very expensive listings.
3. The random forest is the best model (R² 0.50), which suggests the relationship
   between the features and the price is not linear.
4. `room_type` is the strongest single factor: an entire home costs about
   **+22%** compared with the same listing as a private room, and distance to
   Times Square costs about **−15%** per standard deviation.
5. Six clusters emerge, ranging from *cheap private rooms far from the centre*
   (median 70 USD) to *large Manhattan apartments* (median 252 USD).

## Limitations

- Prices are the **listed** prices, not the paid ones; the real transaction
  price is unknown.
- The 2019 snapshot predates the 2023 New York short-term rental regulation,
  so the market has changed since then.
- `price` above the 99th percentile (799 USD) was removed, so the model
  cannot predict the luxury segment.
- The dataset has no information about the interior of a listing (size,
  number of bedrooms, renovation), which is certainly part of the price.

## Repository layout

```
AB_NYC_2019.csv        raw data
clean.csv              cleaned data (generated)
clean_clustered.csv    cleaned data + cluster label (generated)
predictions.csv        test-set predictions (generated)
03_clean.py 04_eda.py 05_model.py 06_cluster.py
figures/               all figures used in the report
```