[README (9).md](https://github.com/user-attachments/files/34166642/README.9.md)
# Olist Late-Delivery Risk Scoring

**A time-aware machine-learning project to rank Brazilian e-commerce orders by late-delivery risk.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-EC6B23)](https://xgboost.readthedocs.io/)
[![Folium](https://img.shields.io/badge/Maps-Folium-77B829)](https://python-visualization.github.io/folium/)

The project uses the public **Olist Brazilian E-Commerce dataset (2016–2018)**. The business question is straightforward: **when an order is placed, which orders should an operations team review first because they are more likely to arrive after the promised date?**

The output is a **risk ranking**, not a guaranteed late/on-time decision. When operational capacity is limited, the score helps prioritise which orders to investigate first.

---

## Project at a glance

| | Result |
|---|---:|
| Overall late-delivery rate | ~6.8% |
| August 2018 orders scored | 6,323 |
| Late rate in August's highest-risk 10% | **17.3%** |
| Late orders captured in that top 10% | **28%** of all August late orders |
| Model selected for ranking | Logistic Regression |

> **Main takeaway:** the model is more useful for prioritising a review queue than for making an automatic promise that an individual order will be late.

## 1. Business question and use case

A customer sees an estimated delivery date at checkout. The operations team would like to identify orders that may miss that date early enough to take action — for example, checking with a seller, monitoring a shipment, or proactively communicating with the customer.

The model assigns each order a score. Orders are sorted from highest to lowest risk, allowing the team to focus on a manageable group rather than treating every order equally.

### August 2018: what does the ranking achieve?

The model was trained using orders purchased before August 2018, then used to rank the 6,323 orders purchased in August.

| August order group | Orders | Late orders | Late rate |
|---|---:|---:|---:|
| All orders | 6,323 | 390 | 6.2% |
| Highest-scoring 10% | 632 | 109 | **17.3%** |
| Highest-scoring 1% | 63 | 22 | **34.9%** |

The highest-scoring 10% contains **109 of the 390 late orders (28%)**. A randomly selected 10% would be expected to contain roughly 10% of late orders. The highest-scoring 1% contains 22 late orders, or 5.6% of all August late orders.

<p align="center">
  <img src="figures/top_slice_late_rate.png" alt="Late-delivery rate within the highest-risk order groups" width="48%" />
  <img src="figures/late_orders_caught.png" alt="Share of late orders captured by the highest-risk groups" width="48%" />
</p>

This is the practical value of the score: **concentrate more late orders into the smaller set of orders an operations team can review.** It does not mean every flagged order will be late. Most orders flagged by an F1-optimised binary threshold can still arrive on time, so the score is treated as a ranking signal rather than a customer-facing probability.

## 2. Geographic view: seller to customer

Customer and seller ZIP-code prefixes are linked to latitude/longitude coordinates. These coordinates are used to estimate the distance between the seller and customer with the Haversine formula.

<p align="center">
  <a href="figures/example_shipping_route.html">
    <img src="figures/example_shipping_route.png" alt="Illustrative seller-to-customer shipment route in Brazil" width="82%" />
  </a>
</p>

<p align="center">
  <a href="figures/example_shipping_route.html"><strong>Open the interactive Folium map</strong></a>
</p>

The map shows **one example shipment** from the supplied notebook: a seller in Curitiba, Paraná (PR), to a customer in São Joaquim da Barra, São Paulo (SP). The straight-line Haversine distance is approximately **566.5 km**. It is an illustration of the geospatial feature, not a density map of all orders or a road-navigation route.

For each item, distance is calculated as:

\[
 a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right),
 \qquad d = 2R\arcsin(\sqrt{a}), \quad R=6{,}371\text{ km}
\]

For multi-seller orders, item-level distances are averaged to create an order-level distance feature.

## 3. Data and preparation

The raw data is split across orders, customers, order items, sellers, products, payments, and ZIP-code geolocation tables. These sources are joined and then aggregated back to **one row per order**, because late delivery is an order-level outcome.

**Target definition**

An order is late when its actual customer-delivery date falls after its estimated delivery date. The estimated date is stored at midnight, so the comparison is made at the **calendar-date level**, not by comparing the time of day.

**Key cleaning decisions**

- Remove records without the delivery timestamps required to determine the target.
- Deduplicate geolocation to one coordinate record per ZIP-code prefix; keep the first record and drop the city label from the coordinate lookup.
- Use inner joins for customer and seller ZIP-code coordinates; orders without a matching location are excluded.
- Remove cancelled orders and `not_defined` payment types.
- Convert payment type into boleto, credit-card, debit-card, and voucher indicators.
- Remove one order with an invalid shipping-limit date after checking the extreme `shipping_window_days` value.

The resulting modelling table is kept at one row per order. Item-level values such as price, freight, weight, volume, distance, and historical rates are aggregated before modelling.

## 4. What the exploratory analysis shows

Three patterns stand out before fitting a model:

1. **Distance matters.** Orders travelling more than 1,000 km have a 10.4% late rate, compared with 4.4% for orders travelling less than 50 km.
2. **The promised window matters.** Orders with a 1–7 day promise have a 15.1% late rate; those with an 8–14 day promise have a 4.5% late rate. The promise is part of the target definition, so this is an association — not evidence that a longer promise causes better delivery performance.
3. **Cross-state shipments are riskier in this dataset.** The late rate is 8.0% for cross-state orders versus 4.5% for same-state orders. The boleto difference is small (7.3% versus 6.6%) and is not treated as a meaningful driver on its own.

<p align="center">
  <img src="figures/late_share.png" alt="Overall share of late and on-time orders" width="48%" />
  <img src="figures/monthly_late_rate.png" alt="Monthly late-delivery rate" width="48%" />
</p>

<p align="center">
  <img src="figures/distance_and_promise.png" alt="Late-delivery rates by distance and promised delivery window" width="48%" />
  <img src="figures/same_state_and_payment.png" alt="Late-delivery rates by same-state status and payment type" width="48%" />
</p>

The monthly rate is not stable. It is near 3% for much of 2017, rises sharply to 18.6% in March 2018, and falls to 1.1% in June 2018. This change in the target rate is why a random row split would be a weak test of future performance.

## 5. Feature engineering

Features are designed to describe information available by the time an order is placed, while avoiding the use of that order's future delivery outcome.

| Feature group | Examples | Why it is included |
|---|---|---|
| Delivery windows | `shipping_window_days`, `promised_delivery_window_days` | Capture the seller's handling window and the delivery promise |
| Logistics | `avg_distance_km`, `same_state` | Represent shipment distance and whether seller/customer are in the same state |
| Order and product | Total price, freight, weight, volume; `multi_item`, `multi_seller` | Describe order value, size, and fulfilment complexity |
| Calendar | Cyclical hour/day features; public holiday, Carnival, Black Friday, Christmas flags | Capture recurring timing patterns |
| Historical performance | Seller, category, customer-state, and seller-to-customer route history | Add past delivery experience available before the current order |
| Recent platform performance | `platform_28d_shrunk_late_rate` | Capture the platform's recent late-delivery level |

### Leakage-safe historical rates

Historical performance is only useful if it was known when the order was placed. Each completed delivery becomes observable at `order_delivered_customer_date`. A backward as-of join is used to attach cumulative history, with exact matches disabled so that a delivery recorded at the same timestamp is not treated as prior knowledge.

Raw rates are unstable when there are few historical orders. They are shrunk toward an 8% prior with strength \(K=20\):

\[
\hat p = \frac{n_{late} + 20\times 0.08}{n+20}
\]

With no history, the estimate is 8%. As more observations accumulate, the group's own historical record receives more weight. The same shrinkage idea is applied to seller, category, customer-state, and route history.

The supplied notebook implements the recent platform feature with a **28-day window**. The 8% prior is frozen from the training period rather than recalculated using future test outcomes.

### Time and interaction features

Hour and day of week are encoded cyclically, so that 23:00 is close to 00:00 and Sunday is close to Monday:

\[
\sin(2\pi h/24),\ \cos(2\pi h/24),\qquad
\sin(2\pi d/7),\ \cos(2\pi d/7)
\]

Two interaction features are added:

- Distance × promised delivery window
- Seller historical late rate × distance

Right-skewed monetary, freight, physical-size, distance, and interaction features are transformed with `log1p`. Item-level features are then aggregated to the order level; item/seller counts are represented as `multi_item` and `multi_seller` flags because most orders contain one item and one seller.

## 6. Modelling and evaluation

### Time-based split

The split follows purchase time rather than randomly assigning rows:

| Dataset | Period | Purpose |
|---|---|---|
| Train | Before 1 March 2018 | Fit preprocessing and model parameters |
| Validation | March–May 2018 | Compare models and select each model's decision threshold |
| Test | June 2018 onward | Final out-of-time evaluation |

The validation period deliberately includes the March 2018 late-delivery spike. The test period has a much lower late-delivery rate, so the model is evaluated under a different target prevalence.

For a reproducible chronological split, sort by purchase time before applying the date cut-offs:

```python
df_full_v5["order_purchase_timestamp"] = pd.to_datetime(
    df_full_v5["order_purchase_timestamp"]
)
df_full_v5 = df_full_v5.sort_values("order_purchase_timestamp").reset_index(drop=True)

train = df_full_v5[df_full_v5["order_purchase_timestamp"] < "2018-03-01"]
valid = df_full_v5[
    (df_full_v5["order_purchase_timestamp"] >= "2018-03-01") &
    (df_full_v5["order_purchase_timestamp"] < "2018-06-01")
]
test = df_full_v5[df_full_v5["order_purchase_timestamp"] >= "2018-06-01"]
```

> **Notebook check before publishing:** the supplied HTML export still contains an earlier 80/20 `.iloc` split and `TimeSeriesSplit` cells operating on the dataframe's existing row order. The displayed fold date ranges overlap, so those cells are not chronological as saved. Update them to sort by `order_purchase_timestamp` (or use the date masks above), then rerun the reported comparison so the notebook and README describe the same experiment.

### Models

Three classifiers are compared using the same feature set:

- **Logistic Regression:** `class_weight='balanced'` with standard scaling.
- **Random Forest:** class-weight balancing to give the minority late class more influence.
- **XGBoost:** `scale_pos_weight` set from the ratio of on-time to late orders in the training data.

The threshold is selected on the validation set by maximising F1, then frozen before test evaluation. The risk-ranking use case does not depend on converting every score into a hard label.

### Model comparison

| Model | Valid PR-AUC | Valid ROC-AUC | Best threshold | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.1801 | 0.6938 | 0.6964 | **0.0717** | **0.7001** |
| Random Forest | **0.2150** | 0.6631 | 0.6717 | 0.0302 | 0.4263 |
| XGBoost | 0.1740 | 0.6542 | 0.6581 | 0.0463 | 0.5886 |

**Logistic Regression is retained for the final ranking.** Random Forest has the highest validation PR-AUC, but its test ROC-AUC falls below 0.5, indicating that its ranking is worse than random on this test window. XGBoost also weakens out of time. Logistic Regression has the strongest test PR-AUC and ROC-AUC among the three models in this comparison.

PR-AUC depends on the late-delivery prevalence, so scores from periods with different late rates should not be compared without also looking at the baseline rate. ROC-AUC is less directly affected by class prevalence, while PR-AUC is more informative for this imbalanced target.

### Monthly walk-forward check

A separate month-by-month check refits the logistic model using only orders purchased before each month, then evaluates that month. This shows how ranking quality changes as the underlying late-delivery rate changes.

| Month | Orders | Late rate | ROC-AUC | PR-AUC | Late rate in top 10% |
|---|---:|---:|---:|---:|---:|
| 2018-03 | 7,170 | 18.6% | 0.71 | 0.36 | 46.7% |
| 2018-04 | 7,015 | 4.5% | 0.78 | 0.14 | 16.0% |
| 2018-05 | 6,900 | 6.4% | 0.74 | 0.16 | 19.9% |
| 2018-06 | 6,224 | 1.1% | 0.78 | 0.05 | 4.0% |
| 2018-07 | 6,438 | 3.6% | 0.69 | 0.10 | 12.0% |
| 2018-08 | 6,323 | 6.2% | 0.70 | 0.15 | 17.3% |

ROC-AUC stays within roughly 0.69–0.78 across these months, while PR-AUC moves with the late-delivery base rate. This is expected: when late orders become rarer, precision-recall performance is harder to maintain even if the model's ranking ability is broadly similar.

The largest standardised Logistic Regression coefficients are directionally consistent with the exploratory analysis: distance, historical state/route performance, and seller handling time contribute to the ranking, while a longer promised delivery window is associated with a lower late-risk score. Coefficients describe conditional associations within this model; they should not be interpreted as causal effects.

<p align="center">
  <img src="figures/coefficients.png" alt="Standardised Logistic Regression coefficients" width="78%" />
</p>

## 7. What I learned

- **Class imbalance changes how success should be measured.** At a late rate of about 6.8%, an always-on-time classifier would already achieve 93.2% accuracy while finding no late orders. PR-AUC, Recall, Precision, and top-risk capture are more useful than accuracy alone.
- **Time-aware evaluation matters.** A model should be tested on later orders, not on a random mixture of past and future records. Monthly monitoring also helps expose changes in delivery conditions.
- **Historical features need strict timing rules.** Seller or route performance must only include deliveries completed before the current order was placed. Shrinkage prevents small-history groups from producing extreme, unreliable rates.
- **A ranking can be useful even when a binary threshold is weak.** The operational question is often which 10% of orders to review first, not whether every order can be classified perfectly.

## Limitations and next steps

- The model is trained on historical 2016–2018 data; current logistics patterns may differ.
- The late-delivery rate changes materially over time, so the score needs ongoing monitoring and periodic evaluation.
- A high-risk score is a prioritisation signal, not a calibrated probability or a guarantee of delay.
- The next useful improvement is to validate the ranking across additional rolling time windows and define an operational review capacity (for example, the top 5% or 10%) before selecting a threshold.

## Data source

[Olist — Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). The data is anonymised and covers orders placed between 2016 and 2018.
