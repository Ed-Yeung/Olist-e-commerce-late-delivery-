[Uploading README (12).md…]()
# Olist Late-Delivery Risk Scoring

**A time-aware machine-learning project that ranks Brazilian e-commerce orders by their risk of arriving after the promised delivery date.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?logo=pandas)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-EC6B23)](https://xgboost.readthedocs.io/)

This project uses the public **Olist Brazilian E-Commerce dataset (2016–2018)** to answer a practical operations question:

> At checkout, which orders should an operations team review first because they are more likely to arrive after the estimated delivery date?

The model is designed as a **risk-ranking tool**, not a guarantee that an individual order will be late. The goal is to concentrate more late orders into a limited review queue so that teams can prioritise shipment monitoring, seller follow-ups, or customer communication.

---

## Project at a glance

| | Result |
|---|---:|
| Overall late-delivery rate | **~6.8%** |
| August 2018 orders scored | 6,323 |
| Late rate in the highest-risk 10% | **17.3%** |
| Share of August late orders captured in the highest-risk 10% | **28%** |
| Strongest model on the supplied test results | **Logistic Regression** |

**Main takeaway:** the model is more useful for prioritising a small review queue than for making an automatic late/on-time decision for every order.

## 1. Business problem

An order is labelled late when its actual customer-delivery date is after the estimated delivery date shown to the customer. Late orders account for approximately 6.8% of the delivered orders in the overall dataset.

Accuracy alone is misleading for this problem. A classifier that predicts *on time* for every order would achieve approximately 93.2% accuracy while identifying none of the late orders.

The project therefore focuses on ranking orders by predicted late-delivery risk.

### August 2018: how useful is the ranking?

The model ranked 6,323 August orders from highest to lowest risk:

| August order group | Orders reviewed | Late orders found | Late rate within group |
|---|---:|---:|---:|
| All August orders | 6,323 | 390 | 6.2% |
| Highest-scoring 10% | 632 | 109 | **17.3%** |
| Highest-scoring 1% | 63 | 22 | **34.9%** |

The highest-risk 10% contains **109 of the 390 late orders**, capturing approximately **28% of all late orders**. A random 10% sample would be expected to contain roughly 10% of the late orders (about 39 in this example). The highest-risk 1% captures 5.6% of all late orders, compared with approximately 1% under random selection.

![Late and on-time order distribution](figures/late_delivery_share.png)

![Share of late orders captured by the highest-risk groups](figures/top_risk_capture.png)

This is the intended business value: **review fewer orders while finding a larger share of the orders that eventually arrive late**. It does not mean every flagged order will be late. The score is a prioritisation signal, not a customer-facing probability or a guarantee of delay.

## 2. Geographic distribution across Brazil

The map below is the **state-level late-delivery choropleth output from the notebook**. Each Brazilian state is shaded according to its observed late-delivery rate, using the colour scale shown on the map. Green represents lower rates; yellow and orange represent higher rates.

![Brazil state-level late-delivery rate choropleth generated in the notebook](figures/late_delivery_rate_by_state.png)

This view adds geographic context to the problem: late-delivery performance is not uniform across Brazil. The map should be interpreted together with the number of orders in each state, since a rate calculated from a small number of orders can be less stable than one based on a larger sample.

## 3. Data and preparation

The public dataset contains separate tables for orders, customers, order items, sellers, products, payments, and ZIP-code geolocation. These sources are joined and then aggregated back to **one row per order**, because late delivery is an order-level outcome.

### Target definition

An order is late when:

\[
\text{late\_delivery} = \mathbb{1}(\text{delivery date} > \text{estimated delivery date})
\]

The estimated delivery timestamp is stored at midnight, so the comparison is made at the **calendar-date level**, not by comparing the time of day.

### Main cleaning decisions

- Remove records without the delivery timestamps required to determine the target.
- Deduplicate geolocation to one coordinate record per ZIP-code prefix; keep the first coordinate record.
- Use inner joins for customer and seller ZIP-code coordinates; orders without matching coordinates are excluded.
- Remove cancelled orders and `not_defined` payment types.
- Convert payment type into boleto, credit-card, debit-card, and voucher indicators.
- Remove one order with an invalid shipping-limit date after checking the extreme `shipping_window_days` value.
- Aggregate item-level values to the order level before modelling.

## 4. Exploratory analysis

The monthly late-delivery rate changes substantially over time. It stays around 3% for much of 2017, rises sharply in late 2017 and early 2018, reaches approximately 18.9% in March 2018, and falls to approximately 1.2% in June 2018. This variation is one reason to evaluate the models chronologically rather than randomly splitting rows.

![Monthly late-delivery rate](figures/monthly_late_rate.png)

Three patterns stand out:

1. **Distance is associated with late delivery.** Orders travelling more than 1,000 km have a 10.4% late rate, compared with 4.4% for orders travelling less than 50 km.
2. **The promised delivery window is associated with the outcome.** Orders with a 1–7 day promise have a 15.1% late rate; those with an 8–14 day promise have a 4.5% late rate. This is an association, not evidence that a longer promise causes better delivery performance; the promised date is part of the target definition.
3. **Cross-state orders have a higher observed late rate.** The late rate is 8.0% for cross-state orders versus 4.5% for same-state orders. The boleto difference is smaller (7.3% versus 6.6%).

## 5. Feature engineering

Features are intended to represent information available when an order is placed. Variables that reveal the current order's future delivery outcome are not used as model inputs.

| Feature group | Examples | Purpose |
|---|---|---|
| Delivery windows | `shipping_window_days`, `promised_delivery_window_days` | Capture the seller handling window and delivery promise |
| Logistics | `avg_distance_km`, `same_state` | Represent shipment distance and whether seller/customer are in the same state |
| Order and product | Total price, freight, weight, volume; `multi_item`, `multi_seller` | Describe order value, size, and fulfilment complexity |
| Calendar | Cyclical hour/day features; holiday, Carnival, Black Friday, and Christmas flags | Capture recurring timing patterns |
| Historical performance | Seller, category, customer-state, and seller-state → customer-state route history | Summarise delivery outcomes already known before the current order |
| Recent platform performance | `platform_7d_shrunk_late_rate` | Record the platform's shrunk late-delivery rate over the preceding **7 days** |
| Interaction features | Distance × promised window; seller late rate × shipping window | Capture relationships between logistics and historical risk |

### Leakage-safe historical rates

Historical performance is only useful if it was known when the order was placed. Each completed delivery becomes observable at `order_delivered_customer_date`. A backward as-of join attaches cumulative history, with exact matches disabled so a delivery recorded at the same timestamp is not treated as prior knowledge.

Raw rates are unstable when there are few historical orders. They are shrunk toward an 8% prior with strength \(K=20\):

\[
\hat p = \frac{n_{late} + 20\times 0.08}{n+20}
\]

With no history, the estimate is 8%. As more observations accumulate, the group's own historical record receives more weight. The same shrinkage approach is applied to seller, category, customer-state, route, and recent platform performance.

For the platform feature, the recent window is **7 days**. It uses deliveries with a delivery timestamp in the interval \((t-7\text{ days},\ t)\), where \(t\) is the current order's purchase timestamp. Only deliveries completed strictly before the current order are counted. The 8% prior is frozen from the training period rather than recalculated using future validation or test outcomes.

### Time and interaction features

Two time windows are known at checkout and are represented in days:

\[
\text{shipping window} = \text{shipping limit} - \text{purchase time}
\]

\[
\text{promised window} = \text{estimated delivery date} - \text{purchase time}
\]

Hour and day of week are encoded cyclically so that 23:00 is close to 00:00, and Sunday is close to Monday:

\[
\sin(2\pi h/24),\ \cos(2\pi h/24),\qquad
\sin(2\pi d/7),\ \cos(2\pi d/7)
\]

Public holiday, Carnival, Black Friday, and Christmas-period indicators are based on the purchase date. Right-skewed variables such as price, freight, weight, volume, distance, and interaction terms are transformed with \(\log(1+x)\).

## 6. Modelling and evaluation

### Chronological data split

The split is based on purchase time so that validation and test orders occur after the training period.

| Dataset | Purchase period | Orders | Late orders | Late rate |
|---|---|---:|---:|---:|
| Train | Before 2018-03-01 | 56,988 | 3,758 | 6.59% |
| Validation | 2018-03 to 2018-05 | 20,448 | 2,065 | 10.10% |
| Test | 2018-06 to 2018-08 | 18,525 | 665 | 3.59% |

The change in late-delivery prevalence between validation and test matters when interpreting PR-AUC. The random-ranking PR-AUC baseline is approximately equal to the positive-class prevalence: about 10.1% on validation and 3.59% on test.

### Models

Three classifiers were compared using the same feature set:

- **Logistic Regression** — standardised numeric features and `class_weight='balanced'`.
- **Random Forest** — class weighting to give the minority late-delivery class more influence during fitting.
- **XGBoost** — `scale_pos_weight` calculated from the on-time and late order counts in the training data.

The `best_thr` value is selected using validation predictions and then carried forward for test-set classification. PR-AUC and ROC-AUC are calculated from predicted scores, so they evaluate ranking quality independently of a single classification threshold.

### Model comparison

| Model | Validation PR-AUC | Validation ROC-AUC | Best threshold | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.1801 | **0.6938** | 0.6964 | **0.0717** | **0.7001** |
| Random Forest | **0.2150** | 0.6631 | 0.6717 | 0.0302 | 0.4263 |
| XGBoost | 0.1740 | 0.6542 | 0.6581 | 0.0463 | 0.5886 |

**Logistic Regression is the strongest candidate on the supplied test results.** Random Forest has the highest validation PR-AUC, but its test PR-AUC falls below the test prevalence baseline and its test ROC-AUC is 0.4263. XGBoost is also weaker on the test period. Logistic Regression has the strongest test PR-AUC and ROC-AUC among the three models in this comparison.

The Logistic Regression test PR-AUC of 0.0717 is approximately twice the test prevalence baseline of 0.0359. This is a lift over random ranking, but the absolute PR-AUC remains modest; the model should be evaluated as a prioritisation tool, not a highly accurate order-level classifier.

### Why PR-AUC and ROC-AUC tell different stories

- **ROC-AUC** measures how well the model ranks late orders above on-time orders across thresholds. It is less directly affected by positive-class prevalence.
- **PR-AUC / Average Precision** focuses on precision and recall for the late class. Its baseline changes with the late-delivery rate, making it particularly useful for an imbalanced target.
- **Threshold metrics** such as Precision, Recall, and F1 depend on the selected threshold. They are important for operational decisions, but they do not replace threshold-independent ranking metrics.

## 7. What I learned

- **Class imbalance changes how success should be measured.** At a late rate of about 6.8%, an always-on-time classifier can achieve about 93.2% accuracy while finding no late orders. PR-AUC, Recall, Precision, and top-risk capture are more informative than accuracy alone.
- **Time-aware evaluation matters.** A model should be tested on later orders, not on a random mixture of past and future records. The target rate changes substantially from month to month.
- **Historical features need strict timing rules.** Seller, category, route, state, and platform performance must only include deliveries completed before the current order was placed. Shrinkage prevents small-history groups from producing extreme, unreliable rates.
- **A ranking can be useful even when a binary threshold is weak.** The operational question is often which 5% or 10% of orders to review first, not whether every order can be classified perfectly.

## Limitations and next steps

- The dataset covers historical orders from 2016–2018; current logistics patterns may differ.
- Late-delivery prevalence changes materially over time, so performance should be monitored by month and geography.
- A high-risk score is a prioritisation signal, not a calibrated probability or a guarantee of delay.
- The next evaluation step is rolling/expanding-window validation across multiple future periods, followed by a final untouched test period.
- A practical deployment decision should define review capacity and the relative cost of a missed late order versus an unnecessary intervention.

## Project structure

```text
.
├── README.md
└── figures/
    ├── late_delivery_share.png
    ├── late_delivery_rate_by_state.png
    ├── monthly_late_rate.png
    ├── top_risk_capture.png
    └── model_pr_auc_comparison.png
```
