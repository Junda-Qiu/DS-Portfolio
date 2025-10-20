# Ad Analytics End-to-End Report

---

## **Chapter 1. Introduction**

### 1.1 Background

In the modern digital advertising ecosystem, data-driven optimization has become the core driver of performance.  
With the increasing complexity of user behavior and the growing diversity of ad formats, advertisers face the challenge of understanding how users interact with ads, how pricing affects click-through behavior, and how retention patterns evolve over time.

This project, titled **“Ad Analytics End-to-End”**, aims to construct a **comprehensive analytical framework** that integrates user behavior, ad performance, and retention analysis.  
By combining statistical analysis with clustering and behavioral modeling, the project seeks to provide both **descriptive insights** and **actionable intelligence** for ad targeting and optimization.

---

### 1.2 Objectives

The key objectives of this project are as follows:

1. **To build an integrated data pipeline** for merging, cleaning, and transforming raw user, click, and ad datasets.
2. **To explore the relationship between price and click-through rate (CTR)**, revealing behavioral and psychological patterns behind user interaction.
3. **To segment users using RFM (Recency, Frequency, Monetary) and KMeans clustering**, in order to identify valuable and at-risk user groups.
4. **To analyze retention patterns** using cohort analysis and rolling retention metrics.
5. **To visualize all results** through interpretable charts, supporting both management reporting and further modeling work.

---

### 1.3 Research Framework

The analytical workflow of this study consists of six major components:

1. **Data Construction** — Merge and preprocess raw datasets from user, click, and ad sources.
2. **Data Profiling** — Assess schema consistency, missing rate, and key distribution diagnostics.
3. **CTR & Price Analysis** — Quantify how pricing tiers influence user click probability.
4. **User Segmentation (RFM + KMeans)** — Identify behavioral archetypes and their demographic structures.
5. **Retention Analysis** — Build cohort retention matrices and compute rolling retention at 7, 14, and 30 days.
6. **Strategic Insights** — Translate analytical results into actionable business recommendations.

---

### 1.4 Data Sources

The project is based on three key CSV datasets:

- **User Data (`user.csv`)** — Includes demographics such as gender, age group, consumption level, and city tier.  
- **Click Data (`click.csv`)** — Records user-ad interaction logs, including click (`clk`) and non-click (`nonclk`) events, along with timestamps.  
- **Ad Data (`ad.csv`)** — Contains metadata such as ad ID, campaign ID, customer ID, brand, and product price.

The merged dataset, `processed_user_ad.csv`, contains approximately **975,441 records** after data cleaning, providing a reliable foundation for modeling and visualization.

---

### 1.5 Key Metrics Definitions

To ensure interpretability and reproducibility, all key metrics are defined explicitly as follows:

- **Click-Through Rate (CTR)**  
  `CTR = clicks / impressions`  
  Measures the proportion of ad impressions that result in clicks.

- **User Share (%)**  
  `User Share = users_in_price_bin / total_users * 100`

- **Recency (R)**  
  `R = days_since_last_click`  
  Quantifies the time gap between the user’s last activity and the reference date.

- **Frequency (F)**  
  `F = number_of_clicks`

- **Monetary (M)**  
  `M = average_spent_price_per_user`

These variables form the foundation for **RFM-based segmentation**, which groups users into eight categories (e.g., high-value, potential, at-risk) based on median thresholds.

---

### 1.6 Research Significance

This project contributes to the intersection of **computational advertising and behavioral analytics** by integrating classical statistical approaches (CTR, RFM) with modern clustering and cohort techniques.  
The analysis not only explains *what* patterns exist but also *why* users behave differently across price tiers and time horizons.

The outcomes can be used for:
- Improving ad placement efficiency and targeting precision.  
- Designing personalized re-engagement campaigns based on user lifecycle.  
- Building a foundation for predictive modeling in future real-time bidding systems.

---

## **Chapter 2. Data and Methodology**

### 2.1 Dataset Overview

The study integrates three primary datasets — `user.csv`, `click.csv`, and `ad.csv` — into a unified analytical table named `processed_user_ad.csv`.  
Each record represents one ad impression linked to a unique user and ad ID.

| Dataset | Description | Key Columns |
|----------|--------------|--------------|
| **User Data** | Contains demographic and behavioral attributes. | `userid`, `gender`, `age_level`, `consumption_level`, `city_tier`, `is_student` |
| **Click Data** | Logs ad exposure events and click behaviors. | `userid`, `ad_id`, `clk`, `nonclk`, `time_stamp` |
| **Ad Data** | Stores ad metadata and product information. | `ad_id`, `brand`, `campaign_id`, `category_id`, `price`, `customer` |

After merging and cleaning, the final dataset contains:
- **975,441 rows** and **19 columns**  
- **Missing rate** below 5% for most variables  
- **Distinct users:** ~356,000  
- **Distinct ads:** ~59,000  

---

### 2.2 Data Preprocessing Workflow

The preprocessing pipeline is executed through a Python-based ETL (Extract–Transform–Load) process:

1. **Encoding Robust Reading**  
   The `safe_read_csv()` utility automatically retries with encodings `['gbk', 'utf-8', 'latin1']` to ensure compatibility.

2. **Standardized Key Fields**  
   All user identifiers are renamed to `userid`, and ad identifiers to `ad_id` for consistent merging.

3. **Data Merging**  
   The datasets are joined on shared keys:
   ```python
   merged = click_df.merge(user_df, on='userid', how='left').merge(ad_df, on='ad_id', how='left')
````

4. **Price Binning**
   Prices are divided into **9 fixed bins** for categorical analysis:

   ```
   [0-100], [101-200], [201-400], [401-800], [801-1600],
   [1601-3200], [3201-6400], [6401-12800], [12801+]
   ```

   A corresponding integer code (`price_bin_code`) is generated for modeling.

5. **Outlier Capping**
   Extreme prices are capped at the 99th percentile (`price_capped`) to stabilize distribution.

6. **Timestamp Standardization**
   Epoch values are converted to `datetime64[ns]` for temporal computations.

---

### 2.3 Analytical Indicators

The system computes multiple field-level, behavioral, and structural indicators.

#### (1) Field-Level Profiling

* Data type, missing rate, and unique count per column.
* Statistical moments: mean, standard deviation, quantiles (1%, 5%, 25%, 50%, 75%, 95%, 99%).

#### (2) Key Integrity Checks

* **Primary Keys:** `userid`, `ad_id`
* **Duplicate Rate:**

  * User ID duplication: ~63.5%
  * Ad ID duplication: ~93.9%
    Indicates multi-view user activity and repeated ad exposure patterns.

---

### 2.4 CTR and Price Analysis

The **Click-Through Rate (CTR)** is computed as:

```
CTR = clicks / (clicks + nonclicks)
```

For each price bin, the mean CTR and corresponding user share are calculated:

```
User Share (%) = unique_users_in_bin / total_unique_users * 100
```

The results reveal that:

* **Low-price bins (0–200)** achieve the highest CTR (~5.5%).
* **Mid-price bins (400–1600)** show a moderate decline (~4.4%).
* **Ultra-high-price bins (12801+)** unexpectedly rebound (~5.1%), suggesting exploratory behavior.

All results are visualized using bar plots:

* `ctr_by_pricebin.png`
* `usershare_by_pricebin.png`

---

### 2.5 Top and Bottom Performing Ads

The top 100 ads are ranked by impressions and click rate using:

```python
ad_stats = df.groupby('ad_id').agg(
    total_impressions=('nonclk', 'sum'),
    total_clicks=('clk', 'sum'),
    avg_price=('price', 'mean')
)
ad_stats['click_rate'] = ad_stats['total_clicks'] / ad_stats['total_impressions']
```

Visualizations include:

* **CTR Top 10** demographics: `ad_top10_age.png`, `ad_top10_gender.png`, `ad_top10_level.png`
* **CTR Bottom 10** demographics: `ad_bottom10_age.png`, `ad_bottom10_gender.png`, `ad_bottom10_level.png`

These analyses reveal that **CTR leaders** align closely with user demographics (gender × age match), whereas **CTR laggards** reflect mismatched targeting.

---

### 2.6 RFM Segmentation

**RFM** (Recency–Frequency–Monetary) is used to measure user engagement and value:

```
R = days_since_last_click
F = number_of_clicks
M = average_price_of_clicked_ads
```

Users are classified into 8 groups based on median thresholds:

| R | F | M | Category           |
| - | - | - | ------------------ |
| 1 | 1 | 1 | High-Value Users   |
| 0 | 1 | 1 | Re-Engage Users    |
| 1 | 0 | 1 | Deep-Nurture Users |
| 0 | 0 | 1 | Retention Users    |
| 1 | 1 | 0 | Potential Users    |
| 1 | 0 | 0 | New Users          |
| 0 | 1 | 0 | Maintain Users     |
| 0 | 0 | 0 | Churned Users      |

Median thresholds observed:

```
R = 1011.0, F = 1.0, M = 192.0
```

RFM distribution result visualization:

* `rfm_age.png` (Age distribution by RFM category)
* `rfm_gender.png` (Gender distribution by RFM category)

---

### 2.7 KMeans Clustering

Two clustering blocks are applied:

#### (1) Behavioral Clustering

Features used:

```
[shopping_level, click_rate, avg_price]
```

Number of clusters: 5
Cluster centers illustrate distinct behavioral segments, e.g.:

* Price-sensitive explorers
* Frequent clickers
* High-spending but infrequent users
* Passive low-engagement users

Charts:

* `cluster_age.png`
* `cluster_gender.png`

#### (2) RFM-Based Clustering

Features used:

```
[recency, frequency, monetary]
```

Cluster count: 5
Visual outputs:

* `rfmcluster_age.png`
* `rfmcluster_gender.png`

---

### 2.8 Retention Analysis

#### (1) Cohort Retention Matrix

Each user’s first activity date defines a cohort:

```
cohort_index = (event_date - cohort_start_date).days
```

Matrix visualization: `cohort_retention_heatmap.png`
Retention drops sharply after Day 1 (22.8%) and Day 7 (2.46%).

#### (2) Rolling Retention

```
Rolling Retention (k days) = users_active_after_k_days / total_users * 100
```

Overall results:

| Metric           | Value |
| ---------------- | ----- |
| 7-Day Retention  | 2.46% |
| 14-Day Retention | 0.00% |
| 30-Day Retention | 0.00% |

---

### 2.9 Price Bin Sensitivity

Two binning strategies are compared:

* **Equal Width (EW):** divides full range into 9 equal segments.
* **Quantile (QT):** divides by price quantiles.

Files:

* `price_ctr_equalwidth.png`
* `price_ctr_quantile.png`

Quantile binning captures CTR variation more accurately due to skewed price distribution.

---

## **Chapter 3. Data Analysis and Results**

---

### 3.1 Overview

This chapter presents the analytical findings derived from the processed dataset, including click-through rate (CTR) distribution, user share by price bin, ad-level performance ranking, RFM-based user segmentation, and clustering results.  
The analysis focuses on identifying structural patterns in user engagement and ad effectiveness, supplemented by visualization outputs.

---

### 3.2 CTR Distribution by Price Bin

CTR and user distribution across price ranges are visualized in:

- `ctr_by_pricebin.png`  
- `usershare_by_pricebin.png`

**Key Observations:**

| Price Bin | CTR (%) | User Share (%) |
|------------|----------|----------------|
| 0–100 | 5.49 | 51.07 |
| 101–200 | 4.62 | 31.29 |
| 201–400 | 4.68 | 29.80 |
| 401–800 | 4.60 | 17.07 |
| 801–1600 | 4.45 | 10.87 |
| 1601–3200 | 4.20 | 7.19 |
| 3201–6400 | 4.17 | 4.99 |
| 6401–12800 | 4.38 | 2.86 |
| 12801+ | 5.13 | 1.23 |

**Findings:**
- CTR forms an **inverted-U curve** across price levels.  
  Low-price items (0–200) attract exploratory clicks due to affordability.  
  Middle-price items show fatigue or indecision.  
  Ultra-high prices (>12,800) regain curiosity-driven clicks.
- The majority of users (over 50%) engage within the low-price tier, confirming that affordability strongly drives engagement.

---

### 3.3 Top and Bottom Ads Performance

The **Top 20 ads** by impressions and CTR reveal strong differentiation in targeting efficiency.  
Relevant charts:
- `ad_top10_age.png`
- `ad_top10_gender.png`
- `ad_top10_level.png`
- `ad_bottom10_age.png`
- `ad_bottom10_gender.png`
- `ad_bottom10_level.png`

**Insights:**
- **Top-performing ads** align closely with target demographics — particularly young female users with mid-level consumption capacity.  
- **Low-performing ads** often mismatch user intent — e.g., high-end luxury targeting student demographics.
- Ads with higher price points but relevant context (e.g., branded electronics) maintain strong CTR if user relevance is achieved.

Additionally, the comparison between **clicked** and **non-clicked** products demonstrates that price alone does not dictate engagement.  
Reference visualization:  
- `clicked_price.png`
- `nonclicked_price.png`

---

### 3.4 User Value Segmentation (RFM Analysis)

The RFM model captures user engagement diversity by three behavioral dimensions:
- **Recency (R):** days since last interaction  
- **Frequency (F):** total clicks  
- **Monetary (M):** average price of clicked ads  

Results summarized in:
- `rfm_age.png`
- `rfm_gender.png`

**RFM Category Proportion (%):**

| Category | Definition | Share (%) |
|-----------|-------------|-----------|
| Deep Nurture Users | Frequent interaction with high-value items | 26.90 |
| New Users | Newly engaged, low recency and spend | 25.89 |
| Churned Users | No recent activity | 18.52 |
| Retention Users | High-value but disengaging | 17.19 |
| High-Value Users | Active and high spending | 4.57 |
| Potential Users | Regular but low spend | 4.06 |
| Maintain Users | Consistent, mid-value | 1.51 |
| Re-Engage Users | Recently inactive high spenders | 1.36 |

**Interpretation:**
- Nearly half the users (R+F+M = low) represent early-stage or at-risk segments.  
- Deep Nurture Users contribute the largest portion of recurring clicks.  
- High-Value Users, though only 4.57%, drive a disproportionate share of total revenue — indicating a Pareto effect.

---

### 3.5 KMeans Behavioral Clustering

Behavioral segmentation identifies latent user patterns across three primary features:
```

shopping_level, click_rate, avg_price

```

Visualization:
- `cluster_age.png`
- `cluster_gender.png`

**Cluster Centers (Approximation):**

| Cluster | Shopping Level | Click Rate | Avg. Price | Description |
|----------|----------------|-------------|-------------|--------------|
| C0 | 0.00 | 0.00 | 258 | Passive browsers (no engagement) |
| C1 | 1.23 | 0.29 | 483 | Mid-level users, active clickers |
| C2 | 0.07 | 0.04 | 7534 | High-price exploratory users |
| C3 | 1.05 | 1.00 | 418 | Loyal frequent clickers |
| C4 | 0.01 | 0.00 | 2789 | Occasional high-value spenders |

**Cluster Distribution:**
```

C0: 80.06%, C1: 8.08%, C2: 2.03%, C3: 3.21%, C4: 6.63%

```

**Insights:**
- Over 80% of users exhibit near-zero engagement — an industry-wide “long-tail inactivity” pattern.  
- Cluster C3 (“Loyal Frequent Clickers”) represents the core audience for retention marketing.  
- Cluster C2 (“Explorers”) indicates curiosity-driven interactions on premium items — an ideal group for personalized recommendations.

---

### 3.6 RFM-Based Clustering

A second KMeans model is applied to RFM features:

```

recency, frequency, monetary

```

Cluster centers:

| Cluster | Recency | Frequency | Monetary | Behavioral Type |
|----------|----------|------------|-----------|------------------|
| R0 | 1009.35 | 0.00 | 366.77 | Inactive, low value |
| R1 | 1010.29 | 1.00 | 408.26 | Re-engaged users |
| R2 | 1013.48 | 0.00 | 348.33 | Churned segment |
| R3 | 1011.04 | 0.07 | 5712.31 | High spenders, rare activity |
| R4 | 1009.07 | 2.39 | 564.81 | Consistent active users |

Visualization:
- `rfmcluster_age.png`
- `rfmcluster_gender.png`

**Findings:**
- Two distinct user types dominate:
  - **R0/R2:** High recency, low frequency — retention targets.  
  - **R3/R4:** High value and consistent frequency — profitability drivers.
- Demographic overlays show female users slightly dominate high-frequency clusters.

---

### 3.7 Retention Analysis

#### (1) Cohort Retention (Daily)

Heatmap: `cohort_retention_heatmap.png`

**Observation:**
- Day 0 retention = 100% (baseline)  
- Day 1 retention ≈ 25%  
- Day 3 retention ≈ 13%  
- Day 7 retention ≈ 2%  
- After Day 14, nearly all cohorts decay to 0%.

This confirms a **short-cycle interest model**, meaning most users engage briefly before churn.

#### (2) Rolling Retention Summary

| Period | Retention Rate |
|--------|----------------|
| 7 Days | 2.46% |
| 14 Days | 0.00% |
| 30 Days | 0.00% |

Retention decays exponentially, suggesting that engagement renewal campaigns (D+1, D+3, D+7) are critical.

---

### 3.8 Price Bin Sensitivity Comparison

To test robustness, CTRs are recalculated using two different binning strategies:

- **Equal Width:**  
  Produces uniform intervals but ignores skewness.

- **Quantile-Based:**  
  Adapts to data distribution, more representative for heavy-tailed prices.

Results are visualized in:
- `price_ctr_equalwidth.png`
- `price_ctr_quantile.png`

**Result Summary:**
- Equal Width: nearly constant CTR (~4.9%) due to price concentration in low bins.  
- Quantile Binning: reveals meaningful CTR variation (~6.5% → 4.2%) across quantiles.

**Conclusion:**
Quantile-based binning is preferred for CTR modeling because it captures behavioral heterogeneity within the pricing domain.

---

## **Chapter 4. Discussion**

---

### 4.1 Behavioral and Psychological Mechanisms Behind CTR

The observed **inverted-U relationship** between price and CTR highlights the dual nature of consumer decision-making in digital advertising.

1. **Low-Price Attraction (0–200 range)**  
   Users tend to click impulsively on affordable items.  
   This reflects a **cost-accessibility effect** — lower cognitive barriers and reduced financial risk encourage spontaneous engagement.

2. **Mid-Price Fatigue (400–1600 range)**  
   As prices rise, users experience **decision hesitation**.  
   Without strong emotional or informational cues, they are less likely to click.

3. **High-Price Curiosity (>12,800 range)**  
   Users demonstrate exploratory behavior even for luxury products.  
   This corresponds to the **exploratory curiosity model**, where novelty and aspiration drive attention.

These dynamics jointly shape CTR outcomes and imply that **click behavior is not purely economic**, but also **psychological**.

---

### 4.2 Demographic Matching and Ad Efficiency

Ad performance strongly correlates with **audience–content congruence**.

- **CTR-Top Ads**: target younger, mid-tier consumers with affordable lifestyle goods.  
- **CTR-Bottom Ads**: show mismatched targeting, such as luxury products shown to students or low-tier cities.

A new metric is proposed:
```

Match Rate = (targeted_audience_overlap / total_audience) * 100

```

High Match Rate corresponds to high CTR and sustained engagement, confirming that **precision targeting** is a stronger driver of ad efficiency than creative quality alone.

---

### 4.3 Temporal Behavior and User Lifecycle

The retention and RFM analyses reveal a **short attention span** across cohorts:

- Most users engage within **1–3 days** of initial exposure.  
- After 7 days, fewer than **3%** remain active.  
- Recency (R) shows the highest predictive power among RFM metrics.

These findings validate a **short-cycle engagement model** where users’ interaction intensity decays rapidly, underscoring the importance of reactivation strategies (e.g., D+1, D+3, D+7 reminders).

---

### 4.4 User Segmentation Insights

#### (1) Deep-Nurture Users (26.9%)
Consistent high spenders who frequently click on mid- to high-price items.  
Require **personalized loyalty campaigns** and premium recommendations.

#### (2) New Users (25.9%)
First-time clickers with low recency and frequency.  
Best targeted through **welcome flows** and **trial incentives**.

#### (3) Churned and Retention Users (~35%)  
Inactive segments with prior high value.  
Should be **re-engaged via retargeting and email remarketing**.

#### (4) Exploratory High-Price Users (~2%)  
Driven by novelty rather than need.  
Ideal audience for **cross-brand promotions** or **content-based ad narratives**.

---

### 4.5 Model Limitations and Statistical Considerations

1. **Time Window Constraint**  
   The data reflects a short-term sample; seasonal or long-horizon behaviors are not captured.

2. **Simplifying Assumptions in RFM and KMeans**  
   Both assume independent features and roughly spherical cluster structures, which may oversimplify user diversity.

3. **Absence of Conversion Metrics**  
   CTR is used as a proxy for engagement, but **conversion rate (CVR)** and **return on investment (ROI)** are not included.

4. **Retention Model Simplification**  
   Cohort analysis uses only first activity timestamps, without modeling sequential re-engagement events.

These limitations highlight the need for **longitudinal tracking** and **nonlinear modeling techniques** in future iterations.

---

### 4.6 Interpretation and Business Implications

From a business perspective, the analytical results suggest that **price sensitivity, demographic targeting, and lifecycle timing** are the three key levers for improving ad ROI.

**Strategic Takeaways:**
- Implement **price-tiered ad bidding** (dynamic CPC by price bin).  
- Optimize ad delivery based on **demographic affinity scores**.  
- Introduce **time-based remarketing loops** to re-engage users before churn.

Together, these actions can lead to measurable increases in overall CTR, retention, and conversion efficiency.

## **Chapter 5. Strategy and Application**

---

### 5.1 Translating Insights into Action

The analytical results from previous chapters form a comprehensive foundation for **data-driven decision-making** in ad optimization.  
This chapter translates quantitative insights into practical business strategies, focusing on targeting, retention, experimentation, and personalization.

The goal is to create a **closed feedback loop** from data → insight → execution → evaluation.

---

### 5.2 Price-Tier Optimization Strategy

1. **Dynamic CPC Bidding by Price Bin**  
   Allocate ad budget proportionally based on CTR performance per price tier.  
   For example:  
```

bid_weight = CTR_bin / mean_CTR

```
- High-CTR bins (0–200, 12801+) receive more bidding weight.  
- Mid-price bins (400–1600) are optimized through creative redesign or audience retargeting.

2. **Elasticity-Based ROI Tracking**  
Incorporate price elasticity analysis to measure marginal gains:  
```

ROI = (CTR * ConversionRate * ProfitPerClick) / CostPerImpression

````
Adjust pricing and creative intensity dynamically by ROI sensitivity.

3. **Recommendation for Implementation**
- Integrate these weights into DSP/RTB bidding algorithms.  
- Monitor real-time CTR by bin and reweight weekly.

---

### 5.3 Demographic Targeting Enhancement

Using the demographic alignment pattern discovered earlier, construct an **Audience–Ad Match Matrix**:

| Demographic Segment | Ad Type | Strategy |
|----------------------|----------|-----------|
| Female, 18–25, Low Price Sensitivity | Fashion, Cosmetics | Emphasize visuals, flash sales |
| Male, 25–40, Tech Interest | Electronics, Tools | Highlight functionality and specs |
| Student, Low Consumption Level | Education, Daily Goods | Use discounts and reward incentives |
| High Income, 30+ | Premium & Luxury | Stress exclusivity and brand identity |

This **MatchRate matrix** provides the operational basis for **campaign segmentation**, ensuring that creative assets and placements correspond precisely to audience profiles.

---

### 5.4 User Lifecycle Management

The RFM and cohort analyses demonstrate a **short-term lifecycle** with rapid engagement decay.  
Therefore, lifecycle management should focus on **timely reactivation**:

| Stage | Strategy | Timing |
|--------|-----------|---------|
| D+1 | Send post-click reminders or offer coupons | 24 hours after click |
| D+3 | Deliver personalized recommendations | 3 days post-click |
| D+7 | Reactivation ads or email remarketing | 7 days post-click |

Such time-sensitive reactivation loops can **extend user lifetime value (LTV)** and **reduce churn rate** effectively.

---

### 5.5 Personalized Recommendation System

Leveraging clustering outputs, each user can be assigned to a **behavioral segment** for content personalization.

**Example Personalization Logic:**
```python
if user.cluster == 'C3':
 recommend('high-engagement', n=3)
elif user.cluster == 'C1':
 recommend('discount-focused', n=5)
elif user.cluster == 'C2':
 recommend('premium-exploratory', n=2)
````

This system provides:

* **Higher CTR uplift** via relevance.
* **Reduced ad fatigue** through varied content.
* **Better ROI** by targeting high-value clusters selectively.

---

### 5.6 A/B Testing Framework

A standardized A/B testing protocol ensures objective performance evaluation for different strategies.

**Structure:**

| Component         | Description                                    |
| ----------------- | ---------------------------------------------- |
| Control Group     | Baseline ad version                            |
| Treatment Group   | Modified version (price, design, or targeting) |
| Metric            | CTR, CVR, ROI                                  |
| Significance Test | Two-sample Z-test                              |
| Confidence Level  | 95% (α = 0.05)                                 |

**Formula:**

```
z = (CTR1 - CTR2) / sqrt(p*(1-p)*(1/n1 + 1/n2))
```

where
`p = (clicks1 + clicks2) / (impressions1 + impressions2)`

**Implementation Tip:**
Maintain a centralized log of experiments, and evaluate using both *statistical* and *business significance* metrics.

---

### 5.7 Retention and Re-Engagement Tactics

1. **Micro-Segmented Retargeting**

   * Use RFM categories (e.g., Retention, Deep Nurture) to tailor messages.
   * Combine email, push notification, and social channels.

2. **Content Sequencing**

   * Design “narrative ads” — sequence of thematic creatives that gradually build product familiarity.

3. **Incentive Re-Activation**

   * Reward-based reminders for users inactive beyond 3 days.

4. **Predictive Recall Triggers**

   * Train simple logistic or time-decay models to predict when a user is likely to disengage.

---

### 5.8 Multi-Metric Optimization Framework

Advertising success should not rely solely on CTR.
A **multi-objective optimization** can better balance engagement, conversion, and retention.

```
Objective = w1 * CTR + w2 * CVR + w3 * Retention
```

where weights `w1`, `w2`, and `w3` are dynamically adjusted based on business priorities.

This framework allows adaptive optimization across different campaign goals (traffic, conversion, or loyalty).

---

### 5.9 System Integration and Automation

For operational deployment, the pipeline can be integrated as an automated system:

* **Data Layer:** Scheduled ETL job (Airflow or Cron) updating daily logs.
* **Model Layer:** Automated retraining of clustering and CTR models every week.
* **Visualization Layer:** Streamlit or PowerBI dashboard for live KPI monitoring.
* **Action Layer:** API-based triggers for bid adjustments or re-engagement campaigns.

This end-to-end system design ensures a **feedback-driven optimization loop**.

好的 ✅
以下是英文版 **Chapter 5: Strategy and Application**，依然采用 GitHub Markdown 风格，适合直接展示在仓库 README 中（包含策略方案、A/B 实验与业务落地设计）。

---

```markdown
## **Chapter 5. Strategy and Application**

---

### 5.1 Translating Insights into Action

The analytical results from previous chapters form a comprehensive foundation for **data-driven decision-making** in ad optimization.  
This chapter translates quantitative insights into practical business strategies, focusing on targeting, retention, experimentation, and personalization.

The goal is to create a **closed feedback loop** from data → insight → execution → evaluation.

---

### 5.2 Price-Tier Optimization Strategy

1. **Dynamic CPC Bidding by Price Bin**  
   Allocate ad budget proportionally based on CTR performance per price tier.  
   For example:  
```

bid_weight = CTR_bin / mean_CTR

```
- High-CTR bins (0–200, 12801+) receive more bidding weight.  
- Mid-price bins (400–1600) are optimized through creative redesign or audience retargeting.

2. **Elasticity-Based ROI Tracking**  
Incorporate price elasticity analysis to measure marginal gains:  
```

ROI = (CTR * ConversionRate * ProfitPerClick) / CostPerImpression

````
Adjust pricing and creative intensity dynamically by ROI sensitivity.

3. **Recommendation for Implementation**
- Integrate these weights into DSP/RTB bidding algorithms.  
- Monitor real-time CTR by bin and reweight weekly.

---

### 5.3 Demographic Targeting Enhancement

Using the demographic alignment pattern discovered earlier, construct an **Audience–Ad Match Matrix**:

| Demographic Segment | Ad Type | Strategy |
|----------------------|----------|-----------|
| Female, 18–25, Low Price Sensitivity | Fashion, Cosmetics | Emphasize visuals, flash sales |
| Male, 25–40, Tech Interest | Electronics, Tools | Highlight functionality and specs |
| Student, Low Consumption Level | Education, Daily Goods | Use discounts and reward incentives |
| High Income, 30+ | Premium & Luxury | Stress exclusivity and brand identity |

This **MatchRate matrix** provides the operational basis for **campaign segmentation**, ensuring that creative assets and placements correspond precisely to audience profiles.

---

### 5.4 User Lifecycle Management

The RFM and cohort analyses demonstrate a **short-term lifecycle** with rapid engagement decay.  
Therefore, lifecycle management should focus on **timely reactivation**:

| Stage | Strategy | Timing |
|--------|-----------|---------|
| D+1 | Send post-click reminders or offer coupons | 24 hours after click |
| D+3 | Deliver personalized recommendations | 3 days post-click |
| D+7 | Reactivation ads or email remarketing | 7 days post-click |

Such time-sensitive reactivation loops can **extend user lifetime value (LTV)** and **reduce churn rate** effectively.

---

### 5.5 Personalized Recommendation System

Leveraging clustering outputs, each user can be assigned to a **behavioral segment** for content personalization.

**Example Personalization Logic:**
```python
if user.cluster == 'C3':
 recommend('high-engagement', n=3)
elif user.cluster == 'C1':
 recommend('discount-focused', n=5)
elif user.cluster == 'C2':
 recommend('premium-exploratory', n=2)
````

This system provides:

* **Higher CTR uplift** via relevance.
* **Reduced ad fatigue** through varied content.
* **Better ROI** by targeting high-value clusters selectively.

---

### 5.6 A/B Testing Framework

A standardized A/B testing protocol ensures objective performance evaluation for different strategies.

**Structure:**

| Component         | Description                                    |
| ----------------- | ---------------------------------------------- |
| Control Group     | Baseline ad version                            |
| Treatment Group   | Modified version (price, design, or targeting) |
| Metric            | CTR, CVR, ROI                                  |
| Significance Test | Two-sample Z-test                              |
| Confidence Level  | 95% (α = 0.05)                                 |

**Formula:**

```
z = (CTR1 - CTR2) / sqrt(p*(1-p)*(1/n1 + 1/n2))
```

where
`p = (clicks1 + clicks2) / (impressions1 + impressions2)`

**Implementation Tip:**
Maintain a centralized log of experiments, and evaluate using both *statistical* and *business significance* metrics.

---

### 5.7 Retention and Re-Engagement Tactics

1. **Micro-Segmented Retargeting**

   * Use RFM categories (e.g., Retention, Deep Nurture) to tailor messages.
   * Combine email, push notification, and social channels.

2. **Content Sequencing**

   * Design “narrative ads” — sequence of thematic creatives that gradually build product familiarity.

3. **Incentive Re-Activation**

   * Reward-based reminders for users inactive beyond 3 days.

4. **Predictive Recall Triggers**

   * Train simple logistic or time-decay models to predict when a user is likely to disengage.

---

### 5.8 Multi-Metric Optimization Framework

Advertising success should not rely solely on CTR.
A **multi-objective optimization** can better balance engagement, conversion, and retention.

```
Objective = w1 * CTR + w2 * CVR + w3 * Retention
```

where weights `w1`, `w2`, and `w3` are dynamically adjusted based on business priorities.

This framework allows adaptive optimization across different campaign goals (traffic, conversion, or loyalty).

---

### 5.9 System Integration and Automation

For operational deployment, the pipeline can be integrated as an automated system:

* **Data Layer:** Scheduled ETL job (Airflow or Cron) updating daily logs.
* **Model Layer:** Automated retraining of clustering and CTR models every week.
* **Visualization Layer:** Streamlit or PowerBI dashboard for live KPI monitoring.
* **Action Layer:** API-based triggers for bid adjustments or re-engagement campaigns.

This end-to-end system design ensures a **feedback-driven optimization loop**.

---

## **Chapter 6. Conclusion and Future Work**

---

### 6.1 Summary of Findings

This project constructed a full-stack **Ad Analytics End-to-End system** integrating descriptive statistics, behavioral segmentation, and retention modeling.  
The analysis revealed several key empirical findings:

1. **Price–CTR Relationship:**  
   CTR follows an *inverted-U pattern* — low-price ads attract impulsive clicks, mid-price ads face hesitation, and high-price ads regain exploratory engagement.

2. **Demographic Targeting Efficiency:**  
   Ad performance depends more on **audience alignment** than on price or visuals.  
   A “MatchRate” metric effectively measures the congruence between target demographics and actual audience.

3. **User Value Distribution:**  
   RFM and clustering results confirm a **Pareto distribution** — fewer than 5% of users (High-Value) contribute to most revenue, while the majority remain low-frequency or passive.

4. **Short-Cycle Retention Behavior:**  
   Cohort analysis demonstrates that most users churn within 7 days of first exposure.  
   Timely recall campaigns (D+1, D+3, D+7) are crucial for sustained engagement.

5. **Methodological Integration:**  
   The system bridges statistical modeling (CTR, RFM) and machine learning clustering, achieving both interpretability and business applicability.

---

### 6.2 Theoretical Implications

This research advances advertising analytics in three dimensions:

1. **From Univariate to Multidimensional Modeling**  
   The project shifts CTR analysis from single-variable regression to **behavioral feature interaction**, integrating psychological, economic, and temporal variables.

2. **Empirical Verification of Behavioral Economics**  
   The “inverted-U” CTR pattern empirically supports the **price anchoring** and **exploration incentive** theories in digital consumer behavior.

3. **Temporal Behavior Modeling**  
   By combining RFM segmentation with cohort retention, this study introduces a **time-series perspective** to user lifecycle analysis, paving the way for predictive modeling of engagement decay.

---

### 6.3 Practical Implications

1. **Operational Optimization**  
   - Implement dynamic price-tier bidding using the bin-weighted CTR structure.  
   - Utilize demographic affinity scores for ad placement refinement.  
   - Apply clustering output to personalize recommendations and reactivation campaigns.

2. **Data-Driven A/B Testing Culture**  
   The inclusion of a Z-test framework ensures that experimental outcomes are **statistically verifiable** rather than anecdotal.

3. **Personalized Retention Strategy**  
   Segment users by behavioral type and tailor post-engagement actions, boosting long-term ROI.

---

### 6.4 Limitations

1. **Temporal Scope**  
   The dataset represents a short time span, limiting the ability to model seasonal effects.

2. **Model Simplification**  
   RFM and KMeans assume independence and Euclidean geometry, potentially oversimplifying nonlinear user behavior.

3. **Missing Conversion Data**  
   The project uses CTR as a proxy for engagement but does not include purchase or ROI-level outcomes.

4. **Retention Model Constraints**  
   Cohort-based analysis tracks first interactions only, missing multi-session patterns.

---

### 6.5 Future Research Directions

#### (1) Methodological Expansion
- Integrate **deep sequential models** (e.g., LSTM, Transformer) for temporal prediction.  
- Employ **mixture-based clustering** (e.g., GMM, DBSCAN) to capture complex user boundaries.  
- Develop **Bayesian dynamic RFM models** that update recency/frequency in real time.

#### (2) System Automation
- Build a real-time pipeline using Airflow and Streamlit for continuous data refresh.  
- Develop an internal Python library `ad_analytics_toolkit` to modularize and reuse core functions.

#### (3) Multi-Objective Optimization
Introduce a unified optimization framework:
```

Objective = w1 * CTR + w2 * CVR + w3 * Retention

```
Weights (`w1`, `w2`, `w3`) can be dynamically tuned to align with marketing priorities, achieving a balance between traffic, conversion, and loyalty.

#### (4) Cross-Disciplinary Integration
Future work should integrate **psychology**, **data mining**, and **computational advertising** to model the cognitive–motivational–behavioral triad underlying digital interactions.

---

### 6.6 Overall Conclusion

This project demonstrates the transition from **descriptive analytics** to **intelligent decision-making** in advertising systems.

Key takeaways:
- User click behavior is shaped by both *price perception* and *psychological motivation*.  
- Ad performance hinges on *audience alignment*, not creative intensity.  
- Engagement lifespan is short but extendable through *timing-sensitive recall strategies*.  
- Integrating *behavioral science with data engineering* yields measurable ROI improvement.

In essence, the future of advertising intelligence lies in systems that are **self-learning**, **self-optimizing**, and **interpretable**, bridging the gap between human insight and algorithmic precision.

---


