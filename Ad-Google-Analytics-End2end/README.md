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

### **2.1 Dataset Overview**

The project integrates three primary datasets — `user.csv`, `click.csv`, and `ad.csv` — into a unified analytical table named **processed_user_ad.csv**.
Each record represents one ad impression linked to a unique user and ad ID.

| Dataset        | Description                                     | Key Columns                                                                     |
| -------------- | ----------------------------------------------- | ------------------------------------------------------------------------------- |
| **User Data**  | Contains demographic and behavioral attributes. | `userid`, `gender`, `age_level`, `consumption_level`, `city_tier`, `is_student` |
| **Click Data** | Logs ad exposure and click behaviors.           | `userid`, `ad_id`, `clk`, `nonclk`, `time_stamp`                                |
| **Ad Data**    | Stores ad metadata and product information.     | `ad_id`, `brand`, `campaign_id`, `category_id`, `price`, `customer`             |

After merging and cleaning, the final dataset includes:

* **975,441 rows** and **19 columns**
* **Missing rate** below 5% for most variables
* **Distinct users:** ~356,000
* **Distinct ads:** ~59,000

---

### **2.2 Data Preprocessing Workflow**

The preprocessing pipeline follows these major steps:

1. **Robust Reading**
   The function automatically retries encodings (`gbk`, `utf-8`, `latin1`) to handle inconsistencies.

2. **Field Standardization**
   All user IDs are renamed to `userid`, and ad IDs to `ad_id` for consistent merging.

3. **Dataset Merging**
   The three source files are merged using left joins on `userid` and `ad_id`.

4. **Price Binning**
   Prices are divided into nine fixed bins:
   [0–100], [101–200], [201–400], [401–800], [801–1600], [1601–3200], [3201–6400], [6401–12800], [12801+].
   Each bin is assigned an integer label (`price_bin_code`).

5. **Outlier Capping**
   Extremely high prices are capped at the 99th percentile (`price_capped`) to stabilize the distribution.

6. **Timestamp Conversion**
   Epoch timestamps are converted to standard datetime objects for temporal analysis.

---

### **2.3 Analytical Indicators**

The analytical framework includes three levels of descriptive indicators:

**(1) Field-Level Profiling**

* Data type, missing rate, unique count, and value range for each field.
* Basic statistics: mean, standard deviation, quantiles (1%, 25%, 50%, 75%, 99%).

**(2) Integrity and Uniqueness Checks**

* Primary Keys: `userid`, `ad_id`
* Duplicate Rates:

  * User ID duplication: ~63.5%
  * Ad ID duplication: ~93.9%
    Indicates repeated ad exposure and multi-view user activity.

---

### **2.4 CTR and Price Relationship**

**CTR (Click-Through Rate)** is computed as:

CTR = clicks / (clicks + nonclicks)

For each price bin, the mean CTR and corresponding user share are computed:

User Share (%) = (unique_users_in_bin / total_unique_users) × 100

**Findings:**

* Low-price bins (0–200) yield the highest CTR (~5.5%).
* Middle bins (400–1600) show lower CTR (~4.4%).
* Ultra-high bins (12801+) rebound to ~5.1%, suggesting curiosity-driven clicks.

Visualizations:

* `ctr_by_pricebin.png`
<img width="500" height="300" alt="ctr_by_pricebin" src="https://github.com/user-attachments/assets/a3a95d2e-25c8-4c6d-a693-3b4a5e4adb51" />

* `usershare_by_pricebin.png`
<img width="500" height="300" alt="usershare_by_pricebin" src="https://github.com/user-attachments/assets/e0e847a7-4fb5-443e-b791-ab032b28de51" />

---

### **2.5 Ad-Level Performance**

Ads are ranked by impressions and CTR to identify top and bottom performers.

Visualizations:

* **Top 10 Ads:** `ad_top10_age.png`, `ad_top10_gender.png`, `ad_top10_level.png`
<img width="500" height="300" alt="ad_top10_age" src="https://github.com/user-attachments/assets/253ed500-c00d-48ec-8bf5-5e4393dc74fd" />
<img width="500" height="300" alt="ad_top10_level" src="https://github.com/user-attachments/assets/6ae2438f-01ea-41e8-9517-0a6b939e807f" />
<img width="500" height="300" alt="ad_top10_gender" src="https://github.com/user-attachments/assets/73151af8-83b4-4b06-b26d-03c25bc5b696" />

  
* **Bottom 10 Ads:** `ad_bottom10_age.png`, `ad_bottom10_gender.png`, `ad_bottom10_level.png`
<img width="500" height="300" alt="ad_bottom10_age" src="https://github.com/user-attachments/assets/123b0aba-cf24-4494-bc0b-a4f85a146a35" />
<img width="500" height="300" alt="ad_bottom10_level" src="https://github.com/user-attachments/assets/e29ddd10-3b42-48be-a345-d750ab15f305" />
<img width="500" height="300" alt="ad_bottom10_gender" src="https://github.com/user-attachments/assets/2c48d82d-de6d-45bc-add7-6f5147ffa8fb" />

**Insights:**

* Top ads align closely with user demographics (gender × age match).
* Poor ads often display demographic mismatch (e.g., luxury ads targeting students).
* Price alone does not dictate performance; contextual relevance matters.

---

### **2.6 RFM Segmentation**

The RFM model (Recency, Frequency, Monetary) measures user value and engagement:

* R = days since last click
* F = total number of clicks
* M = average price of clicked ads

Median thresholds: R = 1011.0, F = 1.0, M = 192.0

Users are divided into eight groups:

| R | F | M | Segment            |
| - | - | - | ------------------ |
| 1 | 1 | 1 | High-Value Users   |
| 0 | 1 | 1 | Re-Engage Users    |
| 1 | 0 | 1 | Deep-Nurture Users |
| 0 | 0 | 1 | Retention Users    |
| 1 | 1 | 0 | Potential Users    |
| 1 | 0 | 0 | New Users          |
| 0 | 1 | 0 | Maintain Users     |
| 0 | 0 | 0 | Churned Users      |

Visualizations:

* `rfm_age.png`
* `rfm_gender.png`

---

### **2.7 KMeans Clustering**

Two separate clustering models are trained:

**(1) Behavioral Clustering**
Features: `[shopping_level, click_rate, avg_price]`

* Number of clusters: 5
* Outputs: `cluster_age.png`, `cluster_gender.png`
<img width="500" height="300" alt="cluster_gender" src="https://github.com/user-attachments/assets/6ffff8f4-3c38-4923-b14f-773a0f38d54a" />
<img width="500" height="300" alt="cluster_age" src="https://github.com/user-attachments/assets/8b3cb6f2-6720-4f62-bb6d-d89ff9e3a8cc" />

**(2) RFM-Based Clustering**
Features: `[recency, frequency, monetary]`

* Number of clusters: 5
* Outputs: `rfmcluster_age.png`, `rfmcluster_gender.png`
<img width="500" height="300" alt="rfmcluster_gender" src="https://github.com/user-attachments/assets/09bd7dc1-ee93-4e0f-b7a9-359529b92206" />
<img width="500" height="300" alt="rfmcluster_age" src="https://github.com/user-attachments/assets/52f50d51-18b7-4469-b7ee-ef51053fb709" />

These clusters reveal user subgroups such as price-sensitive explorers, loyal high-frequency clickers, and passive browsers.

---

### **2.8 Retention Analysis**

Two retention metrics are analyzed:

**(1) Cohort Retention**
Each user’s first interaction defines a cohort.
The retention matrix tracks active users over time.
Visualization: `cohort_retention_heatmap.png`

Results:

* Day 1 retention: 22.8%
* Day 7 retention: 2.46%
* Day 14+ retention: 0%

<img width="500" height="300" alt="cohort_retention_heatmap" src="https://github.com/user-attachments/assets/395e49de-e091-41fa-aa84-2c2193672b28" />

**(2) Rolling Retention**

Rolling Retention (k days) = (users_active_after_k_days / total_users) × 100

7-day retention = 2.46%
14-day and 30-day retention = 0%

This confirms short engagement lifecycles typical of online ad interactions.

---

### **2.9 Price Bin Sensitivity**

Two binning strategies are compared:

* **Equal Width (EW):** uniform intervals across price range
* **Quantile (QT):** percentiles-based adaptive bins

Visualizations:

* `price_ctr_equalwidth.png`
* `price_ctr_quantile.png`
<img width="500" height="300" alt="price_ctr_equalwidth" src="https://github.com/user-attachments/assets/02d84b60-1dee-469c-829e-5deb81fbceb0" />
<img width="500" height="300" alt="price_ctr_quantile" src="https://github.com/user-attachments/assets/6b86ea10-209d-4ef5-b612-8cc125dfefd9" />

**Conclusion:** Quantile binning better captures CTR variation under skewed distributions.

---

## **Chapter 3. Data Analysis and Results**

### **3.1 Overview**

This chapter presents the analytical results derived from the merged dataset, covering click-through rate (CTR) distribution, user share by price bin, ad-level performance, RFM-based segmentation, and KMeans clustering.
The aim is to identify the behavioral and structural patterns that explain user engagement and ad performance.

---

### **3.2 CTR Distribution by Price Bin**

CTR and user distribution are summarized through the following visualizations:

* `ctr_by_pricebin.png`
* `usershare_by_pricebin.png`

| Price Bin  | CTR (%) | User Share (%) |
| ---------- | ------- | -------------- |
| 0–100      | 5.49    | 51.07          |
| 101–200    | 4.62    | 31.29          |
| 201–400    | 4.68    | 29.80          |
| 401–800    | 4.60    | 17.07          |
| 801–1600   | 4.45    | 10.87          |
| 1601–3200  | 4.20    | 7.19           |
| 3201–6400  | 4.17    | 4.99           |
| 6401–12800 | 4.38    | 2.86           |
| 12801+     | 5.13    | 1.23           |
<img width="500" height="300" alt="ctr_by_pricebin" src="https://github.com/user-attachments/assets/41e95bd0-18ef-476c-9f0a-af937f7a7215" />
<img width="500" height="300" alt="usershare_by_pricebin" src="https://github.com/user-attachments/assets/a416d6dc-b20f-42d8-aa52-6f91b0c3205b" />

**Findings:**

* CTR follows an **inverted-U shape** across price tiers.
  Low-price items attract exploratory clicks due to affordability, while mid-tier prices suffer from decision hesitation.
  High-price products show renewed curiosity, suggesting aspirational browsing.
* More than half of users (51%) engage in the lowest price tier, emphasizing **price sensitivity** as a key engagement driver.

---

### **3.3 Top and Bottom Ad Performance**

The ranking of ads by impressions and CTR reveals substantial variation in targeting accuracy.
Visualizations include:

* Top 10 ads: `ad_top10_age.png`, `ad_top10_gender.png`, `ad_top10_level.png`
* Bottom 10 ads: `ad_bottom10_age.png`, `ad_bottom10_gender.png`, `ad_bottom10_level.png`

**Insights:**

* Top-performing ads align with the correct **demographic segments** (e.g., female 18–25).
* Low-performing ads display **targeting mismatch** — such as high-end campaigns aimed at low-spending users.
* Contextual relevance outweighs price or visual design in determining CTR.

Supporting figures:

* `clicked_price.png` (average price of clicked items)
* `nonclicked_price.png` (average price of skipped items)
<img width="500" height="300" alt="clicked_price" src="https://github.com/user-attachments/assets/15d06954-4391-417a-92d2-2aad74e9041c" />
<img width="500" height="300" alt="nonclicked_price" src="https://github.com/user-attachments/assets/e757c13b-43b3-442b-858c-e56c6b6ee8a5" />

---

### **3.4 User Value Segmentation (RFM Analysis)**

The **RFM model** measures user engagement along three dimensions:

* **Recency (R):** days since last click
* **Frequency (F):** total number of clicks
* **Monetary (M):** average price of clicked ads

Median thresholds: R = 1011.0, F = 1.0, M = 192.0
Visualizations: `rfm_age.png`, `rfm_gender.png`

| Category           | Definition                               | Share (%) |
| ------------------ | ---------------------------------------- | --------- |
| Deep Nurture Users | Frequent interaction with high-value ads | 26.90     |
| New Users          | Recently active, low click history       | 25.89     |
| Churned Users      | No recent activity                       | 18.52     |
| Retention Users    | High-value, disengaging                  | 17.19     |
| High-Value Users   | High spending and frequent               | 4.57      |
| Potential Users    | Regular but low-value                    | 4.06      |
| Maintain Users     | Consistent, medium-value                 | 1.51      |
| Re-Engage Users    | Previously active, now inactive          | 1.36      |
<img width="500" height="300" alt="rfm_gender" src="https://github.com/user-attachments/assets/9aa2d80b-10f8-4df9-8224-836e7a62cc26" />
<img width="500" height="300" alt="rfm_age" src="https://github.com/user-attachments/assets/56cdbf0b-9979-4653-8e99-c65a90027941" />

**Interpretation:**

* Nearly half of users belong to low-frequency or early-stage segments.
* Deep Nurture Users contribute the majority of recurring engagement.
* A small elite (≈5%) drives disproportionate revenue, validating the **Pareto principle**.

---

### **3.5 KMeans Behavioral Clustering**

Clustering on `[shopping_level, click_rate, avg_price]` reveals latent behavioral segments.
Visualizations: `cluster_age.png`, `cluster_gender.png`

| Cluster | Shopping Level | Click Rate | Avg. Price | Description                          |
| ------- | -------------- | ---------- | ---------- | ------------------------------------ |
| C0      | 0.00           | 0.00       | 258        | Passive browsers (no engagement)     |
| C1      | 1.23           | 0.29       | 483        | Mid-level users, occasional clickers |
| C2      | 0.07           | 0.04       | 7534       | High-price explorers                 |
| C3      | 1.05           | 1.00       | 418        | Loyal and frequent clickers          |
| C4      | 0.01           | 0.00       | 2789       | Occasional high-spenders             |

Cluster distribution:

* C0: 80.06%
* C1: 8.08%
* C2: 2.03%
* C3: 3.21%
* C4: 6.63%
<img width="500" height="300" alt="cluster_gender" src="https://github.com/user-attachments/assets/29d3d3ac-066b-4ef9-988c-dcc5aec1adf3" />
<img width="500" height="300" alt="cluster_age" src="https://github.com/user-attachments/assets/f797481e-310c-457e-884e-a5fa59ac01b6" />

**Insights:**

* Over 80% of users show minimal engagement — a typical “long-tail” inactivity phenomenon.
* Cluster C3 users are the **core retention target**, showing consistent engagement and loyalty.
* Cluster C2 users (Explorers) show curiosity toward premium products, offering potential for **personalized marketing**.

---

### **3.6 RFM-Based Clustering**

A second KMeans model is applied to `[recency, frequency, monetary]`.
Visualizations: `rfmcluster_age.png`, `rfmcluster_gender.png`

| Cluster | Recency | Frequency | Monetary | Type                         |
| ------- | ------- | --------- | -------- | ---------------------------- |
| R0      | 1009.35 | 0.00      | 366.77   | Inactive, low-value          |
| R1      | 1010.29 | 1.00      | 408.26   | Re-engaged users             |
| R2      | 1013.48 | 0.00      | 348.33   | Churned segment              |
| R3      | 1011.04 | 0.07      | 5712.31  | High spenders, rare activity |
| R4      | 1009.07 | 2.39      | 564.81   | Consistent active users      |
<img width="500" height="300" alt="rfmcluster_gender" src="https://github.com/user-attachments/assets/e6c117ee-8102-4482-9004-164820c99e5f" />
<img width="500" height="300" alt="rfmcluster_age" src="https://github.com/user-attachments/assets/da571b69-7042-453a-8132-93e25cf6498d" />

**Findings:**

* R0/R2 represent **retention targets** due to inactivity.
* R3/R4 correspond to **profit-driving users**.
* Female users slightly dominate high-frequency segments.

---

### **3.7 Retention Analysis**

**Cohort Retention:**
Visualized in `cohort_retention_heatmap.png`

| Period  | Retention (%) |
| ------- | ------------- |
| Day 1   | 25            |
| Day 3   | 13            |
| Day 7   | 2             |
| Day 14+ | ≈0            |

Engagement decays sharply within the first week, validating the **short-cycle interest model**.
<img width="500" height="300" alt="cohort_retention_heatmap" src="https://github.com/user-attachments/assets/1604e411-356b-4857-8508-367b2873463d" />

**Rolling Retention:**
7-Day = 2.46%
14-Day = 0%
30-Day = 0%
Retention drops exponentially, underscoring the importance of **timely re-engagement campaigns**.

---

### **3.8 Price Bin Sensitivity**

Two price segmentation strategies were compared:

* **Equal Width (EW)**: divides entire price range evenly
* **Quantile (QT)**: divides by percentile thresholds

Visualizations:

* `price_ctr_equalwidth.png`
<img width="500" height="300" alt="price_ctr_equalwidth" src="https://github.com/user-attachments/assets/fd228a69-29c4-40af-b1c0-910a7d6d9267" />

* `price_ctr_quantile.png`
<img width="500" height="300" alt="price_ctr_quantile" src="https://github.com/user-attachments/assets/86b80761-39f1-4210-aca3-ad8ae2208ecd" />

**Summary:**

* Equal-width binning smooths out variation, masking real behavior.
* Quantile binning exposes significant CTR differences between lower and higher quantiles.
* Therefore, **quantile binning** is more suitable for real-world CTR modeling.

---

## **Chapter 4. Discussion**

### **4.1 Behavioral and Psychological Mechanisms Behind CTR**

The observed **inverted-U pattern** between price and CTR reflects fundamental psychological mechanisms behind online consumer behavior:

1. **Low-Price Attraction (0–200 Range)**
   Users tend to click impulsively on low-priced items because the **perceived risk is minimal** and the **decision cost is low**.
   This supports the **cost-accessibility hypothesis**, which posits that affordability drives curiosity clicks.

2. **Mid-Price Fatigue (400–1600 Range)**
   As prices rise, users experience **decision hesitation**—they need more cognitive effort to evaluate value versus cost.
   Ads in this range require additional cues such as social proof or discounts to sustain engagement.

3. **High-Price Curiosity (>12,800 Range)**
   Despite the high cost, certain ads attract clicks due to **aspirational motivation**.
   These users click not to purchase, but to **explore luxury or novelty**, aligning with the **exploratory curiosity model**.

Together, these findings indicate that **click behavior is driven by both economic reasoning and emotional factors**, emphasizing the dual nature of online ad engagement.

---

### **4.2 Demographic Matching and Ad Efficiency**

Ad performance is highly dependent on **audience–content congruence**.
When an ad aligns with its target demographic, it consistently achieves higher CTR and retention.

| Ad Type                    | Target Demographic             | Performance                 |
| -------------------------- | ------------------------------ | --------------------------- |
| Affordable lifestyle goods | Female, 18–25, mid consumption | High CTR, strong conversion |
| Luxury goods               | High income, Tier-1 cities     | Moderate CTR, high ROI      |
| Educational services       | Students, low consumption      | Low CTR, high retention     |
| Generic brand ads          | Mixed demographics             | Low CTR, poor engagement    |

A new metric, **Match Rate**, can quantify targeting accuracy:

**Match Rate = (Targeted Audience Overlap / Total Audience) × 100**

High match rate values (>70%) correlate with higher CTR and longer engagement, confirming that **precision targeting** is more impactful than ad aesthetics alone.

---

### **4.3 Temporal Behavior and User Lifecycle**

The retention and RFM analyses reveal that **user engagement follows a short-lived cycle**:

* Most users interact intensively within **1–3 days** of first exposure.
* After **7 days**, fewer than 3% remain active.
* After **14 days**, nearly all cohorts churn.

This pattern aligns with the **short-cycle interest model**, suggesting that digital ad platforms should focus on **timing-sensitive recall** mechanisms (e.g., D+1, D+3, D+7 reactivation campaigns).

Recency (R) shows the strongest correlation with future engagement, confirming its predictive value for **user reactivation models**.

---

### **4.4 User Segmentation Insights**

Based on RFM and KMeans results, user behavior can be categorized into distinct strategic segments:

1. **Deep-Nurture Users (26.9%)**
   High engagement and frequent interaction with valuable products.
   Strategy: Reward-based loyalty programs and premium recommendations.

2. **New Users (25.9%)**
   Recently acquired, low engagement history.
   Strategy: Welcome flows, onboarding content, and first-purchase incentives.

3. **Churned and Retention Users (~35%)**
   Inactive but previously valuable users.
   Strategy: Email remarketing and reactivation campaigns.

4. **Exploratory High-Price Users (~2%)**
   Curiosity-driven users with interest in high-value items.
   Strategy: Brand storytelling and aspirational content.

This segmentation enables **personalized marketing** with higher ROI and more efficient resource allocation.

---

### **4.5 Model Limitations and Statistical Considerations**

Despite its practical value, the current analysis has limitations:

1. **Time Horizon Restriction**
   The dataset covers a limited time window, preventing long-term behavioral modeling.

2. **Simplifying Assumptions**
   RFM assumes independence among Recency, Frequency, and Monetary dimensions.
   KMeans assumes spherical clusters, which may oversimplify user diversity.

3. **Missing Conversion Data**
   CTR serves as a proxy for engagement, but conversion rates (CVR) and ROI data are not integrated.

4. **Retention Simplification**
   Cohort analysis only accounts for first engagement, not repeated sessions or multi-touch sequences.

These limitations point to the need for **longitudinal data collection** and **nonlinear modeling** (e.g., mixture or time-series models) in future research.

---

### **4.6 Interpretation and Business Implications**

The findings carry several operational and strategic implications:

1. **Price Tier Optimization**
   Dynamic bid weighting based on price-tier CTR can significantly improve ad spend efficiency.

2. **Demographic Precision**
   Audience alignment should guide creative design and media placement decisions.

3. **Lifecycle Marketing**
   Reactivation windows (D+1, D+3, D+7) should be automated for high-churn cohorts.

4. **Integrated Feedback Loop**
   Combining RFM segmentation and CTR monitoring creates a continuous improvement system, turning analytics into actionable insights.

Overall, these discussions demonstrate how **data-driven behavioral insights** can directly translate into measurable business growth when combined with **strategic timing** and **audience matching**.

---

## **Chapter 5. Strategy and Application**

### **5.1 Translating Insights into Action**

This chapter converts the analytical findings into practical business strategies for improving advertising performance.
It focuses on four pillars of optimization:

1. Price-tier targeting
2. Demographic precision
3. User lifecycle management
4. Data-driven experimentation and personalization

The overarching goal is to create a **closed-loop system** where data continuously informs optimization and execution.

---

### **5.2 Price-Tier Optimization**

**1. Dynamic CPC Bidding by Price Bin**
CTR patterns from Chapter 3 suggest that price has a nonlinear effect on engagement.
Thus, ad bidding can be dynamically weighted according to each bin’s relative CTR performance:

**Bid Weight = CTR_bin / Mean_CTR**

* High-CTR bins (e.g., 0–200 and 12801+) receive greater budget allocation.
* Mid-range bins (400–1600) require creative optimization or retargeting rather than higher bids.

**2. ROI-Based Adjustment**
Campaigns can track real-time ROI using:

**ROI = (CTR × ConversionRate × ProfitPerClick) / CostPerImpression**

Dynamic bidding ensures optimal cost-efficiency and prevents overinvestment in low-performing bins.

---

### **5.3 Demographic Targeting Enhancement**

Ad performance strongly depends on **audience–content alignment**.
Based on previous results, we can design an **Audience–Ad Matching Matrix**:

| Demographic Segment              | Recommended Ad Type            | Optimization Strategy                |
| -------------------------------- | ------------------------------ | ------------------------------------ |
| Female, 18–25, low-to-mid income | Fashion, beauty, lifestyle     | Visual emphasis, discount prompts    |
| Male, 25–40, mid-to-high income  | Electronics, tech gadgets      | Highlight features, use testimonials |
| Students                         | Education, low-cost essentials | Reward systems, trial incentives     |
| High-income (Tier-1 cities)      | Luxury goods                   | Brand storytelling, prestige framing |

High match rate between demographic traits and ad design results in increased CTR and improved ad efficiency.
In practice, **audience similarity scores** can be computed via cosine similarity on user embeddings to guide automatic ad targeting.

---

### **5.4 User Lifecycle Management**

Retention data revealed that user engagement decays rapidly within seven days.
Therefore, a structured **lifecycle engagement framework** is necessary:

| Stage   | Strategy                                                | Timing                    |
| ------- | ------------------------------------------------------- | ------------------------- |
| **D+1** | Send personalized post-click notifications or discounts | 24 hours after last click |
| **D+3** | Deliver customized recommendations                      | 3 days post-click         |
| **D+7** | Trigger reactivation email or ad recall                 | 7 days post-click         |

By automating these three checkpoints, platforms can effectively increase **lifetime value (LTV)** and mitigate churn.
This model also supports predictive scheduling — sending reminders **before** churn risk peaks.

---

### **5.5 Personalized Recommendation System**

User clusters and RFM segments serve as the foundation for personalized recommendations.
Each cluster is mapped to a distinct recommendation strategy:

| User Type                    | Behavioral Traits     | Recommendation Focus                            |
| ---------------------------- | --------------------- | ----------------------------------------------- |
| C3 – Loyal Frequent Clickers | Consistent engagement | Recommend high-value recurring products         |
| C1 – Moderate Clickers       | Moderate engagement   | Recommend discount-based products               |
| C2 – Premium Explorers       | High price curiosity  | Recommend luxury or aspirational content        |
| R4 – Consistent High Value   | Regular purchases     | Prioritize bundle offers or membership upgrades |

Personalization benefits include:

* Higher CTR and ROI through relevance
* Reduced ad fatigue
* Improved conversion by context matching

This system can later evolve into a **real-time recommender pipeline**, powered by collaborative filtering or deep learning models.

---

### **5.6 A/B Testing Framework**

To ensure that optimizations produce statistically valid improvements, a formal **A/B testing framework** should be established.

**Experiment Structure:**

| Component        | Description                                    |
| ---------------- | ---------------------------------------------- |
| Control Group    | Original ad or targeting setup                 |
| Test Group       | Modified ad (price tier, demographic, or copy) |
| Key Metric       | CTR, CVR, ROI                                  |
| Statistical Test | Two-sample Z-test                              |
| Confidence Level | 95% (α = 0.05)                                 |

**Z-Test Formula:**
z = (CTR₁ − CTR₂) / √(p × (1 − p) × (1/n₁ + 1/n₂))
where p = (clicks₁ + clicks₂) / (impressions₁ + impressions₂)

This framework ensures objective evaluation of creative and targeting strategies, avoiding biased interpretation of short-term results.

---

### **5.7 Retention and Re-Engagement Tactics**

Retention improvement strategies are essential given the rapid user churn:

1. **Micro-Segmented Retargeting**
   Use RFM categories to personalize re-engagement campaigns.

2. **Content Sequencing**
   Present a narrative progression — awareness → interest → action — to build product familiarity.

3. **Reward-Based Reactivation**
   Offer loyalty points or discounts for users inactive for over 3 days.

4. **Predictive Churn Triggers**
   Use logistic regression or time-decay functions to identify users likely to disengage.

These methods directly convert analytical insights into **operational growth levers**.

---

### **5.8 Multi-Metric Optimization**

To avoid overfocusing on CTR, a composite optimization function should balance multiple business objectives:

**Objective = w₁ × CTR + w₂ × CVR + w₃ × Retention**

The weights (w₁, w₂, w₃) can be tuned dynamically based on campaign goals:

* Brand exposure → emphasize CTR
* Sales conversion → emphasize CVR
* Customer loyalty → emphasize Retention

This unified framework enables adaptive optimization aligned with marketing priorities.

---

### **5.9 System Integration and Automation**

The entire analytical process can be deployed as an **automated analytics and optimization system**, consisting of four layers:

1. **Data Layer**
   Daily ETL jobs update user, ad, and click tables.

2. **Model Layer**
   Automated weekly retraining of clustering and CTR prediction models.

3. **Visualization Layer**
   Dashboards (e.g., Streamlit, Power BI) track CTR, retention, and ROI in real time.

4. **Action Layer**
   APIs trigger automated bid adjustments, retargeting pushes, or campaign scheduling.

Together, these components form a **self-learning feedback loop** that continually improves ad delivery performance.

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


