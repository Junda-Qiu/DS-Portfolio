
# 📊 Ad Analytics End-to-End Report

**Project Overview**  
An end-to-end user behavior and ad performance analysis project, built upon a large-scale advertising clickstream dataset.  
This report explores click-through patterns, price sensitivity, RFM segmentation, behavioral clustering, and user retention.

---

## 1. Project Background & Objectives
In digital advertising, advertisers aim to maximize ROI through accurate targeting, platforms strive for better CTR and user engagement, and users desire relevant and valuable ad content.  
This project constructs a full analytical framework to uncover insights across the ad engagement funnel.

### Key Goals
- Identify CTR differences across price tiers  
- Detect high- and low-performing ad placements  
- Segment users with RFM methodology  
- Analyze behavioral clusters and engagement types  
- Quantify retention and repurchase trends

---

## 2. Dataset Overview & Field Value Analysis

**Source:** `processed_user_ad.csv`  
**Rows:** 975,441 | **Columns:** 19

| Field | Type | Missing Rate | Description | Analytical Value |
|--------|------|--------------|--------------|------------------|
| `userid` | int | 0% | Unique user identifier | Key for user-level aggregation |
| `ad_id` | int | 0% | Unique ad placement ID | Primary key for CTR computation |
| `clk` | int | 0% | Click flag (1=clicked) | Target variable |
| `price` / `price_capped` | float | 0% | Product price (trimmed extreme) | Measures price sensitivity |
| `shopping_level` | float | 0% | User shopping intent (1–3) | Captures user engagement level |
| `age_level` | float | 0% | Age segmentation (1–7) | Core user demographic |
| `gender` | object | 0% | Gender | Behavioral segmentation dimension |
| `city_tier` | float | 27.3% | City class | Regional differentiation potential |
| `is_student` | float | 0% | Student flag | Younger audience detection |
| `time_stamp` | datetime | 0% | Timestamp | Enables cohort and retention analysis |

---

## 3. CTR & Price Tier Analysis  

📊 **Figures:** `ctr_by_pricebin.png`, `usershare_by_pricebin.png`

| Price Bin | CTR (%) | User Share (%) |
|------------|----------|----------------|
| 0–100 | **5.49** | **51.07** |
| 101–200 | 4.62 | 31.29 |
| 201–400 | 4.68 | 29.80 |
| 401–800 | 4.60 | 17.07 |
| 801–1600 | 4.45 | 10.87 |
| 1601–3200 | 4.20 | 7.19 |
| 3201–6400 | 4.17 | 4.99 |
| 6401–12800 | 4.38 | 2.86 |
| 12801+ | 5.13 | 1.23 |

<img width="500" height="300" alt="ctr_by_pricebin" src="https://github.com/user-attachments/assets/1c78c650-0052-4248-a25d-ff26f0e1a9a3" />
<img width="500" height="300" alt="usershare_by_pricebin" src="https://github.com/user-attachments/assets/106fb0eb-6547-4ba4-9a05-1d1b576296a3" />


**Insights:**  
- CTR peaks in the low-price band (0–100) at **5.49%**, dominating user volume.  
- High-priced goods (>12,801) show a rebound, indicating **brand-driven clicks**.  
- Mid-price items yield stable performance but with diminishing sensitivity.

**Recommendations:**  
- Use **dual-tier strategy**: low-price for ROI efficiency, high-price for brand engagement.

---

## 4. Ad Placement & Audience Behavior  

📊 **Figures:** `ad_top10_age.png`, `ad_bottom10_age.png`, `clicked_price.png`

High CTR ads concentrate among young-to-middle-age groups, suggesting strong creative resonance.  
Low CTR ads, despite exposure, lack contextual relevance or visual appeal.

<img width="500" height="300" alt="ad_top10_age" src="https://github.com/user-attachments/assets/0c4b48d6-4da1-4f2b-98f7-f2231e2087cd" />
<img width="500" height="300" alt="ad_bottom10_age" src="https://github.com/user-attachments/assets/f5d58d1f-4902-4120-9835-67d8e44f8f0c" />
<img width="500" height="300" alt="clicked_price" src="https://github.com/user-attachments/assets/5208f828-92bc-4798-a202-c81ed269a419" />


**Takeaways:**  
1. Replicate top CTR creative patterns.  
2. Replace underperforming ads or limit frequency.  
3. Cross-check age × placement correlation for mismatch detection.

---

## 5. RFM Segmentation & Precision Strategy  

📊 **Figure:** `rfmcluster_age.png`

| Segment | Share (%) | CTR | Avg Price | Strategic Action |
|----------|------------|-----|------------|------------------|
| Loyal Heavy Users | 26.9 | High | Mid | Exclusive loyalty rewards |
| New Users | 25.9 | Mid | Low | Welcome coupons + guidance |
| Churned Users | 18.5 | Low | High | Reactivation campaigns |
| Retention Targets | 17.2 | Mid | Mid | Renewal incentives |
| Potential Users | 4.0 | High | Mid | Personalized recommendations |
| Premium Users | 4.6 | High | High | VIP tier programs |
| Stable Users | 1.5 | Mid | Mid | Light-touch engagement |
<img width="640" height="480" alt="rfmcluster_age" src="https://github.com/user-attachments/assets/2657e10e-0de3-4b00-bc40-4e815d800e8b" />

**Key Insight:** RFM reveals clear value hierarchy—top 10% users contribute most CTR.  
**Operational Focus:** Balance between reactivation (churned) and nurturing (potential).

---

## 6. Behavioral Clustering (K-Means)

📊 **Figure:** `cluster_age.png`

| Cluster | Share (%) | Traits | Label |
|----------|------------|--------|--------|
| C0 | 80.1 | Low engagement, low spend | "Passive Observers" |
| C1 | 8.1 | Stable CTR, mid-price | "Rational Shoppers" |
| C2 | 2.0 | High-price, low frequency | "Brand Seekers" |
| C3 | 3.2 | High CTR, frequent | "Active Engagers" |
| C4 | 6.6 | High spend, high CTR | "High-Value Elites" |
<img width="640" height="480" alt="cluster_age" src="https://github.com/user-attachments/assets/2399e743-4e09-4779-b5e7-c79bda583c4f" />

**Recommendation:** Tailor campaigns per cluster — C3/C4 as top ROI groups, C0 as reactivation focus.

---

## 7. Retention & Cohort Analysis  

📊 **Figure:** `cohort_retention_heatmap.png`

| Period | Retention (%) |
|---------|----------------|
| 7-Day | 2.46 |
| 14-Day | 0.00 |
| 30-Day | 0.00 |
<img width="600" height="300" alt="cohort_retention_heatmap" src="https://github.com/user-attachments/assets/08f8ca26-dc06-4482-bbf6-d1c70976514e" />

**Interpretation:** Retention drops steeply after day 7, reflecting lack of post-click engagement.  
**Action Plan:** Implement automated push cycles at D+1 / D+3 / D+7 intervals.

---

## 8. Price Sensitivity: Equal-Width vs Quantile Binning  

📊 **Figure:** `price_ctr_equalwidth.png`
<img width="500" height="300" alt="price_ctr_equalwidth" src="https://github.com/user-attachments/assets/5d92b0ae-6926-4d24-9ffd-35767c6b2792" />

Equal-width binning skews CTR toward low bands due to outliers.  
Quantile-based bins yield a more realistic representation of click behavior.

**Best Practice:** Prefer quantile segmentation for CTR-related reporting.

---

## 9. Strategic Recommendations

| Layer | Focus | Action |
|--------|--------|--------|
| Advertising | Budget optimization | Reallocate to high-CTR placements |
| User | Segmentation | Build RFM-driven cohorts |
| Retention | Lifecycle triggers | D+1/D+3/D+7 push model |
| Pricing | Dual-tier strategy | ROI + brand split |
| Data | Metric governance | Standardize fields and ROI tracking |

---

## 10. Project Impact & Future Work  

This project establishes a reusable analytical framework for ad performance optimization.  

**Next Steps:**  
1. Integrate cost and order data for ROI computation.  
2. Deploy predictive CTR models (LightGBM / XGBoost).  
3. Automate visualization via Streamlit dashboard.  
4. Conduct A/B testing for strategy validation.

**Business Value:**  
- For Platforms: Quantitative basis for ad placement tuning.  
- For Advertisers: Budget and targeting guidance.  
- For Users: More relevant, engaging ad experience.

---
