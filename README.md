# 📉 Prescriptive ML: Telco Customer Churn & ROI Optimization

## 📊 Project Overview
Predicting customer churn is only half the battle; the real value lies in optimizing the retention strategy to maximize Return on Investment (ROI). This project goes beyond traditional machine learning metrics (like accuracy or AUC) by translating a Random Forest classification model into a direct business profitability calculator. 

Using the classic Kaggle Telco Customer Churn dataset, this pipeline demonstrates how targeted retention campaigns out-perform blind marketing strategies.

## 🛠️ Tech Stack & Methodology
* **Language:** Python
* **Libraries:** `pandas`, `scikit-learn`, `numpy`
* **Algorithm:** Random Forest Classifier (with `class_weight='balanced'` to handle imbalanced churn data).
* **Core Technique:** Cost-Benefit Matrix & ROI Uplift Calculation.

## 💡 Business Logic & Cost Assumptions
To evaluate the true business impact, I applied a realistic financial framework to the confusion matrix:
* **Retention Cost (Offer):** USD 20 (e.g., a discount voucher given to predicted churners).
* **Customer Lifetime Value (LTV) Saved:** USD 100 (revenue retained if a true churner stays).
* **True Positive (TP):** Spend USD 20, Save USD 100 ➡️ **Net Profit: +USD 80**
* **False Positive (FP):** Spend USD 20, Unnecessary ➡️ **Net Loss: -USD 20**

## 📈 Model Performance & Financial Impact
Based on the unseen test set of **1,407** users:

**1. Algorithm Metrics:**
* Built a balanced Random Forest model prioritizing Precision to avoid excessive false-positive marketing spend.
* Overall Accuracy: 78%

**2. Financial ROI Output:**
* **Blind Strategy Profit (Offer to all):** USD 9,260.00
* **Model-Driven Strategy Profit:** USD 11,500.00
* **🚀 Net ROI Uplift:** **+ USD 2,240.00**

**Executive Summary:** By deploying this predictive model to target only high-risk users, the business increases net retention profit by **over 24%** compared to a scattergun marketing approach. Scaled to a user base of 1 million, this prescriptive analytic pipeline translates to an estimated **$1.59M in annualized incremental profit**.

---
*Created by [Chenge Li] | Data Analyst / Data Scientist*
