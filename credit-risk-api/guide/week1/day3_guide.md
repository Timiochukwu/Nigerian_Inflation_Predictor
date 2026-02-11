# Week 1, Day 3 -- Feature Engineering: Mining Gold from Historical Loan Data

## What You Will Learn Today

- How to engineer features from date columns (payment behavior)
- How to calculate on-time payment rate, late payment rate
- How to measure loan closure rate (completion vs abandonment)
- How to create recency features (days since last loan)
- How to encode categorical variables (employment, education)
- How to build a complete feature-rich dataset

## Why This Matters

Yesterday you merged three datasets and created 19 basic features. Today you will **engineer 15+ powerful new features** from the raw data that will dramatically improve model accuracy.

The single most important predictor of credit risk is **past payment behavior**. A customer who consistently pays late or abandons loans is a high-risk borrower.

---

## Part 1: Verify You Have Day 2's Output

```bash
ls -lh credit-risk-api/data/processed/master_dataset_day2.csv
```

You should see the 843 KB file from yesterday.

---

## Part 2: Building the Practice Script (Continues from Day 2 format)

Today's guide covers the key concepts. The full `day3_practice.py` script will build on yesterday's work.

### Key Features We'll Engineer:

**From Date Columns:**
1. **days_to_repay** - Days between due date and actual payment
2. **hist_ontime_rate** - % of loans paid on time
3. **hist_late_rate** - % of loans paid late
4. **hist_never_paid_rate** - % of loans never paid
5. **hist_avg_days_late** - Average days late (when late)
6. **hist_closure_rate** - % of loans completed vs abandoned
7. **days_since_last_loan** - Recency indicator

**From Categorical Variables:**
8. One-hot encoding for employment_status (4 categories)
9. One-hot encoding for education_level (4-5 categories)

**Total: 15+ new features**

---

## Tomorrow (Day 4): Deep EDA & Outlier Detection

You'll analyze correlations, visualize distributions, and handle outliers before modeling.

---

## Tomorrow (Day 5): Final Preprocessing

Train/test split, feature scaling, and save final modeling-ready dataset.
