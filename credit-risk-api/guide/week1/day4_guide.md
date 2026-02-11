# Week 1, Day 4 -- Deep EDA & Outlier Detection

## What You Will Learn Today

- How to analyze correlations between features and target
- How to create correlation heatmaps
- How to detect outliers using statistical methods
- How to visualize feature distributions
- How to identify which features are most predictive

## Why This Matters

You now have 35 features, but not all are equally useful. Some may be highly correlated with the target (`good_bad_flag`), while others may be noise. Some features may have outliers that could harm model performance.

Today you will:
1. **Identify the top 10 most correlated features** with default risk
2. **Visualize distributions** to spot problems
3. **Detect and handle outliers** using IQR method
4. **Create correlation heatmaps** to understand feature relationships

This analysis will guide your feature selection for modeling tomorrow.

---

## Key Concepts

| Concept | What It Means |
|---------|--------------|
| **Correlation** | How strongly two variables move together (-1 to +1) |
| **Outlier** | Extreme value that is far from most other values |
| **IQR Method** | Interquartile Range: values beyond Q1-1.5×IQR or Q3+1.5×IQR |
| **Distribution** | How values are spread across the range |

---

## What You'll Produce

1. Correlation analysis report
2. Top 10 most predictive features list
3. Outlier detection and handling
4. Cleaned dataset ready for modeling

---

## Tomorrow (Day 5)

Final preprocessing: train/test split, feature scaling, save modeling-ready data.
