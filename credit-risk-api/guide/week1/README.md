# 📅 Week 1: Data Preparation & Feature Engineering

**Duration:** 5 Days
**Goal:** Transform raw data into a clean, feature-rich dataset ready for machine learning

---

## 🎯 Week Overview

This week focuses on the **foundation** of any ML project: understanding, cleaning, and preparing your data. You'll learn industry-standard data engineering practices used in production ML systems.

---

## 📊 What You'll Accomplish

By the end of Week 1, you will have:

✅ **Loaded and explored** 3 datasets (18k+ rows total)
✅ **Merged** datasets correctly (handling 1:1 and 1:Many relationships)
✅ **Engineered 50+ features** from historical loan data
✅ **Performed deep EDA** with statistical analysis and visualizations
✅ **Built a preprocessing pipeline** with feature selection and scaling
✅ **Created train/test splits** ready for modeling
✅ **Saved reusable artifacts** for production deployment

---

## 📚 Daily Breakdown

### [Day 1: Project Setup & Data Exploration](day1.md)
**Time:** 2-3 hours

**Topics:**
- Setting up Python environment
- Loading CSV files with pandas
- Initial data exploration (shape, dtypes, missing values)
- Target variable analysis
- Basic visualizations

**Key Outputs:**
- Environment configured with all libraries
- First Jupyter notebook: `day1_exploration.ipynb`
- Understanding of data structure and relationships

**Skills Learned:**
- Pandas basics (`.head()`, `.info()`, `.describe()`)
- Data loading best practices
- Identifying target variable and class imbalance
- Creating histograms, bar plots, scatter plots

---

### [Day 2: Data Merging & Relationships](day2.md)
**Time:** 3-4 hours

**Topics:**
- Understanding 1:1 vs 1:Many relationships
- Merging demographics and performance data
- Aggregating historical loans data
- Creating 20+ features from previous loans
- Data validation after merging

**Key Outputs:**
- Merged master dataset: `merged_data.csv`
- Second notebook: `day2_merging.ipynb`
- Previous loan aggregated features

**Skills Learned:**
- Pandas merging (`.merge()` with different join types)
- GroupBy aggregations (`.groupby()`, `.agg()`)
- Creating features from historical data
- Data integrity validation

**Key Features Created:**
- Loan counts, amounts (sum, mean, min, max, std)
- Repayment behavior (early, late, closed)
- Days to repayment metrics
- Utilization rates

---

### [Day 3: Advanced Feature Engineering](day3.md)
**Time:** 3-4 hours

**Topics:**
- Time-based features (age, recency, tenure)
- Categorical encoding (one-hot, label encoding)
- Missing value strategies
- Ratio and interaction features
- Feature transformations (log, sqrt)

**Key Outputs:**
- Engineered dataset: `featured_data.csv`
- Feature names list: `feature_names.txt`
- Third notebook: `day3_feature_engineering.ipynb`

**Skills Learned:**
- Creating time-based features from dates
- One-hot encoding for categorical variables
- Strategic missing value imputation
- Creating domain-specific features (borrower score)
- Log transformations for skewed distributions

**Key Features Created:**
- Age, age groups
- Loan-to-age ratio
- Borrower behavior score
- Percentage metrics (% paid early, % paid late)
- Processing time features

---

### [Day 4: Deep EDA & Outlier Detection](day4.md)
**Time:** 3-4 hours

**Topics:**
- Univariate analysis (distributions, skewness)
- Bivariate analysis (feature vs target)
- Outlier detection (Z-score, IQR methods)
- Correlation analysis
- Statistical testing (t-tests)
- Multivariate analysis (pair plots)

**Key Outputs:**
- Cleaned dataset: `cleaned_data.csv`
- Feature correlation insights
- Fourth notebook: `day4_deep_eda.ipynb`

**Skills Learned:**
- Statistical outlier detection methods
- Winsorization for handling outliers
- Correlation matrix interpretation
- Identifying multicollinearity
- Statistical hypothesis testing
- Advanced visualization (heatmaps, pair plots)

**Key Insights:**
- Which features correlate with target
- Highly correlated feature pairs
- Outlier patterns and handling strategy
- Feature distributions by class

---

### [Day 5: Preprocessing Pipeline](day5.md)
**Time:** 3-4 hours

**Topics:**
- Removing low-variance features
- Handling multicollinearity
- Feature selection (ANOVA F-test, Mutual Information)
- Feature scaling (StandardScaler)
- Train-test splitting
- Saving preprocessing artifacts

**Key Outputs:**
- Train/test datasets: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- Scaler artifact: `scaler.pkl`
- Selected features: `selected_features.pkl`
- Feature scores: `feature_scores.csv`
- Preprocessing function: `src/utils.py`
- Fifth notebook: `day5_preprocessing_pipeline.ipynb`

**Skills Learned:**
- Feature selection techniques
- Feature scaling best practices
- Stratified train-test splitting
- Saving and loading sklearn artifacts
- Creating reusable preprocessing functions
- Production pipeline design

**Final Result:**
- ~50 selected features (from 100+)
- Scaled features (mean=0, std=1)
- 80-20 train-test split
- Production-ready preprocessing pipeline

---

## 📈 Data Journey

```
Day 1: Raw CSV files (3 files, ~22k rows)
   ↓
Day 2: Merged dataset (1 file, ~4k rows, ~20 columns)
   ↓
Day 3: Engineered features (~4k rows, ~100 columns)
   ↓
Day 4: Cleaned data (outliers handled, correlations analyzed)
   ↓
Day 5: Final preprocessed data (~50 selected features, scaled, split)
   ↓
Ready for modeling! 🚀
```

---

## 🔑 Key Takeaways

### Technical Skills
1. **Data Wrangling:** Merging, aggregating, reshaping data
2. **Feature Engineering:** Creating predictive features from raw data
3. **Statistical Analysis:** Understanding distributions, correlations, outliers
4. **Preprocessing:** Scaling, encoding, splitting data properly
5. **Pipeline Design:** Building reusable, production-ready code

### Domain Knowledge
1. **Credit Risk Factors:**
   - Previous repayment behavior is highly predictive
   - Age and loan amount relationship matters
   - Consistency in repayment is a strong signal

2. **Data Relationships:**
   - One customer can have many previous loans
   - Historical behavior predicts future performance
   - Demographics provide context but behavior dominates

### Best Practices
1. **Always validate after merging** - Check shapes, target distribution
2. **Document your decisions** - Why remove features, how handle missing data
3. **Save intermediate datasets** - Easier debugging and iteration
4. **Use descriptive names** - `prev_loan_mean` > `feature_23`
5. **Visualize before decisions** - Don't blindly apply techniques

---

## 📊 Final Dataset Characteristics

### Shape
- **Rows:** ~4,000 customers
- **Features:** ~50 (after selection from 100+)
- **Target:** Binary (Good=1, Bad=0)

### Feature Categories
- **Demographics:** 10-15 features (age, bank, location, employment)
- **Previous Loan Stats:** 20-25 features (counts, amounts, terms)
- **Behavioral:** 10-15 features (repayment patterns, scores)
- **Ratios & Interactions:** 5-10 features (loan-to-age, utilization)

### Data Quality
- ✅ No missing values
- ✅ Outliers handled (winsorized)
- ✅ Features scaled (StandardScaler)
- ✅ No duplicates
- ✅ Stratified split (class distribution maintained)

---

## 🛠️ Tools & Libraries Used

- **pandas:** Data manipulation and merging
- **numpy:** Numerical operations
- **matplotlib/seaborn:** Visualizations
- **scipy:** Statistical tests
- **scikit-learn:** Preprocessing, feature selection, train-test split
- **joblib:** Saving artifacts

---

## ✅ Week 1 Checklist

### Setup
- [ ] Python environment configured
- [ ] All libraries installed
- [ ] Data downloaded and placed in `data/raw/`
- [ ] Project structure created

### Completion
- [ ] All 5 notebooks created and executed
- [ ] All visualizations reviewed
- [ ] All intermediate datasets saved
- [ ] Preprocessing artifacts saved
- [ ] Week 1 concepts understood

---

## 🚀 Next Steps (Week 2)

You're now ready to build machine learning models!

**Week 2 Preview:**
- **Day 6:** Baseline model and evaluation metrics
- **Day 7:** Logistic Regression (interpretable linear model)
- **Day 8:** Deep dive into model evaluation
- **Day 9:** Feature importance and interpretation
- **Day 10:** Cross-validation and model selection

**What to expect:**
- Train your first real models
- Learn evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Understand which features drive predictions
- Compare models objectively

---

## 💡 Reflection Questions

Before moving to Week 2, ask yourself:

1. **Do I understand the data?**
   - What each feature represents?
   - How features were created?
   - What the target variable means?

2. **Can I explain the pipeline?**
   - Why we merged data this way?
   - Why we selected these features?
   - Why we scaled the data?

3. **Am I comfortable with the tools?**
   - Can I load and explore new data?
   - Can I create basic features?
   - Can I use pandas and sklearn?

If you answered "no" to any, **review that day's guide again!**

---

## 📚 Additional Resources

### Recommended Reading
- [Feature Engineering Book](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Kaggle Feature Engineering Course](https://www.kaggle.com/learn/feature-engineering)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)

### Practice Datasets
- [Kaggle: Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
- [UCI: Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

---

## 🎉 Congratulations!

You've completed **Week 1** of the Credit Risk Scoring API project!

You now have:
- ✅ A thorough understanding of your data
- ✅ 50+ engineered features
- ✅ A production-ready preprocessing pipeline
- ✅ Clean train/test datasets

**This is a HUGE accomplishment!** Many ML projects fail at this stage because of poor data preparation. You've built a solid foundation.

---

## 🌟 Ready for Week 2?

Time to build some models! 🚀

👉 **Continue to Week 2: Baseline Models** (Coming Soon)

---

*"Data preparation is 80% of machine learning. You've just mastered the hardest part!"*

---

**Happy Learning! 📚**
