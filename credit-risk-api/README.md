# 🏦 Credit Risk Scoring API - Complete Guide

**A comprehensive 20-day guide to building a production-ready credit risk prediction API**

---

## 📋 Project Overview

This project builds a **machine learning-powered credit risk scoring system** using data from the **Data Science Nigeria Credit Risk Prediction** competition. You'll learn how to:

- 📊 Process and merge multiple datasets
- 🔧 Engineer powerful features from historical data
- 🤖 Build and compare multiple ML models
- 🚀 Deploy a production API with FastAPI
- 📈 Monitor and maintain model performance

---

## 🎯 Business Problem

**Goal:** Predict whether a customer will be a "Good" or "Bad" loan risk based on:
- Demographics (age, location, employment, education)
- Current loan application details
- Historical loan repayment behavior

**Why it matters:**
- Reduces loan default rates
- Enables data-driven lending decisions
- Improves financial inclusion
- Automates risk assessment

---

## 📊 Dataset

### Data Files (from Kaggle)
1. **traindemographics.csv** (~4k rows)
   - Customer demographics, bank info, location

2. **trainperf.csv** (~4k rows)
   - **Contains TARGET:** `good_bad_flag` (Good/Bad)
   - Current loan application details

3. **trainprevloans.csv** (~18k rows)
   - Historical loan records (multiple per customer)
   - Repayment behavior, amounts, dates

### Data Download
👉 https://www.kaggle.com/competitions/data-science-nigeria-credit-risk-prediction/data

Place files in: `credit-risk-api/data/raw/`

---

## 📚 Guide Structure (20 Days)

### 🔵 **Week 1: Data Preparation** (Days 1-5)

| Day | Topic | Key Learnings |
|-----|-------|---------------|
| [Day 1](guide/week1/day1.md) | Project Setup & Data Exploration | Load data, initial EDA, visualizations |
| [Day 2](guide/week1/day2.md) | Data Merging & Relationships | Join datasets, handle 1:Many relationships |
| [Day 3](guide/week1/day3.md) | Feature Engineering | Create 50+ features from historical data |
| [Day 4](guide/week1/day4.md) | Deep EDA & Outlier Detection | Statistical analysis, correlation, outliers |
| [Day 5](guide/week1/day5.md) | Preprocessing Pipeline | Feature selection, scaling, train/test split |

**Outcomes:**
- ✅ Clean, merged dataset
- ✅ 50+ engineered features
- ✅ Production-ready preprocessing pipeline
- ✅ Train/test splits ready for modeling

---

### 🟢 **Week 2: Baseline Models** (Days 6-10)

| Day | Topic | Key Learnings |
|-----|-------|---------------|
| Day 6 | Baseline Model & Metrics | Simple model, accuracy, precision, recall, F1, ROC-AUC |
| Day 7 | Logistic Regression | Linear model, coefficients, interpretation |
| Day 8 | Model Evaluation Deep Dive | Confusion matrix, classification report, ROC curves |
| Day 9 | Feature Importance | Which features matter most? SHAP values |
| Day 10 | Cross-Validation | K-fold CV, model selection, avoiding overfitting |

**Outcomes:**
- ✅ Baseline model for comparison
- ✅ Logistic regression model
- ✅ Comprehensive evaluation framework
- ✅ Understanding of feature importance

---

### 🟡 **Week 3: Advanced Models** (Days 11-15)

| Day | Topic | Key Learnings |
|-----|-------|---------------|
| Day 11 | Tree-Based Models (Random Forest) | Ensemble methods, bagging, feature importance |
| Day 12 | XGBoost | Gradient boosting, handling imbalance |
| Day 13 | Hyperparameter Tuning | GridSearchCV, RandomizedSearchCV, Optuna |
| Day 14 | Model Comparison & Selection | Compare all models, select best performer |
| Day 15 | Model Calibration & Thresholds | Probability calibration, optimal threshold |

**Outcomes:**
- ✅ Random Forest model
- ✅ XGBoost model (likely best performer)
- ✅ Tuned hyperparameters
- ✅ Final production model selected

---

### 🔴 **Week 4: API Deployment** (Days 16-20)

| Day | Topic | Key Learnings |
|-----|-------|---------------|
| Day 16 | FastAPI Basics | API setup, routes, request/response models |
| Day 17 | API Endpoints | Prediction endpoint, batch prediction, model info |
| Day 18 | API Testing | Unit tests, integration tests, error handling |
| Day 19 | Documentation & README | API docs, usage examples, deployment guide |
| Day 20 | Final Integration | End-to-end testing, demo, presentation |

**Outcomes:**
- ✅ Production FastAPI application
- ✅ RESTful prediction endpoints
- ✅ Comprehensive tests
- ✅ Full documentation
- ✅ Deployable application

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Installation
```bash
# Clone/navigate to project
cd credit-risk-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Data
1. Go to: https://www.kaggle.com/competitions/data-science-nigeria-credit-risk-prediction/data
2. Download `traindemographics.csv`, `trainperf.csv`, `trainprevloans.csv`
3. Place in `credit-risk-api/data/raw/`

### Start Learning
📖 Begin with [Day 1: Project Setup & Data Exploration](guide/week1/day1.md)

---

## 📁 Project Structure

```
credit-risk-api/
├── data/
│   ├── raw/                    # Original CSV files
│   └── processed/              # Cleaned, merged, split data
├── notebooks/
│   ├── day1_exploration.ipynb
│   ├── day2_merging.ipynb
│   ├── day3_feature_engineering.ipynb
│   ├── day4_deep_eda.ipynb
│   ├── day5_preprocessing_pipeline.ipynb
│   └── ... (Week 2-4 notebooks)
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── utils.py                # Preprocessing functions
│   ├── models/                 # Model training scripts
│   └── api/                    # FastAPI application
├── models/
│   ├── artifacts/              # Saved scaler, encoders
│   ├── trained/                # Saved model files (.pkl)
│   └── evaluation/             # Model performance reports
├── tests/                      # Unit and integration tests
├── guide/
│   ├── week1/                  # Day 1-5 guides
│   ├── week2/                  # Day 6-10 guides
│   ├── week3/                  # Day 11-15 guides
│   └── week4/                  # Day 16-20 guides
├── requirements.txt
└── README.md                   # This file
```

---

## 🛠️ Tech Stack

### Data Processing
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning & preprocessing

### Machine Learning
- **scikit-learn** - Logistic Regression, Random Forest
- **xgboost** - Gradient boosting
- **imbalanced-learn** - Handling class imbalance (SMOTE)

### Visualization
- **matplotlib** - Basic plots
- **seaborn** - Statistical visualizations
- **shap** - Model interpretation

### API Development
- **FastAPI** - REST API framework
- **pydantic** - Data validation
- **uvicorn** - ASGI server

### Testing & Tools
- **pytest** - Testing framework
- **jupyter** - Interactive notebooks
- **joblib** - Model serialization

---

## 📈 Expected Results

### Model Performance Goals
- **Accuracy:** >80%
- **ROC-AUC:** >0.85
- **Precision:** >75% (minimize false positives)
- **Recall:** >80% (catch most bad loans)

### API Performance Goals
- **Response time:** <100ms for single prediction
- **Throughput:** >100 requests/second
- **Uptime:** 99%+

---

## 🎓 Learning Outcomes

After completing this guide, you will be able to:

1. **Data Engineering**
   - Merge multiple datasets correctly
   - Handle missing data strategically
   - Engineer features from historical data
   - Create reusable preprocessing pipelines

2. **Machine Learning**
   - Build classification models (Logistic Regression, Random Forest, XGBoost)
   - Evaluate models comprehensively (accuracy, precision, recall, F1, ROC-AUC)
   - Handle class imbalance
   - Tune hyperparameters
   - Interpret model predictions (SHAP, feature importance)

3. **Software Engineering**
   - Build RESTful APIs with FastAPI
   - Write unit and integration tests
   - Create production-ready code
   - Document your work professionally

4. **MLOps**
   - Save and version models
   - Create reproducible pipelines
   - Deploy models to production
   - Monitor model performance

---

## 💡 Tips for Success

### Time Management
- **Consistent practice:** 2-3 hours per day
- **Don't skip days:** Each day builds on previous work
- **Review regularly:** Revisit previous notebooks

### Best Practices
1. **Run all code yourself** - Don't just read, execute!
2. **Experiment** - Try different parameters, features
3. **Document observations** - Add markdown cells with insights
4. **Save frequently** - Git commit after each day
5. **Ask questions** - Research concepts you don't understand

### Troubleshooting
- **Code errors:** Check data types, shapes, missing values
- **Poor performance:** Try different features, models, hyperparameters
- **Overfitting:** More data, regularization, cross-validation
- **Underfitting:** More features, complex models, less regularization

---

## 📖 Resources

### Documentation
- [Pandas](https://pandas.pydata.org/docs/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [FastAPI](https://fastapi.tiangolo.com/)

### Tutorials
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Real Python](https://realpython.com/)

### Books
- "Hands-On Machine Learning" by Aurélien Géron
- "The Elements of Statistical Learning" by Hastie, Tibshirani, Friedman
- "Feature Engineering for Machine Learning" by Alice Zheng

---

## 🤝 Contributing

This is a learning project! Feel free to:
- Add new features
- Try different models
- Improve the API
- Optimize performance
- Share your results

---

## 📝 License

This project is for educational purposes.

---

## 🎯 Current Progress

### Week 1: Data Preparation ✅ COMPLETE
- [x] Day 1: Project Setup & Data Exploration
- [x] Day 2: Data Merging & Relationships
- [x] Day 3: Feature Engineering
- [x] Day 4: Deep EDA & Outlier Detection
- [x] Day 5: Preprocessing Pipeline

### Week 2: Baseline Models 🚧 COMING SOON
- [ ] Day 6: Baseline Model & Metrics
- [ ] Day 7: Logistic Regression
- [ ] Day 8: Model Evaluation Deep Dive
- [ ] Day 9: Feature Importance
- [ ] Day 10: Cross-Validation

### Week 3: Advanced Models 📅 PLANNED
- [ ] Day 11-15

### Week 4: API Deployment 📅 PLANNED
- [ ] Day 16-20

---

## 🚀 Let's Get Started!

Ready to build your credit risk scoring system?

👉 **Start here:** [Day 1: Project Setup & Data Exploration](guide/week1/day1.md)

---

## 📞 Support

Questions or stuck?
- Review the day's guide carefully
- Check previous days for prerequisites
- Experiment with the code
- Research error messages

---

**🌟 Good luck on your machine learning journey! 🌟**

*Last updated: 2026-02-11*
