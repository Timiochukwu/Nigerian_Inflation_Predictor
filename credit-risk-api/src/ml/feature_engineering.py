"""
Feature engineering for credit risk prediction.
Creates powerful features from customer demographics and loan history.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV files from data/raw directory."""
    demographics = pd.read_csv(settings.RAW_DATA_DIR / 'traindemographics.csv')
    performance = pd.read_csv(settings.RAW_DATA_DIR / 'trainperf.csv')
    previous_loans = pd.read_csv(settings.RAW_DATA_DIR / 'trainprevloans.csv')

    print(f"📊 Loaded data:")
    print(f"   Demographics: {demographics.shape}")
    print(f"   Performance: {performance.shape}")
    print(f"   Previous Loans: {previous_loans.shape}")

    return demographics, performance, previous_loans


def engineer_historical_features(prev_loans: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer aggregated features from historical loan data.

    Features include:
    - Count statistics (total loans, closed loans, open loans)
    - Amount statistics (avg, max, min loan amounts)
    - Repayment behavior (on-time rate, late rate, default rate)
    - Temporal features (days since last loan, recency)
    """
    # Convert date columns to datetime
    date_cols = ['creationdate', 'approveddate', 'firstduedate', 'firstrepaiddate', 'closeddate']
    for col in date_cols:
        if col in prev_loans.columns:
            prev_loans[col] = pd.to_datetime(prev_loans[col], errors='coerce')

    # Group by customer
    features_list = []

    for customer_id, group in prev_loans.groupby('customerid'):
        features = {'customerid': customer_id}

        # --- COUNT FEATURES ---
        features['hist_num_loans'] = len(group)
        features['hist_num_closed'] = group['closeddate'].notna().sum()
        features['hist_num_open'] = group['closeddate'].isna().sum()

        # --- AMOUNT FEATURES ---
        features['hist_avg_loan_amount'] = group['loanamount'].mean()
        features['hist_max_loan_amount'] = group['loanamount'].max()
        features['hist_min_loan_amount'] = group['loanamount'].min()
        features['hist_total_borrowed'] = group['loanamount'].sum()
        features['hist_avg_total_due'] = group['totaldue'].mean()

        # --- REPAYMENT FEATURES ---
        # Calculate days to first repayment
        group['days_to_repay'] = (group['firstrepaiddate'] - group['firstduedate']).dt.days

        # On-time payment (paid before or on due date)
        on_time = (group['days_to_repay'] <= 0).sum()
        late = (group['days_to_repay'] > 0).sum()
        never_paid = group['firstrepaiddate'].isna().sum()

        total_repaid = len(group) - group['firstrepaiddate'].isna().sum()
        features['hist_ontime_rate'] = on_time / len(group) if len(group) > 0 else 0
        features['hist_late_rate'] = late / len(group) if len(group) > 0 else 0
        features['hist_never_paid_rate'] = never_paid / len(group) if len(group) > 0 else 0

        # Average days late (for late payments only)
        late_days = group.loc[group['days_to_repay'] > 0, 'days_to_repay']
        features['hist_avg_days_late'] = late_days.mean() if len(late_days) > 0 else 0

        # --- CLOSURE FEATURES ---
        features['hist_closure_rate'] = features['hist_num_closed'] / features['hist_num_loans']

        # --- TEMPORAL FEATURES ---
        # Days since most recent loan
        most_recent_approved = group['approveddate'].max()
        if pd.notna(most_recent_approved):
            reference_date = pd.Timestamp('2024-07-01')  # Assume current date
            features['days_since_last_loan'] = (reference_date - most_recent_approved).days
        else:
            features['days_since_last_loan'] = 9999  # Very old or no loans

        # --- TERM FEATURES ---
        features['hist_avg_term_days'] = group['termdays'].mean()
        features['hist_max_term_days'] = group['termdays'].max()

        # --- RISK INDICATORS ---
        # Has any loan still open (potential ongoing default)
        features['has_open_loans'] = 1 if features['hist_num_open'] > 0 else 0

        # Ratio of total due to total borrowed (interest burden)
        if features['hist_total_borrowed'] > 0:
            features['hist_interest_burden'] = (group['totaldue'].sum() / features['hist_total_borrowed']) - 1
        else:
            features['hist_interest_burden'] = 0

        features_list.append(features)

    hist_features = pd.DataFrame(features_list)
    print(f"✅ Engineered {len(hist_features.columns)-1} historical features for {len(hist_features)} customers")

    return hist_features


def engineer_demographic_features(demographics: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from demographics data."""
    demo_features = demographics.copy()

    # One-hot encode categorical variables
    demo_features = pd.get_dummies(demo_features,
                                   columns=['employment_status_clients', 'level_of_education_clients'],
                                   prefix=['emp', 'edu'],
                                   drop_first=False)

    # Simple bank and location encoding (label encoding for high cardinality)
    demo_features['bank_encoded'] = pd.Categorical(demo_features['bank_name_clients']).codes
    demo_features['location_encoded'] = pd.Categorical(demo_features['bank_branch_clients']).codes

    # Drop original high-cardinality columns
    demo_features = demo_features.drop(['bank_name_clients', 'bank_branch_clients'], axis=1)

    print(f"✅ Engineered demographic features: {demo_features.shape[1]-1} features")

    return demo_features


def engineer_current_loan_features(performance: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from current loan application."""
    current_features = performance.copy()

    # Loan-to-due ratio (lower is better - less interest)
    current_features['loan_to_due_ratio'] = current_features['loanamount'] / current_features['totaldue']

    # Interest amount
    current_features['interest_amount'] = current_features['totaldue'] - current_features['loanamount']

    # Interest rate (implied)
    current_features['implied_interest_rate'] = (current_features['totaldue'] / current_features['loanamount']) - 1

    # Loan amount bins (categorize loan sizes)
    current_features['loan_size_category'] = pd.cut(current_features['loanamount'],
                                                     bins=[0, 50000, 200000, 500000, float('inf')],
                                                     labels=['small', 'medium', 'large', 'very_large'])
    current_features['loan_size_category'] = pd.Categorical(current_features['loan_size_category']).codes

    # Term bins
    current_features['term_category'] = pd.cut(current_features['termdays'],
                                               bins=[0, 60, 180, float('inf')],
                                               labels=['short', 'medium', 'long'])
    current_features['term_category'] = pd.Categorical(current_features['term_category']).codes

    print(f"✅ Engineered current loan features: {current_features.shape[1]-2} features (excluding customerid and systemloanid)")

    return current_features


def create_master_dataset() -> pd.DataFrame:
    """
    Create master dataset by:
    1. Loading raw data
    2. Engineering features
    3. Merging all data sources
    4. Handling missing values for customers with no loan history
    """
    # Load data
    demographics, performance, previous_loans = load_raw_data()

    # Engineer features
    print("\n🔧 Engineering features...")
    demo_features = engineer_demographic_features(demographics)
    current_features = engineer_current_loan_features(performance)
    hist_features = engineer_historical_features(previous_loans)

    # Merge: Start with performance (has target)
    master_df = current_features.copy()

    # Merge demographics
    master_df = master_df.merge(demo_features, on='customerid', how='left')

    # Merge historical features (left join - some customers may have no history)
    master_df = master_df.merge(hist_features, on='customerid', how='left')

    # Fill NaN for customers with no loan history
    hist_cols = hist_features.columns.drop('customerid')
    for col in hist_cols:
        if col in master_df.columns:
            # For count/rate features, fill with 0
            if 'num' in col or 'rate' in col or 'has_' in col:
                master_df[col] = master_df[col].fillna(0)
            # For amount/days features, fill with median or 0
            elif 'amount' in col or 'days' in col or 'term' in col:
                master_df[col] = master_df[col].fillna(master_df[col].median())
            else:
                master_df[col] = master_df[col].fillna(0)

    # Encode target
    master_df['target'] = (master_df['good_bad_flag'] == 'Bad').astype(int)

    print(f"\n✅ Master dataset created: {master_df.shape}")
    print(f"   Features: {master_df.shape[1] - 3}")  # Exclude customerid, systemloanid, good_bad_flag, target
    print(f"   Target distribution:")
    print(f"      Good (0): {(master_df['target'] == 0).sum()} ({(master_df['target'] == 0).sum()/len(master_df)*100:.1f}%)")
    print(f"      Bad (1):  {(master_df['target'] == 1).sum()} ({(master_df['target'] == 1).sum()/len(master_df)*100:.1f}%)")

    return master_df


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features (X) and target (y) for modeling.

    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature column names
    """
    # Drop non-feature columns
    drop_cols = ['customerid', 'systemloanid', 'good_bad_flag', 'target']
    feature_cols = [col for col in df.columns if col not in drop_cols]

    X = df[feature_cols]
    y = df['target']

    print(f"\n📋 Features prepared:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Feature columns: {len(feature_cols)}")

    return X, y, feature_cols


def save_processed_data(df: pd.DataFrame, filename: str = 'master_dataset.csv'):
    """Save processed dataset."""
    output_path = settings.PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved processed data to: {output_path}")


if __name__ == "__main__":
    # Create master dataset
    master_df = create_master_dataset()

    # Save it
    save_processed_data(master_df)

    # Prepare features and target
    X, y, feature_names = prepare_features_and_target(master_df)

    print("\n✨ Feature engineering complete!")
    print(f"\nTop 10 features:")
    for i, feat in enumerate(feature_names[:10], 1):
        print(f"   {i}. {feat}")
