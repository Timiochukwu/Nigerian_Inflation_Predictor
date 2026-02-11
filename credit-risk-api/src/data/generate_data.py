"""
Generate synthetic credit risk data for testing and development.
Simulates realistic credit risk scenarios based on Nigerian lending patterns.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings


def generate_demographics(n_customers: int, seed: int = 42) -> pd.DataFrame:
    """Generate customer demographics data."""
    np.random.seed(seed)

    # Nigerian banks and locations
    banks = ['GTBank', 'Access Bank', 'First Bank', 'Zenith Bank', 'UBA',
             'Stanbic IBTC', 'Fidelity Bank', 'Union Bank']
    locations = ['Lagos', 'Abuja', 'Port Harcourt', 'Kano', 'Ibadan',
                 'Enugu', 'Kaduna', 'Jos', 'Benin City', 'Calabar']
    employment_types = ['Permanent', 'Contract', 'Self-Employed', 'Temporary']
    education_levels = ['Secondary', 'HND/BSc', 'Masters', 'PhD', 'None']

    demographics = pd.DataFrame({
        'customerid': [f'CUST_{i:06d}' for i in range(1, n_customers + 1)],
        'bank_name_clients': np.random.choice(banks, n_customers),
        'bank_branch_clients': np.random.choice(locations, n_customers),
        'latitude_gps': np.random.uniform(4.0, 13.0, n_customers),  # Nigerian latitudes
        'longitude_gps': np.random.uniform(3.0, 15.0, n_customers),  # Nigerian longitudes
        'employment_status_clients': np.random.choice(employment_types, n_customers,
                                                      p=[0.4, 0.25, 0.25, 0.1]),
        'level_of_education_clients': np.random.choice(education_levels, n_customers,
                                                        p=[0.3, 0.4, 0.2, 0.05, 0.05]),
    })

    return demographics


def generate_loan_performance(demographics: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate current loan application data with target variable."""
    np.random.seed(seed + 1)

    n_customers = len(demographics)

    # Generate features that correlate with default risk
    systemloanid = [f'LOAN_{i:08d}' for i in range(1, n_customers + 1)]

    # Loan amounts (₦ thousands) - larger loans have higher default risk
    loanamount = np.random.lognormal(mean=11, sigma=0.8, size=n_customers)
    loanamount = np.clip(loanamount, 10_000, 5_000_000).astype(int)

    # Total due (loan + interest)
    interest_rate = np.random.uniform(0.15, 0.35, n_customers)
    totaldue = (loanamount * (1 + interest_rate)).astype(int)

    # Term (days) - longer terms have slightly higher default risk
    termdays = np.random.choice([30, 60, 90, 180, 365], n_customers,
                                p=[0.2, 0.3, 0.25, 0.15, 0.1])

    # Generate target based on realistic factors
    # Risk factors: high loan amount, long term, self-employed, low education
    risk_score = np.zeros(n_customers)

    # Loan amount risk
    risk_score += (loanamount > np.median(loanamount)) * 0.3

    # Term risk
    risk_score += (termdays > 90) * 0.2

    # Employment risk
    employment = demographics['employment_status_clients'].values
    risk_score += np.where(employment == 'Self-Employed', 0.25, 0)
    risk_score += np.where(employment == 'Temporary', 0.35, 0)

    # Education risk
    education = demographics['level_of_education_clients'].values
    risk_score += np.where(education.isin(['None', 'Secondary']), 0.2, 0)

    # Add randomness
    risk_score += np.random.normal(0, 0.15, n_customers)

    # Convert to binary target (30% default rate)
    threshold = np.percentile(risk_score, 70)
    good_bad_flag = np.where(risk_score < threshold, 'Good', 'Bad')

    performance = pd.DataFrame({
        'customerid': demographics['customerid'].values,
        'systemloanid': systemloanid,
        'loanamount': loanamount,
        'totaldue': totaldue,
        'termdays': termdays,
        'good_bad_flag': good_bad_flag,
    })

    return performance


def generate_previous_loans(demographics: pd.DataFrame,
                            performance: pd.DataFrame,
                            seed: int = 42) -> pd.DataFrame:
    """Generate historical loan records (multiple per customer)."""
    np.random.seed(seed + 2)

    n_customers = len(demographics)
    previous_loans = []

    for idx, customer_id in enumerate(demographics['customerid'].values):
        # Number of previous loans (0-10, weighted towards fewer)
        n_prev_loans = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                        p=[0.1, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01])

        if n_prev_loans == 0:
            continue

        for loan_idx in range(n_prev_loans):
            # Historical loan amount
            prev_loanamount = np.random.lognormal(mean=10.5, sigma=0.9)
            prev_loanamount = int(np.clip(prev_loanamount, 5_000, 3_000_000))

            # Repayment history
            # Good customers have better history
            is_good_customer = performance.loc[performance['customerid'] == customer_id, 'good_bad_flag'].values[0] == 'Good'

            if is_good_customer:
                closeddate = np.random.choice(['2023-05-15', '2023-08-22', '2024-01-10', '2024-06-30'],
                                              p=[0.1, 0.2, 0.3, 0.4])
                approveddate = '2023-01-15'  # Earlier approval
                creationdate = '2022-12-20'
                firstduedate = '2023-02-15'
                firstrepaiddate = np.random.choice(['2023-02-14', '2023-02-15', '2023-02-16', '2023-02-20'],
                                                   p=[0.5, 0.3, 0.15, 0.05])  # Mostly on time
            else:
                closeddate = np.random.choice(['2023-03-10', '2023-07-22', '2024-02-15', None],
                                              p=[0.2, 0.2, 0.3, 0.3])  # 30% still open/defaulted
                approveddate = '2023-01-20'
                creationdate = '2023-01-05'
                firstduedate = '2023-02-20'
                firstrepaiddate = np.random.choice(['2023-02-20', '2023-02-25', '2023-03-05', None],
                                                   p=[0.3, 0.3, 0.2, 0.2])  # Late or missed

            previous_loans.append({
                'customerid': customer_id,
                'systemloanid': f'PREV_{idx:06d}_{loan_idx:02d}',
                'loanamount': prev_loanamount,
                'totaldue': int(prev_loanamount * np.random.uniform(1.15, 1.4)),
                'termdays': np.random.choice([30, 60, 90, 180]),
                'creationdate': creationdate,
                'approveddate': approveddate,
                'firstduedate': firstduedate,
                'firstrepaiddate': firstrepaiddate,
                'closeddate': closeddate,
            })

    return pd.DataFrame(previous_loans)


def generate_dataset(n_customers: int = 5000, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate complete synthetic credit risk dataset.

    Args:
        n_customers: Number of customers to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (demographics, performance, previous_loans) DataFrames
    """
    print(f"🎲 Generating synthetic credit risk data for {n_customers} customers...")

    demographics = generate_demographics(n_customers, seed)
    print(f"✅ Generated demographics: {demographics.shape}")

    performance = generate_loan_performance(demographics, seed)
    print(f"✅ Generated loan performance: {performance.shape}")

    previous_loans = generate_previous_loans(demographics, performance, seed)
    print(f"✅ Generated previous loans: {previous_loans.shape}")

    # Print class distribution
    class_dist = performance['good_bad_flag'].value_counts()
    print(f"\n📊 Class Distribution:")
    print(f"   Good: {class_dist['Good']} ({class_dist['Good']/len(performance)*100:.1f}%)")
    print(f"   Bad:  {class_dist['Bad']} ({class_dist['Bad']/len(performance)*100:.1f}%)")

    return demographics, performance, previous_loans


def save_dataset(demographics: pd.DataFrame,
                performance: pd.DataFrame,
                previous_loans: pd.DataFrame,
                output_dir: Path = None):
    """Save generated datasets to CSV files."""
    if output_dir is None:
        output_dir = settings.RAW_DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    demographics.to_csv(output_dir / 'traindemographics.csv', index=False)
    performance.to_csv(output_dir / 'trainperf.csv', index=False)
    previous_loans.to_csv(output_dir / 'trainprevloans.csv', index=False)

    print(f"\n💾 Saved datasets to: {output_dir}")
    print(f"   - traindemographics.csv")
    print(f"   - trainperf.csv")
    print(f"   - trainprevloans.csv")


if __name__ == "__main__":
    # Generate and save dataset
    demographics, performance, previous_loans = generate_dataset(n_customers=5000)
    save_dataset(demographics, performance, previous_loans)

    print("\n✨ Dataset generation complete!")
