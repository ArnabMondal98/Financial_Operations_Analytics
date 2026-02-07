"""
Data Preprocessing Pipeline
Handles data cleaning, validation, and feature engineering for financial analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Data preprocessing and feature engineering pipeline"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.customers_df = None
        self.transactions_df = None
        self.processed_customers = None
        self.processed_transactions = None
        
    def load_data(self):
        """Load raw datasets"""
        print("Loading datasets...")
        
        self.customers_df = pd.read_csv(os.path.join(self.data_dir, 'customers.csv'))
        self.transactions_df = pd.read_csv(os.path.join(self.data_dir, 'transactions.csv'))
        
        # Convert date columns
        self.customers_df['signup_date'] = pd.to_datetime(self.customers_df['signup_date'])
        self.transactions_df['transaction_date'] = pd.to_datetime(self.transactions_df['transaction_date'])
        
        print(f"Loaded {len(self.customers_df)} customers and {len(self.transactions_df)} transactions")
        
        return self.customers_df, self.transactions_df
    
    def clean_data(self):
        """Clean and validate data"""
        print("Cleaning data...")
        
        # Remove duplicates
        self.customers_df = self.customers_df.drop_duplicates(subset=['customer_id'])
        self.transactions_df = self.transactions_df.drop_duplicates(subset=['transaction_id'])
        
        # Handle missing values
        self.customers_df['monthly_fee'] = self.customers_df['monthly_fee'].fillna(
            self.customers_df['monthly_fee'].median()
        )
        
        # Validate transaction amounts
        self.transactions_df['amount'] = pd.to_numeric(self.transactions_df['amount'], errors='coerce')
        self.transactions_df = self.transactions_df.dropna(subset=['amount'])
        
        # Ensure valid transaction statuses
        valid_statuses = ['Completed', 'Failed', 'Pending', 'Refunded']
        self.transactions_df = self.transactions_df[
            self.transactions_df['transaction_status'].isin(valid_statuses)
        ]
        
        print(f"After cleaning: {len(self.customers_df)} customers, {len(self.transactions_df)} transactions")
        
        return self.customers_df, self.transactions_df
    
    def engineer_customer_features(self):
        """Create customer-level features"""
        print("Engineering customer features...")
        
        # Get the reference date (latest transaction date)
        reference_date = self.transactions_df['transaction_date'].max()
        
        # Calculate customer-level metrics from transactions
        customer_metrics = self.transactions_df.groupby('customer_id').agg({
            'transaction_id': 'count',
            'amount': ['sum', 'mean', 'std'],
            'transaction_date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        customer_metrics.columns = [
            'customer_id', 'total_transactions', 'total_revenue',
            'avg_transaction', 'std_transaction', 'first_transaction', 'last_transaction'
        ]
        
        # Calculate successful transactions
        successful_txns = self.transactions_df[
            self.transactions_df['transaction_status'] == 'Completed'
        ].groupby('customer_id').size().reset_index(name='successful_transactions')
        
        # Calculate failed transactions
        failed_txns = self.transactions_df[
            self.transactions_df['transaction_status'] == 'Failed'
        ].groupby('customer_id').size().reset_index(name='failed_transactions')
        
        # Merge with customer data
        self.processed_customers = self.customers_df.merge(
            customer_metrics, on='customer_id', how='left'
        )
        self.processed_customers = self.processed_customers.merge(
            successful_txns, on='customer_id', how='left'
        )
        self.processed_customers = self.processed_customers.merge(
            failed_txns, on='customer_id', how='left'
        )
        
        # Fill NaN values
        self.processed_customers['failed_transactions'] = self.processed_customers['failed_transactions'].fillna(0)
        self.processed_customers['successful_transactions'] = self.processed_customers['successful_transactions'].fillna(0)
        
        # Calculate derived features
        self.processed_customers['customer_tenure_days'] = (
            reference_date - self.processed_customers['signup_date']
        ).dt.days
        
        self.processed_customers['days_since_last_transaction'] = (
            reference_date - self.processed_customers['last_transaction']
        ).dt.days
        
        self.processed_customers['transaction_frequency'] = (
            self.processed_customers['total_transactions'] / 
            (self.processed_customers['customer_tenure_days'] + 1) * 30  # Monthly frequency
        )
        
        self.processed_customers['payment_failure_rate'] = (
            self.processed_customers['failed_transactions'] / 
            (self.processed_customers['total_transactions'] + 1)
        )
        
        self.processed_customers['avg_monthly_revenue'] = (
            self.processed_customers['total_revenue'] / 
            (self.processed_customers['customer_tenure_days'] / 30 + 1)
        )
        
        # Identify churned customers (no transaction in last 60 days)
        self.processed_customers['is_churned'] = (
            self.processed_customers['days_since_last_transaction'] > 60
        ).astype(int)
        
        # Customer lifetime value
        self.processed_customers['customer_ltv'] = self.processed_customers['total_revenue']
        
        # Fill any remaining NaN values
        self.processed_customers = self.processed_customers.fillna(0)
        
        print(f"Created {len(self.processed_customers.columns)} customer features")
        
        return self.processed_customers
    
    def engineer_transaction_features(self):
        """Create transaction-level features"""
        print("Engineering transaction features...")
        
        self.processed_transactions = self.transactions_df.copy()
        
        # Time-based features
        self.processed_transactions['year'] = self.processed_transactions['transaction_date'].dt.year
        self.processed_transactions['month'] = self.processed_transactions['transaction_date'].dt.month
        self.processed_transactions['quarter'] = self.processed_transactions['transaction_date'].dt.quarter
        self.processed_transactions['day_of_week'] = self.processed_transactions['transaction_date'].dt.dayofweek
        self.processed_transactions['day_of_month'] = self.processed_transactions['transaction_date'].dt.day
        self.processed_transactions['week_of_year'] = self.processed_transactions['transaction_date'].dt.isocalendar().week
        
        # Is weekend
        self.processed_transactions['is_weekend'] = (
            self.processed_transactions['day_of_week'] >= 5
        ).astype(int)
        
        # Month start/end flags
        self.processed_transactions['is_month_start'] = (
            self.processed_transactions['day_of_month'] <= 5
        ).astype(int)
        self.processed_transactions['is_month_end'] = (
            self.processed_transactions['day_of_month'] >= 25
        ).astype(int)
        
        # Transaction success flag
        self.processed_transactions['is_successful'] = (
            self.processed_transactions['transaction_status'] == 'Completed'
        ).astype(int)
        
        # Revenue flag (positive amount)
        self.processed_transactions['is_revenue'] = (
            self.processed_transactions['amount'] > 0
        ).astype(int)
        
        print(f"Created {len(self.processed_transactions.columns)} transaction features")
        
        return self.processed_transactions
    
    def create_time_series_data(self):
        """Create time series aggregations for forecasting"""
        print("Creating time series data...")
        
        # Daily revenue
        daily_revenue = self.processed_transactions[
            self.processed_transactions['is_successful'] == 1
        ].groupby('transaction_date').agg({
            'amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        daily_revenue.columns = ['date', 'revenue', 'transaction_count']
        
        # Monthly revenue
        monthly_revenue = self.processed_transactions[
            self.processed_transactions['is_successful'] == 1
        ].groupby(['year', 'month']).agg({
            'amount': 'sum',
            'transaction_id': 'count',
            'customer_id': 'nunique'
        }).reset_index()
        monthly_revenue.columns = ['year', 'month', 'revenue', 'transaction_count', 'unique_customers']
        monthly_revenue['date'] = pd.to_datetime(
            monthly_revenue['year'].astype(str) + '-' + monthly_revenue['month'].astype(str) + '-01'
        )
        
        # Customer acquisition by month
        customer_acquisition = self.processed_customers.copy()
        customer_acquisition['signup_month'] = customer_acquisition['signup_date'].dt.to_period('M')
        monthly_acquisitions = customer_acquisition.groupby('signup_month').size().reset_index(name='new_customers')
        monthly_acquisitions['signup_month'] = monthly_acquisitions['signup_month'].astype(str)
        
        return daily_revenue, monthly_revenue, monthly_acquisitions
    
    def save_processed_data(self, output_dir):
        """Save processed datasets"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed customers
        customers_path = os.path.join(output_dir, 'processed_customers.csv')
        self.processed_customers.to_csv(customers_path, index=False)
        print(f"Saved processed customers to {customers_path}")
        
        # Save processed transactions
        transactions_path = os.path.join(output_dir, 'processed_transactions.csv')
        self.processed_transactions.to_csv(transactions_path, index=False)
        print(f"Saved processed transactions to {transactions_path}")
        
        # Save time series data
        daily_rev, monthly_rev, acquisitions = self.create_time_series_data()
        
        daily_rev.to_csv(os.path.join(output_dir, 'daily_revenue.csv'), index=False)
        monthly_rev.to_csv(os.path.join(output_dir, 'monthly_revenue.csv'), index=False)
        acquisitions.to_csv(os.path.join(output_dir, 'monthly_acquisitions.csv'), index=False)
        
        print("Saved time series data")
        
        return output_dir
    
    def run_pipeline(self, output_dir=None):
        """Run the complete preprocessing pipeline"""
        print("\n=== Starting Data Preprocessing Pipeline ===\n")
        
        self.load_data()
        self.clean_data()
        self.engineer_customer_features()
        self.engineer_transaction_features()
        
        if output_dir:
            self.save_processed_data(output_dir)
        
        print("\n=== Preprocessing Complete ===")
        
        return self.processed_customers, self.processed_transactions


def main():
    """Main entry point"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'outputs')
    
    preprocessor = DataPreprocessor(data_dir)
    customers, transactions = preprocessor.run_pipeline(output_dir)
    
    print("\n=== Feature Summary ===")
    print(f"Customer features: {list(customers.columns)}")
    print(f"\nChurn rate: {customers['is_churned'].mean():.2%}")
    print(f"Total customers: {len(customers)}")
    
    return customers, transactions


if __name__ == "__main__":
    main()
