"""
Synthetic Financial Data Generator
Generates realistic financial datasets with:
- Customer growth patterns
- Seasonality in revenue
- Churn behavior patterns
- Varied customer profitability
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class FinancialDataGenerator:
    """Generate synthetic financial datasets for analytics"""
    
    def __init__(self, start_date='2021-01-01', end_date='2024-12-31'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        # Configuration
        self.countries = ['USA', 'UK', 'Germany', 'France', 'Canada', 'Australia', 'Japan', 'Brazil', 'India', 'Singapore']
        self.industries = ['Technology', 'Healthcare', 'Finance', 'Retail', 'Manufacturing', 'Education', 'Media', 'Real Estate', 'Energy', 'Consulting']
        self.subscription_types = {
            'Basic': {'fee_range': (29, 99), 'churn_rate': 0.08},
            'Professional': {'fee_range': (99, 299), 'churn_rate': 0.05},
            'Enterprise': {'fee_range': (299, 999), 'churn_rate': 0.02},
            'Premium': {'fee_range': (999, 2999), 'churn_rate': 0.01}
        }
        self.payment_methods = ['Credit Card', 'Bank Transfer', 'PayPal', 'Wire Transfer', 'ACH']
        
    def _generate_customer_growth_pattern(self, total_customers=2500):
        """Generate realistic customer acquisition pattern with growth"""
        dates = []
        date_range = (self.end_date - self.start_date).days
        
        # Simulate S-curve growth with seasonal variations
        for i in range(total_customers):
            # S-curve growth pattern
            progress = i / total_customers
            # More customers join in later periods (growth acceleration)
            day_offset = int(date_range * (1 - np.exp(-3 * progress)) / (1 - np.exp(-3)))
            
            # Add some randomness
            day_offset += random.randint(-30, 30)
            day_offset = max(0, min(day_offset, date_range))
            
            signup_date = self.start_date + timedelta(days=day_offset)
            dates.append(signup_date)
        
        return sorted(dates)
    
    def generate_customers(self, num_customers=2500):
        """Generate customers.csv with realistic patterns"""
        print(f"Generating {num_customers} customers...")
        
        signup_dates = self._generate_customer_growth_pattern(num_customers)
        
        customers = []
        for i in range(num_customers):
            customer_id = f"CUST_{i+1:06d}"
            signup_date = signup_dates[i]
            
            # Country distribution (weighted)
            country_weights = [0.30, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
            country = np.random.choice(self.countries, p=country_weights)
            
            # Industry distribution
            industry = random.choice(self.industries)
            
            # Subscription type (weighted towards lower tiers)
            sub_weights = [0.40, 0.35, 0.18, 0.07]
            subscription_type = np.random.choice(list(self.subscription_types.keys()), p=sub_weights)
            
            # Monthly fee based on subscription type
            fee_range = self.subscription_types[subscription_type]['fee_range']
            monthly_fee = round(random.uniform(fee_range[0], fee_range[1]), 2)
            
            customers.append({
                'customer_id': customer_id,
                'signup_date': signup_date.strftime('%Y-%m-%d'),
                'country': country,
                'industry': industry,
                'subscription_type': subscription_type,
                'monthly_fee': monthly_fee
            })
        
        return pd.DataFrame(customers)
    
    def generate_transactions(self, customers_df):
        """Generate transactions.csv with realistic patterns including churn"""
        print("Generating transactions...")
        
        transactions = []
        transaction_id = 0
        
        for _, customer in customers_df.iterrows():
            customer_id = customer['customer_id']
            signup_date = pd.to_datetime(customer['signup_date'])
            monthly_fee = customer['monthly_fee']
            subscription_type = customer['subscription_type']
            
            # Get base churn rate for subscription type
            base_churn_rate = self.subscription_types[subscription_type]['churn_rate']
            
            # Determine if and when customer churns
            churned = False
            churn_date = None
            
            # Some customers will churn
            if random.random() < 0.25:  # 25% of customers eventually churn
                # Churn probability increases with time
                months_active = random.randint(2, 24)
                churn_date = signup_date + timedelta(days=months_active * 30)
                if churn_date > self.end_date:
                    churn_date = None
                else:
                    churned = True
            
            # Generate monthly transactions
            current_date = signup_date
            consecutive_failures = 0
            
            while current_date <= self.end_date:
                if churned and current_date > churn_date:
                    break
                
                transaction_id += 1
                
                # Seasonality factor (higher revenue in Q4, lower in Q1)
                month = current_date.month
                seasonality = 1.0
                if month in [11, 12]:  # Holiday season
                    seasonality = 1.15
                elif month in [1, 2]:  # Post-holiday slump
                    seasonality = 0.90
                elif month in [6, 7, 8]:  # Summer
                    seasonality = 0.95
                
                # Transaction amount with some variation
                base_amount = monthly_fee * seasonality
                amount = round(base_amount * random.uniform(0.98, 1.02), 2)
                
                # Occasional upgrades/add-ons
                if random.random() < 0.05:
                    amount = round(amount * random.uniform(1.1, 1.5), 2)
                
                # Payment method
                payment_method = random.choices(
                    self.payment_methods,
                    weights=[0.50, 0.25, 0.15, 0.05, 0.05]
                )[0]
                
                # Transaction status
                # Failed payments can lead to churn
                if churned and (churn_date - current_date).days < 60:
                    # More failures near churn
                    fail_prob = 0.20
                else:
                    fail_prob = 0.03
                
                if random.random() < fail_prob:
                    status = 'Failed'
                    consecutive_failures += 1
                elif random.random() < 0.02:
                    status = 'Pending'
                elif random.random() < 0.01:
                    status = 'Refunded'
                    amount = -abs(amount)
                else:
                    status = 'Completed'
                    consecutive_failures = 0
                
                # Add some date variation (not exactly monthly)
                transaction_date = current_date + timedelta(days=random.randint(-2, 2))
                
                transactions.append({
                    'transaction_id': f"TXN_{transaction_id:08d}",
                    'customer_id': customer_id,
                    'transaction_date': transaction_date.strftime('%Y-%m-%d'),
                    'amount': amount,
                    'payment_method': payment_method,
                    'transaction_status': status
                })
                
                # Move to next month
                current_date = current_date + timedelta(days=30)
        
        return pd.DataFrame(transactions)
    
    def generate_all(self, output_dir, num_customers=2500):
        """Generate all datasets and save to output directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate customers
        customers_df = self.generate_customers(num_customers)
        customers_path = os.path.join(output_dir, 'customers.csv')
        customers_df.to_csv(customers_path, index=False)
        print(f"Saved {len(customers_df)} customers to {customers_path}")
        
        # Generate transactions
        transactions_df = self.generate_transactions(customers_df)
        transactions_path = os.path.join(output_dir, 'transactions.csv')
        transactions_df.to_csv(transactions_path, index=False)
        print(f"Saved {len(transactions_df)} transactions to {transactions_path}")
        
        return customers_df, transactions_df


def main():
    """Main entry point for data generation"""
    generator = FinancialDataGenerator(
        start_date='2021-01-01',
        end_date='2024-12-31'
    )
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    customers_df, transactions_df = generator.generate_all(output_dir, num_customers=2500)
    
    print("\n=== Data Generation Summary ===")
    print(f"Customers: {len(customers_df)}")
    print(f"Transactions: {len(transactions_df)}")
    print(f"Date range: {transactions_df['transaction_date'].min()} to {transactions_df['transaction_date'].max()}")
    print(f"Total revenue: ${transactions_df[transactions_df['amount'] > 0]['amount'].sum():,.2f}")
    
    return customers_df, transactions_df


if __name__ == "__main__":
    main()
