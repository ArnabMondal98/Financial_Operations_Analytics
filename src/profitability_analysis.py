"""
Customer Profitability Analysis Module
Comprehensive profitability calculations, segmentation, and reporting
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns


class ProfitabilityAnalyzer:
    """Customer profitability analysis and segmentation"""
    
    def __init__(self, data_dir, output_dir, visuals_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.visuals_dir = visuals_dir
        
        self.customers_df = None
        self.transactions_df = None
        self.profitability_df = None
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(visuals_dir, exist_ok=True)
        
        # Cost structure configuration
        self.cost_config = {
            'acquisition_costs': {
                'Basic': 50,
                'Professional': 100,
                'Enterprise': 200,
                'Premium': 300
            },
            'servicing_cost_pct': 0.15,  # 15% of revenue
            'transaction_fee': 0.50,     # $0.50 per transaction
            'support_cost_per_month': {
                'Basic': 5,
                'Professional': 15,
                'Enterprise': 40,
                'Premium': 75
            },
            'infrastructure_cost_pct': 0.05,  # 5% of revenue
            'payment_processing_pct': 0.029   # 2.9% payment processing
        }
    
    def load_data(self):
        """Load processed data"""
        print("Loading data...")
        
        self.customers_df = pd.read_csv(
            os.path.join(self.data_dir, 'processed_customers.csv')
        )
        self.transactions_df = pd.read_csv(
            os.path.join(self.data_dir, 'processed_transactions.csv')
        )
        
        # Convert dates
        self.customers_df['signup_date'] = pd.to_datetime(self.customers_df['signup_date'])
        self.transactions_df['transaction_date'] = pd.to_datetime(self.transactions_df['transaction_date'])
        
        print(f"Loaded {len(self.customers_df)} customers and {len(self.transactions_df)} transactions")
        
        return self.customers_df, self.transactions_df
    
    def calculate_total_revenue(self):
        """Step 1: Calculate total revenue per customer"""
        print("\n=== Step 1: Calculating Total Revenue Per Customer ===")
        
        df = self.customers_df.copy()
        
        # Revenue is already in processed_customers, but let's verify from transactions
        revenue_check = self.transactions_df[
            self.transactions_df['is_successful'] == 1
        ].groupby('customer_id')['amount'].sum().reset_index()
        revenue_check.columns = ['customer_id', 'verified_revenue']
        
        df = df.merge(revenue_check, on='customer_id', how='left')
        df['verified_revenue'] = df['verified_revenue'].fillna(0)
        
        # Use the verified revenue
        df['total_revenue'] = df['verified_revenue']
        
        print(f"Total Revenue: ${df['total_revenue'].sum():,.2f}")
        print(f"Avg Revenue per Customer: ${df['total_revenue'].mean():,.2f}")
        print(f"Median Revenue: ${df['total_revenue'].median():,.2f}")
        
        # Revenue distribution
        print("\nRevenue Percentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            val = df['total_revenue'].quantile(p/100)
            print(f"  P{p}: ${val:,.2f}")
        
        self.profitability_df = df
        return df
    
    def estimate_customer_costs(self):
        """Step 2: Estimate customer servicing and operational costs"""
        print("\n=== Step 2: Estimating Customer Costs ===")
        
        df = self.profitability_df
        
        # 1. Customer Acquisition Cost (CAC)
        df['acquisition_cost'] = df['subscription_type'].map(
            self.cost_config['acquisition_costs']
        )
        
        # 2. Servicing Cost (support, account management)
        df['servicing_cost'] = df['total_revenue'] * self.cost_config['servicing_cost_pct']
        
        # 3. Transaction Processing Costs
        df['transaction_cost'] = df['total_transactions'] * self.cost_config['transaction_fee']
        
        # 4. Support Cost (monthly based on tier)
        months_active = df['customer_tenure_days'] / 30
        df['support_cost'] = df['subscription_type'].map(
            self.cost_config['support_cost_per_month']
        ) * months_active
        
        # 5. Infrastructure Cost
        df['infrastructure_cost'] = df['total_revenue'] * self.cost_config['infrastructure_cost_pct']
        
        # 6. Payment Processing Fees
        df['payment_processing_cost'] = df['total_revenue'] * self.cost_config['payment_processing_pct']
        
        # Total Cost
        df['total_cost'] = (
            df['acquisition_cost'] +
            df['servicing_cost'] +
            df['transaction_cost'] +
            df['support_cost'] +
            df['infrastructure_cost'] +
            df['payment_processing_cost']
        )
        
        print(f"Total Costs: ${df['total_cost'].sum():,.2f}")
        print(f"Avg Cost per Customer: ${df['total_cost'].mean():,.2f}")
        
        print("\nCost Breakdown:")
        cost_cols = ['acquisition_cost', 'servicing_cost', 'transaction_cost', 
                     'support_cost', 'infrastructure_cost', 'payment_processing_cost']
        for col in cost_cols:
            total = df[col].sum()
            pct = total / df['total_cost'].sum() * 100
            print(f"  {col.replace('_', ' ').title()}: ${total:,.2f} ({pct:.1f}%)")
        
        self.profitability_df = df
        return df
    
    def compute_customer_profit(self):
        """Step 3: Compute profit per customer"""
        print("\n=== Step 3: Computing Profit Per Customer ===")
        
        df = self.profitability_df
        
        # Gross Profit
        df['gross_profit'] = df['total_revenue'] - df['total_cost']
        
        # Profit Margin (%)
        df['profit_margin'] = np.where(
            df['total_revenue'] > 0,
            (df['gross_profit'] / df['total_revenue']) * 100,
            -100  # Assign -100% margin for zero revenue
        )
        
        # Monthly Profit
        months_active = np.maximum(df['customer_tenure_days'] / 30, 1)
        df['monthly_profit'] = df['gross_profit'] / months_active
        
        # ROI (Return on Investment)
        df['customer_roi'] = np.where(
            df['total_cost'] > 0,
            (df['gross_profit'] / df['total_cost']) * 100,
            0
        )
        
        # Profit per transaction
        df['profit_per_transaction'] = np.where(
            df['total_transactions'] > 0,
            df['gross_profit'] / df['total_transactions'],
            0
        )
        
        print(f"Total Gross Profit: ${df['gross_profit'].sum():,.2f}")
        print(f"Overall Profit Margin: {(df['gross_profit'].sum() / df['total_revenue'].sum()) * 100:.1f}%")
        print(f"Avg Profit per Customer: ${df['gross_profit'].mean():,.2f}")
        print(f"Avg Monthly Profit: ${df['monthly_profit'].mean():,.2f}")
        
        # Profit statistics
        profitable = df[df['gross_profit'] > 0]
        unprofitable = df[df['gross_profit'] <= 0]
        
        print(f"\nProfit Distribution:")
        print(f"  Profitable Customers: {len(profitable)} ({len(profitable)/len(df)*100:.1f}%)")
        print(f"  Unprofitable Customers: {len(unprofitable)} ({len(unprofitable)/len(df)*100:.1f}%)")
        
        self.profitability_df = df
        return df
    
    def segment_profitability_tiers(self):
        """Step 4: Segment customers into profitability tiers"""
        print("\n=== Step 4: Segmenting Profitability Tiers ===")
        
        df = self.profitability_df
        
        # Define tier boundaries based on profit margin
        def assign_tier(row):
            margin = row['profit_margin']
            profit = row['gross_profit']
            
            if profit < 0:
                return 'Loss-Making'
            elif margin < 20:
                return 'Low Profit'
            elif margin < 50:
                return 'Medium Profit'
            else:
                return 'High Profit'
        
        df['profitability_tier'] = df.apply(assign_tier, axis=1)
        
        # Also create numeric tier for sorting
        tier_order = {'Loss-Making': 0, 'Low Profit': 1, 'Medium Profit': 2, 'High Profit': 3}
        df['tier_rank'] = df['profitability_tier'].map(tier_order)
        
        # Tier summary
        tier_summary = df.groupby('profitability_tier').agg({
            'customer_id': 'count',
            'total_revenue': 'sum',
            'total_cost': 'sum',
            'gross_profit': 'sum',
            'profit_margin': 'mean',
            'monthly_profit': 'mean'
        }).reset_index()
        
        tier_summary.columns = [
            'tier', 'customer_count', 'total_revenue', 'total_cost',
            'total_profit', 'avg_margin', 'avg_monthly_profit'
        ]
        
        # Calculate tier percentage
        tier_summary['customer_pct'] = tier_summary['customer_count'] / len(df) * 100
        tier_summary['revenue_pct'] = tier_summary['total_revenue'] / df['total_revenue'].sum() * 100
        tier_summary['profit_pct'] = tier_summary['total_profit'] / df['gross_profit'].sum() * 100
        
        # Sort by tier order
        tier_summary['sort_order'] = tier_summary['tier'].map(tier_order)
        tier_summary = tier_summary.sort_values('sort_order').drop('sort_order', axis=1)
        
        print("\nProfitability Tier Summary:")
        print("-" * 100)
        print(f"{'Tier':<15} {'Customers':>10} {'Revenue':>15} {'Profit':>15} {'Avg Margin':>12} {'% of Profit':>12}")
        print("-" * 100)
        
        for _, row in tier_summary.iterrows():
            print(f"{row['tier']:<15} {row['customer_count']:>10,} ${row['total_revenue']:>13,.0f} ${row['total_profit']:>13,.0f} {row['avg_margin']:>11.1f}% {row['profit_pct']:>11.1f}%")
        
        # Save tier summary
        tier_summary.to_csv(
            os.path.join(self.output_dir, 'profitability_tier_summary.csv'),
            index=False
        )
        
        self.profitability_df = df
        self.tier_summary = tier_summary
        return df, tier_summary
    
    def analyze_revenue_profit_contribution(self):
        """Step 5: Identify revenue vs profit contribution by segment"""
        print("\n=== Step 5: Revenue vs Profit Contribution Analysis ===")
        
        df = self.profitability_df
        
        # By Profitability Tier
        tier_contribution = df.groupby('profitability_tier').agg({
            'customer_id': 'count',
            'total_revenue': 'sum',
            'gross_profit': 'sum'
        }).reset_index()
        
        total_revenue = df['total_revenue'].sum()
        total_profit = df['gross_profit'].sum()
        
        tier_contribution['revenue_contribution_pct'] = (
            tier_contribution['total_revenue'] / total_revenue * 100
        )
        tier_contribution['profit_contribution_pct'] = (
            tier_contribution['gross_profit'] / total_profit * 100
        )
        
        # By Subscription Type
        sub_contribution = df.groupby('subscription_type').agg({
            'customer_id': 'count',
            'total_revenue': 'sum',
            'gross_profit': 'sum',
            'profit_margin': 'mean'
        }).reset_index()
        
        sub_contribution['revenue_contribution_pct'] = (
            sub_contribution['total_revenue'] / total_revenue * 100
        )
        sub_contribution['profit_contribution_pct'] = (
            sub_contribution['gross_profit'] / total_profit * 100
        )
        sub_contribution = sub_contribution.sort_values('gross_profit', ascending=False)
        
        # By Country
        country_contribution = df.groupby('country').agg({
            'customer_id': 'count',
            'total_revenue': 'sum',
            'gross_profit': 'sum',
            'profit_margin': 'mean'
        }).reset_index()
        
        country_contribution['revenue_contribution_pct'] = (
            country_contribution['total_revenue'] / total_revenue * 100
        )
        country_contribution['profit_contribution_pct'] = (
            country_contribution['gross_profit'] / total_profit * 100
        )
        country_contribution = country_contribution.sort_values('gross_profit', ascending=False)
        
        # By Industry
        industry_contribution = df.groupby('industry').agg({
            'customer_id': 'count',
            'total_revenue': 'sum',
            'gross_profit': 'sum',
            'profit_margin': 'mean'
        }).reset_index()
        
        industry_contribution['revenue_contribution_pct'] = (
            industry_contribution['total_revenue'] / total_revenue * 100
        )
        industry_contribution['profit_contribution_pct'] = (
            industry_contribution['gross_profit'] / total_profit * 100
        )
        industry_contribution = industry_contribution.sort_values('gross_profit', ascending=False)
        
        # Save all contribution analyses
        sub_contribution.to_csv(
            os.path.join(self.output_dir, 'contribution_by_subscription.csv'),
            index=False
        )
        country_contribution.to_csv(
            os.path.join(self.output_dir, 'contribution_by_country.csv'),
            index=False
        )
        industry_contribution.to_csv(
            os.path.join(self.output_dir, 'contribution_by_industry.csv'),
            index=False
        )
        
        print("\nRevenue vs Profit Contribution by Subscription:")
        print("-" * 80)
        for _, row in sub_contribution.iterrows():
            print(f"  {row['subscription_type']:<15} Revenue: {row['revenue_contribution_pct']:>5.1f}%  |  Profit: {row['profit_contribution_pct']:>5.1f}%  |  Margin: {row['profit_margin']:>5.1f}%")
        
        print("\nTop 5 Countries by Profit:")
        for _, row in country_contribution.head(5).iterrows():
            print(f"  {row['country']:<15} Profit: ${row['gross_profit']:>12,.0f}  ({row['profit_contribution_pct']:.1f}%)")
        
        self.contribution_data = {
            'tier': tier_contribution,
            'subscription': sub_contribution,
            'country': country_contribution,
            'industry': industry_contribution
        }
        
        return self.contribution_data
    
    def generate_profitability_report(self):
        """Step 6: Generate customer profitability report"""
        print("\n=== Step 6: Generating Profitability Report ===")
        
        df = self.profitability_df
        report_date = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate key metrics
        total_revenue = df['total_revenue'].sum()
        total_cost = df['total_cost'].sum()
        total_profit = df['gross_profit'].sum()
        overall_margin = (total_profit / total_revenue) * 100
        
        profitable_customers = len(df[df['gross_profit'] > 0])
        loss_customers = len(df[df['gross_profit'] <= 0])
        
        report = f"""
================================================================================
                    CUSTOMER PROFITABILITY ANALYSIS REPORT
                              {report_date}
================================================================================

EXECUTIVE SUMMARY
-----------------
Total Customers Analyzed: {len(df):,}
Analysis Period: Full Customer Lifetime

FINANCIAL OVERVIEW
------------------
Total Revenue:        ${total_revenue:>15,.2f}
Total Costs:          ${total_cost:>15,.2f}
Gross Profit:         ${total_profit:>15,.2f}
Overall Margin:       {overall_margin:>15.1f}%

CUSTOMER PROFITABILITY
----------------------
Profitable Customers: {profitable_customers:,} ({profitable_customers/len(df)*100:.1f}%)
Loss-Making Customers: {loss_customers:,} ({loss_customers/len(df)*100:.1f}%)

Average Metrics:
  • Avg Revenue per Customer:   ${df['total_revenue'].mean():,.2f}
  • Avg Cost per Customer:      ${df['total_cost'].mean():,.2f}
  • Avg Profit per Customer:    ${df['gross_profit'].mean():,.2f}
  • Avg Profit Margin:          {df['profit_margin'].mean():.1f}%
  • Avg Monthly Profit:         ${df['monthly_profit'].mean():,.2f}

PROFITABILITY TIERS
-------------------
"""
        
        for _, row in self.tier_summary.iterrows():
            report += f"""
{row['tier'].upper()}:
  Customers: {row['customer_count']:,} ({row['customer_pct']:.1f}% of total)
  Revenue: ${row['total_revenue']:,.2f} ({row['revenue_pct']:.1f}% of total)
  Profit: ${row['total_profit']:,.2f} ({row['profit_pct']:.1f}% of total)
  Avg Margin: {row['avg_margin']:.1f}%
"""
        
        # Top profitable customers
        top_profitable = df.nlargest(10, 'gross_profit')
        report += """
TOP 10 MOST PROFITABLE CUSTOMERS
--------------------------------
"""
        for i, (_, row) in enumerate(top_profitable.iterrows(), 1):
            report += f"{i:>2}. {row['customer_id']} ({row['subscription_type']}): ${row['gross_profit']:,.2f} profit ({row['profit_margin']:.1f}% margin)\n"
        
        # Loss-making customers
        loss_making = df[df['gross_profit'] < 0].nsmallest(10, 'gross_profit')
        if len(loss_making) > 0:
            report += """
TOP 10 LOSS-MAKING CUSTOMERS (Require Attention)
------------------------------------------------
"""
            for i, (_, row) in enumerate(loss_making.iterrows(), 1):
                report += f"{i:>2}. {row['customer_id']} ({row['subscription_type']}): ${row['gross_profit']:,.2f} loss\n"
        
        report += """
RECOMMENDATIONS
---------------
1. HIGH PROFIT TIER: Maintain engagement, upsell opportunities
2. MEDIUM PROFIT TIER: Optimize cost structure, increase usage
3. LOW PROFIT TIER: Review pricing, reduce servicing costs
4. LOSS-MAKING: Evaluate retention value, consider price adjustment

================================================================================
                              END OF REPORT
================================================================================
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, 'profitability_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {report_path}")
        
        return report
    
    def export_dashboard_data(self):
        """Step 7: Export dashboard-ready profitability dataset"""
        print("\n=== Step 7: Exporting Dashboard-Ready Data ===")
        
        df = self.profitability_df
        
        # Select key columns for dashboard
        dashboard_cols = [
            'customer_id', 'country', 'industry', 'subscription_type',
            'total_revenue', 'total_cost', 'gross_profit', 'profit_margin',
            'monthly_profit', 'customer_roi', 'profitability_tier',
            'customer_tenure_days', 'total_transactions'
        ]
        
        dashboard_df = df[dashboard_cols].copy()
        
        # Add formatted columns for dashboard display
        dashboard_df['revenue_formatted'] = dashboard_df['total_revenue'].apply(lambda x: f"${x:,.2f}")
        dashboard_df['profit_formatted'] = dashboard_df['gross_profit'].apply(lambda x: f"${x:,.2f}")
        dashboard_df['margin_formatted'] = dashboard_df['profit_margin'].apply(lambda x: f"{x:.1f}%")
        
        # Add date dimensions
        dashboard_df['tenure_months'] = (dashboard_df['customer_tenure_days'] / 30).round(1)
        
        # Export main dashboard file
        dashboard_path = os.path.join(self.output_dir, 'profitability_dashboard.csv')
        dashboard_df.to_csv(dashboard_path, index=False)
        print(f"  Dashboard data: {dashboard_path}")
        
        # Export aggregated summaries for dashboard cards
        summary_metrics = {
            'metric': [
                'total_revenue', 'total_cost', 'gross_profit', 'overall_margin',
                'total_customers', 'profitable_customers', 'loss_customers',
                'avg_profit_per_customer', 'avg_monthly_profit', 'avg_roi'
            ],
            'value': [
                df['total_revenue'].sum(),
                df['total_cost'].sum(),
                df['gross_profit'].sum(),
                (df['gross_profit'].sum() / df['total_revenue'].sum()) * 100,
                len(df),
                len(df[df['gross_profit'] > 0]),
                len(df[df['gross_profit'] <= 0]),
                df['gross_profit'].mean(),
                df['monthly_profit'].mean(),
                df['customer_roi'].mean()
            ]
        }
        
        metrics_df = pd.DataFrame(summary_metrics)
        metrics_path = os.path.join(self.output_dir, 'profitability_kpis.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"  KPI metrics: {metrics_path}")
        
        # Export tier breakdown for charts
        tier_path = os.path.join(self.output_dir, 'profitability_tier_breakdown.csv')
        self.tier_summary.to_csv(tier_path, index=False)
        print(f"  Tier breakdown: {tier_path}")
        
        return dashboard_df
    
    def create_visualizations(self):
        """Step 8: Produce profitability visualization charts"""
        print("\n=== Step 8: Generating Profitability Visualizations ===")
        
        df = self.profitability_df
        
        # 1. Profitability Distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Profit margin histogram
        ax1 = axes[0, 0]
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in df['profit_margin']]
        ax1.hist(df['profit_margin'], bins=40, color='steelblue', edgecolor='white', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax1.axvline(x=df['profit_margin'].mean(), color='green', linestyle='-', linewidth=2, label=f"Mean: {df['profit_margin'].mean():.1f}%")
        ax1.set_xlabel('Profit Margin (%)', fontsize=10)
        ax1.set_ylabel('Customer Count', fontsize=10)
        ax1.set_title('Distribution of Profit Margin', fontsize=12, fontweight='bold')
        ax1.legend()
        
        # Tier pie chart
        ax2 = axes[0, 1]
        tier_counts = df['profitability_tier'].value_counts()
        colors_pie = {'High Profit': '#2ecc71', 'Medium Profit': '#3498db', 
                      'Low Profit': '#f39c12', 'Loss-Making': '#e74c3c'}
        pie_colors = [colors_pie.get(t, '#95a5a6') for t in tier_counts.index]
        ax2.pie(tier_counts, labels=tier_counts.index, autopct='%1.1f%%',
                colors=pie_colors, startangle=90, explode=[0.05]*len(tier_counts))
        ax2.set_title('Customers by Profitability Tier', fontsize=12, fontweight='bold')
        
        # Revenue vs Profit scatter
        ax3 = axes[1, 0]
        scatter = ax3.scatter(df['total_revenue'], df['gross_profit'],
                              c=df['profit_margin'], cmap='RdYlGn', 
                              alpha=0.6, s=30, vmin=-50, vmax=100)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Total Revenue ($)', fontsize=10)
        ax3.set_ylabel('Gross Profit ($)', fontsize=10)
        ax3.set_title('Revenue vs Profit by Customer', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax3, label='Profit Margin (%)')
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Profit by subscription type
        ax4 = axes[1, 1]
        sub_profit = df.groupby('subscription_type')['gross_profit'].sum().sort_values()
        colors_bar = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        ax4.barh(sub_profit.index, sub_profit.values, color=colors_bar)
        ax4.set_xlabel('Total Profit ($)', fontsize=10)
        ax4.set_title('Profit by Subscription Type', fontsize=12, fontweight='bold')
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visuals_dir, 'profitability_overview.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: profitability_overview.png")
        
        # 2. Revenue vs Profit Contribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # By tier - stacked bar
        ax1 = axes[0]
        tier_data = self.tier_summary[['tier', 'revenue_pct', 'profit_pct']].set_index('tier')
        x = range(len(tier_data))
        width = 0.35
        ax1.bar([i - width/2 for i in x], tier_data['revenue_pct'], width, 
                label='Revenue %', color='#3498db')
        ax1.bar([i + width/2 for i in x], tier_data['profit_pct'], width, 
                label='Profit %', color='#2ecc71')
        ax1.set_xticks(x)
        ax1.set_xticklabels(tier_data.index, rotation=15)
        ax1.set_ylabel('Contribution (%)', fontsize=10)
        ax1.set_title('Revenue vs Profit Contribution by Tier', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # By subscription
        ax2 = axes[1]
        sub_data = self.contribution_data['subscription'][['subscription_type', 'revenue_contribution_pct', 'profit_contribution_pct']]
        sub_data = sub_data.set_index('subscription_type')
        x = range(len(sub_data))
        ax2.bar([i - width/2 for i in x], sub_data['revenue_contribution_pct'], width, 
                label='Revenue %', color='#3498db')
        ax2.bar([i + width/2 for i in x], sub_data['profit_contribution_pct'], width, 
                label='Profit %', color='#2ecc71')
        ax2.set_xticks(x)
        ax2.set_xticklabels(sub_data.index)
        ax2.set_ylabel('Contribution (%)', fontsize=10)
        ax2.set_title('Revenue vs Profit by Subscription', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visuals_dir, 'revenue_profit_contribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: revenue_profit_contribution.png")
        
        # 3. Geographic Profitability
        fig, ax = plt.subplots(figsize=(12, 6))
        
        country_data = self.contribution_data['country'].head(10)
        y_pos = range(len(country_data))
        
        bars = ax.barh(y_pos, country_data['gross_profit'], 
                       color=sns.color_palette('Blues_r', len(country_data)))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(country_data['country'])
        ax.set_xlabel('Gross Profit ($)', fontsize=10)
        ax.set_title('Top 10 Countries by Profitability', fontsize=12, fontweight='bold')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        ax.invert_yaxis()
        
        # Add margin labels
        for i, (idx, row) in enumerate(country_data.iterrows()):
            ax.text(row['gross_profit'] + 50000, i, f"{row['profit_margin']:.0f}%",
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visuals_dir, 'profitability_by_country.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: profitability_by_country.png")
        
        # 4. Cost Structure Analysis
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cost breakdown pie
        cost_breakdown = {
            'Servicing': df['servicing_cost'].sum(),
            'Support': df['support_cost'].sum(),
            'Infrastructure': df['infrastructure_cost'].sum(),
            'Payment Processing': df['payment_processing_cost'].sum(),
            'Acquisition': df['acquisition_cost'].sum(),
            'Transaction Fees': df['transaction_cost'].sum()
        }
        ax1 = axes[0]
        ax1.pie(cost_breakdown.values(), labels=cost_breakdown.keys(), 
                autopct='%1.1f%%', startangle=90,
                colors=sns.color_palette('Set2', len(cost_breakdown)))
        ax1.set_title('Cost Structure Breakdown', fontsize=12, fontweight='bold')
        
        # Cost per tier
        ax2 = axes[1]
        tier_costs = df.groupby('profitability_tier')[['total_revenue', 'total_cost']].mean()
        tier_order = ['Loss-Making', 'Low Profit', 'Medium Profit', 'High Profit']
        tier_costs = tier_costs.reindex(tier_order)
        
        x = range(len(tier_costs))
        ax2.bar([i - 0.2 for i in x], tier_costs['total_revenue'], 0.4, 
                label='Avg Revenue', color='#3498db')
        ax2.bar([i + 0.2 for i in x], tier_costs['total_cost'], 0.4, 
                label='Avg Cost', color='#e74c3c')
        ax2.set_xticks(x)
        ax2.set_xticklabels(tier_costs.index, rotation=15)
        ax2.set_ylabel('Amount ($)', fontsize=10)
        ax2.set_title('Avg Revenue vs Cost by Tier', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visuals_dir, 'cost_structure_analysis.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: cost_structure_analysis.png")
        
        print("\nAll visualizations generated successfully!")
        
        return True
    
    def run_profitability_analysis(self):
        """Run complete profitability analysis pipeline"""
        print("\n" + "="*60)
        print("CUSTOMER PROFITABILITY ANALYSIS PIPELINE")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Step 1: Calculate total revenue
        self.calculate_total_revenue()
        
        # Step 2: Estimate costs
        self.estimate_customer_costs()
        
        # Step 3: Compute profit
        self.compute_customer_profit()
        
        # Step 4: Segment into tiers
        self.segment_profitability_tiers()
        
        # Step 5: Analyze contributions
        self.analyze_revenue_profit_contribution()
        
        # Step 6: Generate report
        self.generate_profitability_report()
        
        # Step 7: Export dashboard data
        self.export_dashboard_data()
        
        # Step 8: Create visualizations
        self.create_visualizations()
        
        # Save full profitability data
        full_path = os.path.join(self.output_dir, 'customer_profitability_full.csv')
        self.profitability_df.to_csv(full_path, index=False)
        print(f"\nFull profitability data saved to {full_path}")
        
        print("\n" + "="*60)
        print("PROFITABILITY ANALYSIS COMPLETE")
        print("="*60)
        
        return {
            'profitability_df': self.profitability_df,
            'tier_summary': self.tier_summary,
            'contribution_data': self.contribution_data
        }


def main():
    """Main entry point"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'outputs')
    output_dir = os.path.join(base_dir, 'outputs', 'profitability')
    visuals_dir = os.path.join(base_dir, 'visuals')
    
    analyzer = ProfitabilityAnalyzer(data_dir, output_dir, visuals_dir)
    results = analyzer.run_profitability_analysis()
    
    return results


if __name__ == "__main__":
    main()
