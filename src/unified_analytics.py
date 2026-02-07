"""
Unified Analytics Dashboard Module
Combines all analytics outputs into comprehensive executive dashboards
"""

from isort import file
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib.gridspec import GridSpec


class UnifiedAnalyticsDashboard:
    """Create unified analytics outputs for executive dashboard and reporting"""
    
    def __init__(self, base_dir, output_dir, visuals_dir):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.visuals_dir = visuals_dir
        self.dashboard_dir = os.path.join(output_dir, 'dashboard')
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.dashboard_dir, exist_ok=True)
        os.makedirs(visuals_dir, exist_ok=True)
        
        self.master_df = None
        self.monthly_summary = None
        self.kpis = {}
    
    def load_all_outputs(self):
        """Step 1: Load and combine outputs from all modules"""
        print("\n=== Step 1: Loading All Module Outputs ===")
        
        outputs_dir = os.path.join(self.base_dir, 'outputs')
        
        # Load base customer data
        print("  Loading customer data...")
        self.customers = pd.read_csv(
            os.path.join(outputs_dir, 'processed_customers.csv')
        )
        self.customers['signup_date'] = pd.to_datetime(self.customers['signup_date'])
        
        # Load transactions
        print("  Loading transaction data...")
        self.transactions = pd.read_csv(
            os.path.join(outputs_dir, 'processed_transactions.csv')
        )
        self.transactions['transaction_date'] = pd.to_datetime(self.transactions['transaction_date'])
        
        # Load churn predictions
        print("  Loading churn predictions...")
        churn_path = os.path.join(outputs_dir, 'churn', 'churn_predictions.csv')
        if os.path.exists(churn_path):
            self.churn_data = pd.read_csv(churn_path)
        else:
            self.churn_data = None
            print("    Warning: Churn data not found")
        
        # Load retention recommendations
        print("  Loading retention recommendations...")
        retention_path = os.path.join(outputs_dir, 'retention', 'retention_recommendations_full.csv')
        if os.path.exists(retention_path):
            self.retention_data = pd.read_csv(retention_path)
        else:
            self.retention_data = None
            print("    Warning: Retention data not found")
        
        # Load profitability data
        print("  Loading profitability data...")
        profit_path = os.path.join(outputs_dir, 'profitability', 'customer_profitability_full.csv')
        if os.path.exists(profit_path):
            self.profitability_data = pd.read_csv(profit_path)
        else:
            self.profitability_data = None
            print("    Warning: Profitability data not found")
        
        # Load segmentation data
        print("  Loading segmentation data...")
        segment_path = os.path.join(outputs_dir, 'segmentation', 'customer_segmentation_full.csv')
        if os.path.exists(segment_path):
            self.segmentation_data = pd.read_csv(segment_path)
        else:
            self.segmentation_data = None
            print("    Warning: Segmentation data not found")
        
        # Load forecasts
        print("  Loading forecast data...")
        forecast_path = os.path.join(outputs_dir, 'forecasts', 'combined_forecast.csv')
        if os.path.exists(forecast_path):
            self.forecast_data = pd.read_csv(forecast_path)
            self.forecast_data['date'] = pd.to_datetime(self.forecast_data['date'])
        else:
            self.forecast_data = None
            print("    Warning: Forecast data not found")
        
        # Load monthly revenue
        print("  Loading monthly revenue...")
        monthly_rev_path = os.path.join(outputs_dir, 'monthly_revenue.csv')
        if os.path.exists(monthly_rev_path):
            self.monthly_revenue = pd.read_csv(monthly_rev_path)
            self.monthly_revenue['date'] = pd.to_datetime(self.monthly_revenue['date'])
        
        print(f"\nLoaded data for {len(self.customers)} customers")
        
        return True
    
    def create_master_customer_dataset(self):
        """Step 2: Create master customer analytics dataset"""
        print("\n=== Step 2: Creating Master Customer Dataset ===")
        
        # Start with base customer data
        master = self.customers[['customer_id', 'signup_date', 'country', 'industry', 
                                  'subscription_type', 'monthly_fee', 'total_revenue',
                                  'customer_tenure_days', 'is_churned']].copy()
        
        # Add churn probability and risk level
        if self.churn_data is not None:
            churn_cols = ['customer_id', 'churn_probability', 'churn_risk_level']
            master = master.merge(
                self.churn_data[churn_cols],
                on='customer_id',
                how='left'
            )
        else:
            master['churn_probability'] = 0
            master['churn_risk_level'] = 'Unknown'
        
        # Add retention action
        if self.retention_data is not None:
            retention_cols = ['customer_id', 'primary_action', 'urgency', 
                             'recommended_discount_pct', 'specific_recommendation']
            master = master.merge(
                self.retention_data[retention_cols],
                on='customer_id',
                how='left'
            )
            master.rename(columns={'primary_action': 'retention_action'}, inplace=True)
        else:
            master['retention_action'] = 'Not Assigned'
            master['urgency'] = 'Unknown'
        
        # Add profitability metrics
        if self.profitability_data is not None:
            profit_cols = ['customer_id', 'gross_profit', 'profit_margin', 
                          'profitability_tier', 'total_cost']
            master = master.merge(
                self.profitability_data[profit_cols],
                on='customer_id',
                how='left'
            )
        else:
            master['gross_profit'] = master['total_revenue'] * 0.6
            master['profit_margin'] = 60
            master['profitability_tier'] = 'Unknown'
        
        # Add segmentation data
        if self.segmentation_data is not None:
            segment_cols = ['customer_id', 'rfm_segment', 'behavioral_segment',
                           'R_score', 'F_score', 'M_score', 'RFM_total']
            master = master.merge(
                self.segmentation_data[segment_cols],
                on='customer_id',
                how='left'
            )
            master.rename(columns={'rfm_segment': 'customer_segment'}, inplace=True)
        else:
            master['customer_segment'] = 'Unknown'
            master['behavioral_segment'] = 'Unknown'
        
        # Add cohort information
        master['signup_cohort'] = master['signup_date'].dt.to_period('M').astype(str)
        master['signup_year'] = master['signup_date'].dt.year
        master['signup_quarter'] = master['signup_date'].dt.quarter
        
        # Calculate customer lifetime months
        reference_date = self.transactions['transaction_date'].max()
        master['lifetime_months'] = ((reference_date - master['signup_date']).dt.days / 30).round(1)
        
        # Add customer value tier
        master['customer_value_tier'] = pd.qcut(
            master['total_revenue'].rank(method='first'),
            q=4,
            labels=['Bronze', 'Silver', 'Gold', 'Platinum']
        )
        
        # Fill any remaining NaN values
        master = master.fillna({
            'churn_probability': 0,
            'churn_risk_level': 'Unknown',
            'retention_action': 'Not Assigned',
            'profitability_tier': 'Unknown',
            'customer_segment': 'Unknown'
        })
        
        self.master_df = master
        
        # Save master dataset
        master_path = os.path.join(self.output_dir, 'master_customer_analytics.csv')
        master.to_csv(master_path, index=False)
        print(f"  Created master dataset with {len(master)} customers and {len(master.columns)} attributes")
        print(f"  Saved to: {master_path}")
        
        return master
    
    def create_monthly_business_summary(self):
        """Step 3: Create monthly business summary dataset"""
        print("\n=== Step 3: Creating Monthly Business Summary ===")
        
        # Get successful transactions
        successful_txns = self.transactions[
            self.transactions['transaction_status'] == 'Completed'
        ].copy()
        successful_txns['month'] = successful_txns['transaction_date'].dt.to_period('M')
        
        # Monthly revenue and transaction metrics
        monthly = successful_txns.groupby('month').agg({
            'amount': 'sum',
            'transaction_id': 'count',
            'customer_id': 'nunique'
        }).reset_index()
        monthly.columns = ['month', 'revenue', 'transactions', 'active_customers']
        monthly['month'] = monthly['month'].astype(str)
        
        # Calculate month-over-month growth
        monthly['revenue_mom'] = monthly['revenue'].pct_change() * 100
        monthly['customer_mom'] = monthly['active_customers'].pct_change() * 100
        
        # Add customer acquisition by month
        customer_acquisition = self.customers.copy()
        customer_acquisition['signup_month'] = customer_acquisition['signup_date'].dt.to_period('M').astype(str)
        acquisitions = customer_acquisition.groupby('signup_month').size().reset_index(name='new_customers')
        monthly = monthly.merge(acquisitions, left_on='month', right_on='signup_month', how='left')
        monthly.drop('signup_month', axis=1, inplace=True)
        monthly['new_customers'] = monthly['new_customers'].fillna(0).astype(int)
        
        # Add churn metrics by month
        if self.master_df is not None:
            churn_by_cohort = self.master_df.groupby('signup_cohort').agg({
                'is_churned': ['sum', 'count'],
                'churn_probability': 'mean'
            }).reset_index()
            churn_by_cohort.columns = ['month', 'churned_customers', 'total_customers', 'avg_churn_prob']
            churn_by_cohort['churn_rate'] = (churn_by_cohort['churned_customers'] / churn_by_cohort['total_customers'] * 100).round(2)
            
            monthly = monthly.merge(
                churn_by_cohort[['month', 'churned_customers', 'churn_rate', 'avg_churn_prob']],
                on='month',
                how='left'
            )
        
        # Add profitability metrics
        if self.profitability_data is not None:
            # Estimate monthly profit (simplified)
            avg_margin = self.profitability_data['profit_margin'].mean() / 100
            monthly['estimated_profit'] = monthly['revenue'] * avg_margin
            monthly['profit_margin_pct'] = avg_margin * 100
        
        # Add retention metrics
        if self.master_df is not None:
            high_risk = self.master_df[self.master_df['churn_risk_level'] == 'High']
            monthly['total_high_risk_customers'] = len(high_risk)
            monthly['revenue_at_risk'] = high_risk['total_revenue'].sum()
        
        # Calculate cumulative metrics
        monthly['cumulative_revenue'] = monthly['revenue'].cumsum()
        monthly['cumulative_customers'] = monthly['new_customers'].cumsum()
        
        # Fill NaN values
        monthly = monthly.fillna(0)
        
        self.monthly_summary = monthly
        
        # Save monthly summary
        monthly_path = os.path.join(self.output_dir, 'monthly_business_summary.csv')
        monthly.to_csv(monthly_path, index=False)
        print(f"  Created monthly summary with {len(monthly)} months")
        print(f"  Saved to: {monthly_path}")
        
        return monthly
    
    def export_powerbi_datasets(self):
        """Step 4: Export Power BI dashboard-ready datasets"""
        print("\n=== Step 4: Exporting Power BI Datasets ===")
        
        # 1. Fact Table - Customer Metrics
        fact_customers = self.master_df[[
            'customer_id', 'signup_date', 'total_revenue', 'gross_profit',
            'churn_probability', 'customer_tenure_days', 'lifetime_months'
        ]].copy()
        fact_customers['date_key'] = fact_customers['signup_date'].dt.strftime('%Y%m%d').astype(int)
        
        fact_path = os.path.join(self.dashboard_dir, 'fact_customer_metrics.csv')
        fact_customers.to_csv(fact_path, index=False)
        print(f"  Fact table: {fact_path}")
        
        # 2. Dimension Table - Customer Attributes
        dim_customers = self.master_df[[
            'customer_id', 'country', 'industry', 'subscription_type',
            'customer_segment', 'behavioral_segment', 'profitability_tier',
            'churn_risk_level', 'customer_value_tier', 'signup_cohort'
        ]].copy()
        
        dim_path = os.path.join(self.dashboard_dir, 'dim_customer_attributes.csv')
        dim_customers.to_csv(dim_path, index=False)
        print(f"  Dimension table: {dim_path}")
        
        # 3. Dimension Table - Actions
        dim_actions = self.master_df[[
            'customer_id', 'retention_action', 'urgency'
        ]].copy()
        if 'recommended_discount_pct' in self.master_df.columns:
            dim_actions['recommended_discount'] = self.master_df['recommended_discount_pct']
        if 'specific_recommendation' in self.master_df.columns:
            dim_actions['specific_action'] = self.master_df['specific_recommendation']
        
        actions_path = os.path.join(self.dashboard_dir, 'dim_retention_actions.csv')
        dim_actions.to_csv(actions_path, index=False)
        print(f"  Actions table: {actions_path}")
        
        # 4. Monthly Summary for Time Intelligence
        monthly_path = os.path.join(self.dashboard_dir, 'fact_monthly_summary.csv')
        self.monthly_summary.to_csv(monthly_path, index=False)
        print(f"  Monthly summary: {monthly_path}")
        
        # 5. Forecast Data
        if self.forecast_data is not None:
            forecast_df = self.forecast_data.copy()
            forecast_df['date_key'] = forecast_df['date'].dt.strftime('%Y%m%d').astype(int)
            forecast_df['forecast_type'] = 'Ensemble'
            
            forecast_path = os.path.join(self.dashboard_dir, 'fact_revenue_forecast.csv')
            forecast_df.to_csv(forecast_path, index=False)
            print(f"  Forecast data: {forecast_path}")
        
        # 6. KPI Cards Data
        kpi_data = self._calculate_kpis()
        kpi_df = pd.DataFrame([
            {'kpi_name': k, 'kpi_value': v, 'kpi_category': self._categorize_kpi(k)}
            for k, v in kpi_data.items()
        ])
        
        kpi_path = os.path.join(self.dashboard_dir, 'kpi_summary.csv')
        kpi_df.to_csv(kpi_path, index=False)
        print(f"  KPI summary: {kpi_path}")
        
        # 7. Segment Summary
        segment_summary = self.master_df.groupby('customer_segment').agg({
            'customer_id': 'count',
            'total_revenue': 'sum',
            'gross_profit': 'sum',
            'churn_probability': 'mean'
        }).reset_index()
        segment_summary.columns = ['segment', 'customers', 'revenue', 'profit', 'avg_churn_prob']
        
        segment_path = os.path.join(self.dashboard_dir, 'segment_summary.csv')
        segment_summary.to_csv(segment_path, index=False)
        print(f"  Segment summary: {segment_path}")
        
        # 8. Date Dimension
        all_dates = pd.date_range(
            start=self.customers['signup_date'].min(),
            end=self.transactions['transaction_date'].max() + pd.DateOffset(months=12),
            freq='D'
        )
        
        dim_date = pd.DataFrame({
            'date_key': all_dates.strftime('%Y%m%d').astype(int),
            'date': all_dates,
            'year': all_dates.year,
            'month': all_dates.month,
            'month_name': all_dates.strftime('%B'),
            'quarter': all_dates.quarter,
            'year_month': all_dates.strftime('%Y-%m'),
            'is_forecast_period': (all_dates > self.transactions['transaction_date'].max()).astype(int)
        })
        
        date_path = os.path.join(self.dashboard_dir, 'dim_date.csv')
        dim_date.to_csv(date_path, index=False)
        print(f"  Date dimension: {date_path}")
        
        print(f"\n  Total Power BI files exported: 8")
        
        return self.dashboard_dir
    
    def _calculate_kpis(self):
        """Calculate all KPIs for executive summary"""
        kpis = {}
        
        # Revenue KPIs
        kpis['total_revenue'] = self.master_df['total_revenue'].sum()
        kpis['avg_revenue_per_customer'] = self.master_df['total_revenue'].mean()
        kpis['monthly_avg_revenue'] = self.monthly_summary['revenue'].mean()
        
        # Customer KPIs
        kpis['total_customers'] = len(self.master_df)
        kpis['active_customers'] = len(self.master_df[self.master_df['is_churned'] == 0])
        kpis['churned_customers'] = len(self.master_df[self.master_df['is_churned'] == 1])
        kpis['churn_rate'] = (kpis['churned_customers'] / kpis['total_customers']) * 100
        
        # Profitability KPIs
        if 'gross_profit' in self.master_df.columns:
            kpis['total_profit'] = self.master_df['gross_profit'].sum()
            kpis['avg_profit_margin'] = self.master_df['profit_margin'].mean()
            kpis['profitable_customers'] = len(self.master_df[self.master_df['gross_profit'] > 0])
        
        # Churn Risk KPIs
        high_risk = self.master_df[self.master_df['churn_risk_level'] == 'High']
        kpis['high_risk_customers'] = len(high_risk)
        kpis['revenue_at_risk'] = high_risk['total_revenue'].sum()
        
        # Lifetime KPIs
        kpis['avg_customer_lifetime_months'] = self.master_df['lifetime_months'].mean()
        kpis['avg_customer_ltv'] = self.master_df['total_revenue'].mean()
        
        # Forecast KPIs
        if self.forecast_data is not None:
            kpis['forecasted_revenue_12m'] = self.forecast_data['forecast_ensemble'].sum()
            kpis['avg_monthly_forecast'] = self.forecast_data['forecast_ensemble'].mean()
        
        # Segment KPIs
        kpis['top_segment'] = self.master_df['customer_segment'].value_counts().index[0]
        kpis['num_segments'] = self.master_df['customer_segment'].nunique()
        
        self.kpis = kpis
        return kpis
    
    def _categorize_kpi(self, kpi_name):
        """Categorize KPI for dashboard grouping"""
        if 'revenue' in kpi_name.lower() or 'profit' in kpi_name.lower():
            return 'Financial'
        elif 'customer' in kpi_name.lower() or 'churn' in kpi_name.lower():
            return 'Customer'
        elif 'forecast' in kpi_name.lower():
            return 'Forecast'
        else:
            return 'Other'
    
    def generate_executive_report(self):
        """Step 5: Generate executive KPI summary report"""
        print("\n=== Step 5: Generating Executive KPI Report ===")
        
        kpis = self.kpis if self.kpis else self._calculate_kpis()
        report_date = datetime.now().strftime('%Y-%m-%d')
        
        report = f"""
================================================================================
                    EXECUTIVE ANALYTICS SUMMARY REPORT
                              {report_date}
================================================================================

BUSINESS OVERVIEW
-----------------
Total Customers:          {kpis['total_customers']:>15,}
Active Customers:         {kpis['active_customers']:>15,}
Customer Churn Rate:      {kpis['churn_rate']:>14.1f}%

FINANCIAL PERFORMANCE
---------------------
Total Revenue:            ${kpis['total_revenue']:>14,.2f}
Total Profit:             ${kpis.get('total_profit', 0):>14,.2f}
Average Profit Margin:    {kpis.get('avg_profit_margin', 0):>14.1f}%
Avg Revenue/Customer:     ${kpis['avg_revenue_per_customer']:>14,.2f}

CUSTOMER LIFETIME VALUE
-----------------------
Average Customer LTV:     ${kpis['avg_customer_ltv']:>14,.2f}
Avg Lifetime (months):    {kpis['avg_customer_lifetime_months']:>14.1f}

RISK ANALYSIS
-------------
High-Risk Customers:      {kpis['high_risk_customers']:>15,}
Revenue at Risk:          ${kpis['revenue_at_risk']:>14,.2f}
Risk as % of Total:       {(kpis['revenue_at_risk']/kpis['total_revenue']*100):>14.1f}%

12-MONTH FORECAST
-----------------
Forecasted Revenue:       ${kpis.get('forecasted_revenue_12m', 0):>14,.2f}
Avg Monthly Forecast:     ${kpis.get('avg_monthly_forecast', 0):>14,.2f}

CUSTOMER SEGMENTATION
---------------------
Number of Segments:       {kpis['num_segments']:>15}
Top Segment:              {kpis['top_segment']:>15}
"""
        
        # Add segment breakdown
        segment_stats = self.master_df.groupby('customer_segment').agg({
            'customer_id': 'count',
            'total_revenue': 'sum'
        }).reset_index()
        segment_stats.columns = ['segment', 'customers', 'revenue']
        segment_stats = segment_stats.sort_values('revenue', ascending=False)
        
        report += "\nSegment Performance:\n"
        for _, row in segment_stats.head(5).iterrows():
            pct = row['customers'] / kpis['total_customers'] * 100
            report += f"  {row['segment']:<20} {row['customers']:>6} customers ({pct:>5.1f}%) | ${row['revenue']:>12,.0f}\n"
        
        # Add profitability breakdown
        if 'profitability_tier' in self.master_df.columns:
            profit_stats = self.master_df.groupby('profitability_tier').agg({
                'customer_id': 'count',
                'gross_profit': 'sum'
            }).reset_index()
            profit_stats.columns = ['tier', 'customers', 'profit']
            
            report += "\nProfitability Tiers:\n"
            for _, row in profit_stats.iterrows():
                report += f"  {row['tier']:<20} {row['customers']:>6} customers | ${row['profit']:>12,.0f} profit\n"
        
        # Key insights
        report += """
KEY INSIGHTS & RECOMMENDATIONS
------------------------------
"""
        
        # Generate dynamic insights
        if kpis['churn_rate'] > 10:
            report += f"⚠️  Churn rate ({kpis['churn_rate']:.1f}%) is elevated - prioritize retention campaigns\n"
        
        if kpis['high_risk_customers'] > 100:
            report += f"⚠️  {kpis['high_risk_customers']} high-risk customers need immediate attention\n"
        
        revenue_at_risk_pct = kpis['revenue_at_risk'] / kpis['total_revenue'] * 100
        if revenue_at_risk_pct > 10:
            report += f"⚠️  {revenue_at_risk_pct:.1f}% of revenue is at risk from potential churn\n"
        
        if kpis.get('avg_profit_margin', 0) > 60:
            report += f"✓  Strong profit margins at {kpis.get('avg_profit_margin', 0):.1f}%\n"
        
        report += f"""
RECOMMENDED ACTIONS
-------------------
1. Focus retention efforts on {kpis['high_risk_customers']} high-risk customers
2. Protect ${kpis['revenue_at_risk']:,.0f} revenue at risk through targeted campaigns
3. Expand engagement with top segment: {kpis['top_segment']}
4. Monitor forecast vs actual for next 12 months

================================================================================
#                              END OF REPORT
================================================================================
"""

    def generate_executive_report(self):
        """Step 5: Generate executive KPI summary report"""
        print("\n=== Step 5: Generating Executive KPI Report ===")

        kpis = self.kpis if self.kpis else self._calculate_kpis()
        report_date = datetime.now().strftime('%Y-%m-%d')

        report = f"""
==============================================================
EXECUTIVE ANALYTICS SUMMARY REPORT
Date: {report_date}
==============================================================

Total Customers: {kpis['total_customers']}
Total Revenue: ${kpis['total_revenue']:,.2f}
Churn Rate: {kpis['churn_rate']:.2f}%

Revenue at Risk: ${kpis['revenue_at_risk']:,.2f}
High Risk Customers: {kpis['high_risk_customers']}

Top Segment: {kpis['top_segment']}
==============================================================
"""

        os.makedirs(self.output_dir, exist_ok=True)

        report_path = os.path.join(
        self.output_dir,
        "executive_kpi_report.txt"
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"Report saved to: {report_path}")

        return report

    
    def create_executive_visualizations(self):
        """Step 6: Create final visualization charts summarizing business performance"""
        print("\n=== Step 6: Creating Executive Visualizations ===")
        
        # 1. Executive Dashboard Overview (4-panel)
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # KPI Cards Row
        kpi_cards = [
            ('Total Revenue', f"${self.kpis['total_revenue']/1e6:.1f}M", '#3498db'),
            ('Total Customers', f"{self.kpis['total_customers']:,}", '#2ecc71'),
            ('Churn Rate', f"{self.kpis['churn_rate']:.1f}%", '#e74c3c'),
            ('Avg LTV', f"${self.kpis['avg_customer_ltv']:,.0f}", '#9b59b6')
        ]
        
        for i, (title, value, color) in enumerate(kpi_cards):
            ax = fig.add_subplot(gs[0, i])
            ax.text(0.5, 0.6, value, fontsize=28, fontweight='bold',
                   ha='center', va='center', color=color)
            ax.text(0.5, 0.2, title, fontsize=12, ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False,
                                       edgecolor=color, linewidth=3, transform=ax.transAxes))
        
        # Revenue Trend with Forecast
        ax1 = fig.add_subplot(gs[1, :2])
        ax1.plot(self.monthly_revenue['date'], self.monthly_revenue['revenue'],
                 'b-', linewidth=2, label='Actual Revenue')
        if self.forecast_data is not None:
            ax1.plot(self.forecast_data['date'], self.forecast_data['forecast_ensemble'],
                     'r--', linewidth=2, label='Forecast')
        ax1.fill_between(self.monthly_revenue['date'], self.monthly_revenue['revenue'], alpha=0.3)
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('Revenue ($)', fontsize=10)
        ax1.set_title('Revenue Trend & Forecast', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax1.grid(True, alpha=0.3)
        
        # Customer Segments
        ax2 = fig.add_subplot(gs[1, 2:])
        segment_counts = self.master_df['customer_segment'].value_counts().head(8)
        colors = sns.color_palette('Set2', len(segment_counts))
        wedges, texts, autotexts = ax2.pie(
            segment_counts, labels=segment_counts.index,
            autopct='%1.1f%%', colors=colors, startangle=90
        )
        ax2.set_title('Customer Segments', fontsize=12, fontweight='bold')
        
        # Profitability by Tier
        ax3 = fig.add_subplot(gs[2, :2])
        if 'profitability_tier' in self.master_df.columns:
            tier_profit = self.master_df.groupby('profitability_tier')['gross_profit'].sum()
            tier_order = ['Loss-Making', 'Low Profit', 'Medium Profit', 'High Profit']
            tier_profit = tier_profit.reindex([t for t in tier_order if t in tier_profit.index])
            colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71'][:len(tier_profit)]
            ax3.bar(tier_profit.index, tier_profit.values, color=colors)
            ax3.set_xlabel('Profitability Tier', fontsize=10)
            ax3.set_ylabel('Total Profit ($)', fontsize=10)
            ax3.set_title('Profit by Customer Tier', fontsize=12, fontweight='bold')
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
            ax3.tick_params(axis='x', rotation=15)
        
        # Churn Risk Distribution
        ax4 = fig.add_subplot(gs[2, 2:])
        if 'churn_risk_level' in self.master_df.columns:
            risk_counts = self.master_df['churn_risk_level'].value_counts()
            risk_order = ['Low', 'Medium', 'High']
            risk_counts = risk_counts.reindex([r for r in risk_order if r in risk_counts.index])
            colors = ['#2ecc71', '#f39c12', '#e74c3c'][:len(risk_counts)]
            ax4.bar(risk_counts.index, risk_counts.values, color=colors)
            ax4.set_xlabel('Risk Level', fontsize=10)
            ax4.set_ylabel('Customer Count', fontsize=10)
            ax4.set_title('Churn Risk Distribution', fontsize=12, fontweight='bold')
        
        plt.suptitle('Executive Business Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(
            os.path.join(self.visuals_dir, 'executive_dashboard_final.png'),
            dpi=150, bbox_inches='tight', facecolor='white'
        )
        plt.close()
        print("  Saved: executive_dashboard_final.png")
        
        # 2. Business Performance Summary
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Revenue by Country
        ax1 = axes[0, 0]
        country_rev = self.master_df.groupby('country')['total_revenue'].sum().nlargest(8)
        colors = sns.color_palette('Blues_r', len(country_rev))
        ax1.barh(country_rev.index, country_rev.values, color=colors)
        ax1.set_xlabel('Revenue ($)', fontsize=10)
        ax1.set_title('Revenue by Country (Top 8)', fontsize=12, fontweight='bold')
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        ax1.invert_yaxis()
        
        # Revenue by Subscription
        ax2 = axes[0, 1]
        sub_rev = self.master_df.groupby('subscription_type')['total_revenue'].sum()
        colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        ax2.pie(sub_rev, labels=sub_rev.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax2.set_title('Revenue by Subscription Type', fontsize=12, fontweight='bold')
        
        # Monthly Revenue Trend
        ax3 = axes[1, 0]
        monthly = self.monthly_summary.copy()
        ax3.bar(range(len(monthly)), monthly['revenue'],
                color=sns.color_palette('Blues', len(monthly)))
        ax3.set_xticks(range(0, len(monthly), 6))
        ax3.set_xticklabels(monthly['month'].iloc[::6], rotation=45, ha='right')
        ax3.set_xlabel('Month', fontsize=10)
        ax3.set_ylabel('Revenue ($)', fontsize=10)
        ax3.set_title('Monthly Revenue Trend', fontsize=12, fontweight='bold')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Customer Growth
        ax4 = axes[1, 1]
        ax4.plot(range(len(monthly)), monthly['cumulative_customers'],
                 'g-o', linewidth=2, markersize=4)
        ax4.fill_between(range(len(monthly)), monthly['cumulative_customers'],
                         alpha=0.3, color='green')
        ax4.set_xticks(range(0, len(monthly), 6))
        ax4.set_xticklabels(monthly['month'].iloc[::6], rotation=45, ha='right')
        ax4.set_xlabel('Month', fontsize=10)
        ax4.set_ylabel('Cumulative Customers', fontsize=10)
        ax4.set_title('Customer Growth Over Time', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, 'business_performance_summary.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print("  Saved: business_performance_summary.png")
        
        # 3. Risk and Retention Analysis
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Revenue at Risk by Segment
        ax1 = axes[0]
        risk_by_segment = self.master_df[self.master_df['churn_risk_level'] == 'High'].groupby(
            'customer_segment'
        )['total_revenue'].sum().nlargest(6)
        colors = sns.color_palette('Reds_r', len(risk_by_segment))
        ax1.barh(risk_by_segment.index, risk_by_segment.values, color=colors)
        ax1.set_xlabel('Revenue at Risk ($)', fontsize=10)
        ax1.set_title('Revenue at Risk by Segment', fontsize=12, fontweight='bold')
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax1.invert_yaxis()
        
        # Churn Probability Distribution
        ax2 = axes[1]
        ax2.hist(self.master_df['churn_probability'], bins=30,
                 color='steelblue', edgecolor='white', alpha=0.7)
        ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Risk Threshold')
        ax2.set_xlabel('Churn Probability', fontsize=10)
        ax2.set_ylabel('Customer Count', fontsize=10)
        ax2.set_title('Churn Probability Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        
        # Retention Actions Distribution
        ax3 = axes[2]
        if 'retention_action' in self.master_df.columns:
            action_counts = self.master_df['retention_action'].value_counts().head(5)
            colors = sns.color_palette('Set2', len(action_counts))
            ax3.barh(action_counts.index, action_counts.values, color=colors)
            ax3.set_xlabel('Customer Count', fontsize=10)
            ax3.set_title('Recommended Retention Actions', fontsize=12, fontweight='bold')
            ax3.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, 'risk_retention_analysis.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print("  Saved: risk_retention_analysis.png")
        
        print("\n  All executive visualizations created!")
        
        return True
    
    def run_unified_analytics(self):
        """Run complete unified analytics pipeline"""
        print("\n" + "="*60)
        print("UNIFIED ANALYTICS DASHBOARD PIPELINE")
        print("="*60)
        
        # Step 1: Load all outputs
        self.load_all_outputs()
        
        # Step 2: Create master customer dataset
        self.create_master_customer_dataset()
        
        # Step 3: Create monthly business summary
        self.create_monthly_business_summary()
        
        # Step 4: Export Power BI datasets
        self.export_powerbi_datasets()
        
        # Step 5: Generate executive report
        self.generate_executive_report()
        
        # Step 6: Create executive visualizations
        self.create_executive_visualizations()
        
        print("\n" + "="*60)
        print("UNIFIED ANALYTICS COMPLETE")
        print("="*60)
        
        # Print summary
        print(f"\nOutput Summary:")
        print(f"  Master dataset: {len(self.master_df)} customers, {len(self.master_df.columns)} attributes")
        print(f"  Monthly summary: {len(self.monthly_summary)} months")
        print(f"  Power BI files: 8 datasets")
        print(f"  Visualizations: 3 charts")
        
        return {
            'master_df': self.master_df,
            'monthly_summary': self.monthly_summary,
            'kpis': self.kpis
        }


def main():
    """Main entry point"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, 'outputs', 'unified')
    visuals_dir = os.path.join(base_dir, 'visuals')
    
    dashboard = UnifiedAnalyticsDashboard(base_dir, output_dir, visuals_dir)
    results = dashboard.run_unified_analytics()
    
    return results


if __name__ == "__main__":
    main()
