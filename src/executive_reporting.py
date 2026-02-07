"""
Executive Reporting Module
Generates executive summaries, KPI dashboards, and Power BI-ready outputs
"""

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


class ExecutiveReporter:
    """Generate executive reports and Power BI-ready outputs"""
    
    def __init__(self, base_dir, output_dir, visuals_dir, dashboard_dir):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.visuals_dir = visuals_dir
        self.dashboard_dir = dashboard_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(visuals_dir, exist_ok=True)
        os.makedirs(dashboard_dir, exist_ok=True)
    
    def load_all_outputs(self):
        """Load all analysis outputs"""
        print("Loading analysis outputs...")
        
        self.outputs = {}
        outputs_dir = os.path.join(self.base_dir, 'outputs')
        
        # Load main processed data
        self.outputs['customers'] = pd.read_csv(
            os.path.join(outputs_dir, 'processed_customers.csv')
        )
        self.outputs['transactions'] = pd.read_csv(
            os.path.join(outputs_dir, 'processed_transactions.csv')
        )
        self.outputs['monthly_revenue'] = pd.read_csv(
            os.path.join(outputs_dir, 'monthly_revenue.csv')
        )
        
        # Load forecasts
        forecasts_dir = os.path.join(outputs_dir, 'forecasts')
        if os.path.exists(forecasts_dir):
            self.outputs['prophet_forecast'] = pd.read_csv(
                os.path.join(forecasts_dir, 'prophet_forecast.csv')
            )
            self.outputs['combined_forecast'] = pd.read_csv(
                os.path.join(forecasts_dir, 'combined_forecast.csv')
            )
        
        # Load churn predictions
        churn_dir = os.path.join(outputs_dir, 'churn')
        if os.path.exists(churn_dir):
            self.outputs['churn_predictions'] = pd.read_csv(
                os.path.join(churn_dir, 'churn_predictions.csv')
            )
            self.outputs['churn_risk_summary'] = pd.read_csv(
                os.path.join(churn_dir, 'churn_risk_summary.csv')
            )
        
        # Load profitability
        profit_dir = os.path.join(outputs_dir, 'profitability')
        if os.path.exists(profit_dir):
            self.outputs['profitability'] = pd.read_csv(
                os.path.join(profit_dir, 'customer_profitability.csv')
            )
            self.outputs['segmentation'] = pd.read_csv(
                os.path.join(profit_dir, 'customer_segmentation.csv')
            )
        
        # Load cohort/RFM
        cohort_dir = os.path.join(outputs_dir, 'cohort_rfm')
        if os.path.exists(cohort_dir):
            self.outputs['rfm'] = pd.read_csv(
                os.path.join(cohort_dir, 'rfm_analysis.csv')
            )
            self.outputs['lifecycle'] = pd.read_csv(
                os.path.join(cohort_dir, 'lifecycle_analysis.csv')
            )
        
        print(f"Loaded {len(self.outputs)} datasets")
        return self.outputs
    
    def calculate_kpis(self):
        """Calculate key performance indicators"""
        print("\n=== Calculating KPIs ===")
        
        customers = self.outputs['customers']
        transactions = self.outputs['transactions']
        
        # Convert dates
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        customers['signup_date'] = pd.to_datetime(customers['signup_date'])
        
        # Filter successful transactions
        successful_txns = transactions[transactions['transaction_status'] == 'Completed']
        
        # Basic KPIs
        kpis = {
            # Customer metrics
            'total_customers': len(customers),
            'active_customers': len(customers[customers['is_churned'] == 0]),
            'churned_customers': len(customers[customers['is_churned'] == 1]),
            'churn_rate': customers['is_churned'].mean() * 100,
            
            # Revenue metrics
            'total_revenue': successful_txns['amount'].sum(),
            'avg_revenue_per_customer': successful_txns.groupby('customer_id')['amount'].sum().mean(),
            'avg_transaction_value': successful_txns['amount'].mean(),
            
            # Transaction metrics
            'total_transactions': len(transactions),
            'successful_transactions': len(successful_txns),
            'transaction_success_rate': (len(successful_txns) / len(transactions)) * 100,
            
            # Growth metrics (YoY if applicable)
            'avg_monthly_revenue': successful_txns.groupby(
                successful_txns['transaction_date'].dt.to_period('M')
            )['amount'].sum().mean(),
            
            # Customer lifetime
            'avg_customer_tenure_days': customers['customer_tenure_days'].mean(),
            'avg_customer_ltv': customers['customer_ltv'].mean() if 'customer_ltv' in customers.columns else customers['total_revenue'].mean(),
        }
        
        # Subscription distribution
        sub_distribution = customers['subscription_type'].value_counts().to_dict()
        for sub_type, count in sub_distribution.items():
            kpis[f'customers_{sub_type.lower()}'] = count
        
        # Add profitability KPIs if available
        if 'profitability' in self.outputs:
            profit_df = self.outputs['profitability']
            kpis['total_profit'] = profit_df['gross_profit'].sum()
            kpis['avg_profit_margin'] = profit_df['profit_margin'].mean()
            kpis['profitable_customers_pct'] = (profit_df['profit_margin'] > 0).mean() * 100
        
        # Add churn KPIs if available
        if 'churn_risk_summary' in self.outputs:
            risk_df = self.outputs['churn_risk_summary']
            high_risk = risk_df[risk_df['risk_level'] == 'High']
            if len(high_risk) > 0:
                kpis['high_risk_customers'] = high_risk['customer_count'].values[0]
                kpis['revenue_at_risk'] = high_risk['revenue_at_risk'].values[0]
        
        # Convert to DataFrame
        kpi_df = pd.DataFrame([
            {'metric': k, 'value': v} for k, v in kpis.items()
        ])
        
        kpi_df.to_csv(
            os.path.join(self.output_dir, 'executive_kpis.csv'),
            index=False
        )
        
        self.kpis = kpis
        
        print("\n=== Key Performance Indicators ===")
        print(f"Total Customers: {kpis['total_customers']:,}")
        print(f"Active Customers: {kpis['active_customers']:,}")
        print(f"Churn Rate: {kpis['churn_rate']:.1f}%")
        print(f"Total Revenue: ${kpis['total_revenue']:,.2f}")
        print(f"Avg Revenue per Customer: ${kpis['avg_revenue_per_customer']:,.2f}")
        print(f"Avg Monthly Revenue: ${kpis['avg_monthly_revenue']:,.2f}")
        
        return kpis
    
    def create_executive_dashboard_data(self):
        """Create data files optimized for Power BI"""
        print("\n=== Creating Power BI-Ready Data ===")
        
        # 1. Fact table - Transactions
        transactions = self.outputs['transactions'].copy()
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        transactions['date_key'] = transactions['transaction_date'].dt.strftime('%Y%m%d').astype(int)
        
        fact_transactions = transactions[[
            'transaction_id', 'customer_id', 'date_key', 'amount',
            'payment_method', 'transaction_status', 'is_successful'
        ]].copy()
        
        fact_transactions.to_csv(
            os.path.join(self.dashboard_dir, 'fact_transactions.csv'),
            index=False
        )
        
        # 2. Dimension table - Customers
        customers = self.outputs['customers'].copy()
        
        dim_customers = customers[[
            'customer_id', 'signup_date', 'country', 'industry',
            'subscription_type', 'monthly_fee'
        ]].copy()
        dim_customers['signup_date'] = pd.to_datetime(dim_customers['signup_date'])
        dim_customers['signup_date_key'] = dim_customers['signup_date'].dt.strftime('%Y%m%d').astype(int)
        
        dim_customers.to_csv(
            os.path.join(self.dashboard_dir, 'dim_customers.csv'),
            index=False
        )
        
        # 3. Dimension table - Date
        all_dates = pd.date_range(
            start=transactions['transaction_date'].min(),
            end=transactions['transaction_date'].max() + pd.DateOffset(months=12),
            freq='D'
        )
        
        dim_date = pd.DataFrame({
            'date_key': all_dates.strftime('%Y%m%d').astype(int),
            'date': all_dates,
            'year': all_dates.year,
            'month': all_dates.month,
            'month_name': all_dates.strftime('%B'),
            'quarter': all_dates.quarter,
            'day_of_week': all_dates.dayofweek,
            'day_name': all_dates.strftime('%A'),
            'is_weekend': (all_dates.dayofweek >= 5).astype(int),
            'fiscal_year': all_dates.year,
            'fiscal_quarter': all_dates.quarter
        })
        
        dim_date.to_csv(
            os.path.join(self.dashboard_dir, 'dim_date.csv'),
            index=False
        )
        
        # 4. Summary tables
        # Monthly revenue summary
        monthly_summary = transactions[transactions['is_successful'] == 1].groupby(
            transactions['transaction_date'].dt.to_period('M')
        ).agg({
            'amount': 'sum',
            'transaction_id': 'count',
            'customer_id': 'nunique'
        }).reset_index()
        monthly_summary.columns = ['month', 'revenue', 'transactions', 'unique_customers']
        monthly_summary['month'] = monthly_summary['month'].astype(str)
        
        monthly_summary.to_csv(
            os.path.join(self.dashboard_dir, 'monthly_summary.csv'),
            index=False
        )
        
        # 5. Customer metrics
        customer_metrics = customers[[
            'customer_id', 'total_revenue', 'total_transactions',
            'customer_tenure_days', 'is_churned', 'customer_ltv'
        ]].copy()
        
        # Add churn probability if available
        if 'churn_predictions' in self.outputs:
            churn_df = self.outputs['churn_predictions'][['customer_id', 'churn_probability', 'churn_risk_level']]
            customer_metrics = customer_metrics.merge(churn_df, on='customer_id', how='left')
        
        # Add RFM if available
        if 'rfm' in self.outputs:
            rfm_df = self.outputs['rfm'][['customer_id', 'R_score', 'F_score', 'M_score', 'segment']]
            customer_metrics = customer_metrics.merge(rfm_df, on='customer_id', how='left')
        
        customer_metrics.to_csv(
            os.path.join(self.dashboard_dir, 'customer_metrics.csv'),
            index=False
        )
        
        # 6. Forecast data
        if 'combined_forecast' in self.outputs:
            self.outputs['combined_forecast'].to_csv(
                os.path.join(self.dashboard_dir, 'revenue_forecast.csv'),
                index=False
            )
        
        print(f"Created Power BI-ready data files in {self.dashboard_dir}")
        
        return self.dashboard_dir
    
    def create_executive_summary_visual(self):
        """Create executive summary dashboard visual"""
        print("\n=== Creating Executive Summary Visual ===")
        
        fig = plt.figure(figsize=(20, 12))
        
        # Grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. KPI Cards (top row)
        kpi_values = [
            ('Total Customers', f"{self.kpis['total_customers']:,}", '#3498db'),
            ('Total Revenue', f"${self.kpis['total_revenue']/1e6:.1f}M", '#2ecc71'),
            ('Churn Rate', f"{self.kpis['churn_rate']:.1f}%", '#e74c3c'),
            ('Avg LTV', f"${self.kpis['avg_customer_ltv']:,.0f}", '#9b59b6')
        ]
        
        for i, (title, value, color) in enumerate(kpi_values):
            ax = fig.add_subplot(gs[0, i])
            ax.text(0.5, 0.6, value, fontsize=28, fontweight='bold',
                   ha='center', va='center', color=color)
            ax.text(0.5, 0.2, title, fontsize=12, ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False,
                                       edgecolor=color, linewidth=3))
        
        # 2. Revenue trend (middle left)
        ax2 = fig.add_subplot(gs[1, :2])
        monthly_rev = self.outputs['monthly_revenue'].copy()
        monthly_rev['date'] = pd.to_datetime(monthly_rev['date'])
        
        ax2.plot(monthly_rev['date'], monthly_rev['revenue'], 'b-', linewidth=2)
        ax2.fill_between(monthly_rev['date'], monthly_rev['revenue'], alpha=0.3)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Revenue ($)', fontsize=10)
        ax2.set_title('Monthly Revenue Trend', fontsize=12, fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax2.grid(True, alpha=0.3)
        
        # 3. Customer segments (middle right)
        ax3 = fig.add_subplot(gs[1, 2:])
        if 'segmentation' in self.outputs:
            seg_counts = self.outputs['segmentation']['segment_name'].value_counts()
            colors = sns.color_palette('Set2', len(seg_counts))
            ax3.pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax3.set_title('Customer Segments', fontsize=12, fontweight='bold')
        else:
            sub_counts = self.outputs['customers']['subscription_type'].value_counts()
            ax3.pie(sub_counts, labels=sub_counts.index, autopct='%1.1f%%',
                   startangle=90)
            ax3.set_title('Subscription Types', fontsize=12, fontweight='bold')
        
        # 4. Revenue by country (bottom left)
        ax4 = fig.add_subplot(gs[2, :2])
        country_rev = self.outputs['customers'].groupby('country')['total_revenue'].sum().nlargest(8)
        colors = sns.color_palette('Blues_r', len(country_rev))
        ax4.barh(country_rev.index, country_rev.values, color=colors)
        ax4.set_xlabel('Revenue ($)', fontsize=10)
        ax4.set_title('Revenue by Country (Top 8)', fontsize=12, fontweight='bold')
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        ax4.invert_yaxis()
        
        # 5. Churn risk distribution (bottom right)
        ax5 = fig.add_subplot(gs[2, 2:])
        if 'churn_predictions' in self.outputs:
            churn_data = self.outputs['churn_predictions']
            risk_counts = churn_data['churn_risk_level'].value_counts()
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            ax5.bar(risk_counts.index, risk_counts.values, color=colors)
            ax5.set_xlabel('Risk Level', fontsize=10)
            ax5.set_ylabel('Customer Count', fontsize=10)
            ax5.set_title('Churn Risk Distribution', fontsize=12, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'Churn data not available',
                    ha='center', va='center', fontsize=12)
            ax5.axis('off')
        
        plt.suptitle('Executive Business Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(
            os.path.join(self.visuals_dir, 'executive_dashboard.png'),
            dpi=150, bbox_inches='tight',
            facecolor='white'
        )
        plt.close()
        print("Saved executive dashboard visualization")
    
    def generate_text_report(self):
        """Generate text-based executive report"""
        print("\n=== Generating Executive Report ===")
        
        report_date = datetime.now().strftime('%Y-%m-%d')
        
        report = f"""
================================================================================
                    FINANCIAL OPERATIONS ANALYTICS REPORT
                              {report_date}
================================================================================

EXECUTIVE SUMMARY
-----------------
This report provides a comprehensive analysis of financial operations including
revenue forecasting, customer churn prediction, profitability analysis, and
customer segmentation.

KEY PERFORMANCE INDICATORS
--------------------------
• Total Customers: {self.kpis['total_customers']:,}
• Active Customers: {self.kpis['active_customers']:,}
• Customer Churn Rate: {self.kpis['churn_rate']:.1f}%
• Total Revenue: ${self.kpis['total_revenue']:,.2f}
• Average Revenue per Customer: ${self.kpis['avg_revenue_per_customer']:,.2f}
• Average Monthly Revenue: ${self.kpis['avg_monthly_revenue']:,.2f}
• Transaction Success Rate: {self.kpis['transaction_success_rate']:.1f}%
• Average Customer Tenure: {self.kpis['avg_customer_tenure_days']:.0f} days
• Average Customer LTV: ${self.kpis['avg_customer_ltv']:,.2f}

"""
        
        # Add profitability section
        if 'profitability' in self.outputs:
            profit_df = self.outputs['profitability']
            report += f"""
PROFITABILITY ANALYSIS
----------------------
• Total Gross Profit: ${self.kpis.get('total_profit', 0):,.2f}
• Average Profit Margin: {self.kpis.get('avg_profit_margin', 0):.1f}%
• Profitable Customers: {self.kpis.get('profitable_customers_pct', 0):.1f}%

Profitability by Subscription Type:
"""
            sub_profit = profit_df.groupby('subscription_type')['profit_margin'].mean()
            for sub, margin in sub_profit.items():
                report += f"  • {sub}: {margin:.1f}% avg margin\n"
        
        # Add churn section
        if 'churn_risk_summary' in self.outputs:
            report += f"""
CHURN RISK ANALYSIS
-------------------
• High Risk Customers: {self.kpis.get('high_risk_customers', 0):,}
• Revenue at Risk: ${self.kpis.get('revenue_at_risk', 0):,.2f}

Risk Distribution:
"""
            risk_df = self.outputs['churn_risk_summary']
            for _, row in risk_df.iterrows():
                report += f"  • {row['risk_level']}: {row['customer_count']:,} customers\n"
        
        # Add forecast section
        if 'combined_forecast' in self.outputs:
            forecast = self.outputs['combined_forecast']
            next_12_revenue = forecast['forecast_ensemble'].sum()
            report += f"""
REVENUE FORECAST (Next 12 Months)
---------------------------------
• Projected Revenue: ${next_12_revenue:,.2f}
• Average Monthly Forecast: ${next_12_revenue/12:,.2f}

"""
        
        # Add RFM section
        if 'rfm' in self.outputs:
            rfm = self.outputs['rfm']
            segment_counts = rfm['segment'].value_counts()
            report += f"""
CUSTOMER SEGMENTATION (RFM Analysis)
------------------------------------
Top Customer Segments:
"""
            for segment, count in segment_counts.head(5).items():
                pct = count / len(rfm) * 100
                report += f"  • {segment}: {count:,} ({pct:.1f}%)\n"
        
        report += """
================================================================================
                              END OF REPORT
================================================================================
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, 'executive_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Saved executive report to {report_path}")
        
        return report
    
    def run_executive_reporting(self):
        """Run complete executive reporting pipeline"""
        print("\n" + "="*60)
        print("EXECUTIVE REPORTING MODULE")
        print("="*60)
        
        self.load_all_outputs()
        self.calculate_kpis()
        self.create_executive_dashboard_data()
        self.create_executive_summary_visual()
        report = self.generate_text_report()
        
        print("\n=== Executive Reporting Complete ===")
        print(f"Outputs saved to: {self.output_dir}")
        print(f"Dashboard data saved to: {self.dashboard_dir}")
        
        return {
            'kpis': self.kpis,
            'report': report
        }


def main():
    """Main entry point"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, 'outputs', 'reports')
    visuals_dir = os.path.join(base_dir, 'visuals')
    dashboard_dir = os.path.join(base_dir, 'dashboard')
    
    reporter = ExecutiveReporter(base_dir, output_dir, visuals_dir, dashboard_dir)
    results = reporter.run_executive_reporting()
    
    return results


if __name__ == "__main__":
    main()
