"""
Customer Segmentation and Cohort Analysis Module
RFM Analysis, Behavioral Clustering, and Cohort Retention Analysis
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


class CustomerSegmentationAnalyzer:
    """Customer segmentation using RFM, clustering, and cohort analysis"""
    
    def __init__(self, data_dir, output_dir, visuals_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.visuals_dir = visuals_dir
        
        self.customers_df = None
        self.transactions_df = None
        self.rfm_df = None
        self.segments_df = None
        self.cohort_data = None
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(visuals_dir, exist_ok=True)
        
        # Segment definitions
        self.segment_definitions = {
            'Champions': {
                'description': 'Best customers - high value, recent, frequent buyers',
                'action': 'Reward program, early access, referral program'
            },
            'Loyal Customers': {
                'description': 'Consistent customers with good engagement',
                'action': 'Upsell higher-value products, loyalty program'
            },
            'Potential Loyalists': {
                'description': 'Recent customers with good potential',
                'action': 'Onboarding support, membership offers'
            },
            'New Customers': {
                'description': 'Recently acquired customers',
                'action': 'Welcome series, product education'
            },
            'Promising': {
                'description': 'New shoppers with above-average activity',
                'action': 'Create brand awareness, free trials'
            },
            'Need Attention': {
                'description': 'Above-average but showing declining engagement',
                'action': 'Limited time offers, personalized outreach'
            },
            'About to Sleep': {
                'description': 'Below-average engagement, at risk of churning',
                'action': 'Re-engagement campaign, win-back offers'
            },
            'At Risk': {
                'description': 'High-value customers showing declining activity',
                'action': 'Personal outreach, special discounts'
            },
            'Cant Lose': {
                'description': 'Previously high-value, now inactive',
                'action': 'Win-back campaign, survey for feedback'
            },
            'Hibernating': {
                'description': 'Low engagement, been inactive for a while',
                'action': 'Re-activation campaign, special offers'
            },
            'Lost': {
                'description': 'Lowest engagement, likely churned',
                'action': 'Ignore or minimal effort reactivation'
            }
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
    
    def perform_rfm_segmentation(self):
        """Step 1: Perform RFM segmentation using Recency, Frequency, and Monetary metrics"""
        print("\n=== Step 1: RFM Segmentation ===")
        
        # Reference date (latest transaction + 1 day)
        reference_date = self.transactions_df['transaction_date'].max() + pd.Timedelta(days=1)
        print(f"Reference date: {reference_date.date()}")
        
        # Filter successful transactions
        successful_txns = self.transactions_df[
            self.transactions_df['transaction_status'] == 'Completed'
        ].copy()
        
        # Calculate RFM metrics
        rfm = successful_txns.groupby('customer_id').agg({
            'transaction_date': lambda x: (reference_date - x.max()).days,  # Recency
            'transaction_id': 'count',  # Frequency
            'amount': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
        
        # Handle edge cases
        rfm['monetary'] = rfm['monetary'].clip(lower=0)
        
        print(f"\nRFM Metrics Summary:")
        print(f"  Recency:   Min={rfm['recency'].min()}, Max={rfm['recency'].max()}, Avg={rfm['recency'].mean():.1f} days")
        print(f"  Frequency: Min={rfm['frequency'].min()}, Max={rfm['frequency'].max()}, Avg={rfm['frequency'].mean():.1f} transactions")
        print(f"  Monetary:  Min=${rfm['monetary'].min():.2f}, Max=${rfm['monetary'].max():.2f}, Avg=${rfm['monetary'].mean():.2f}")
        
        # Calculate RFM scores (1-5 scale)
        # Recency: Lower is better (more recent)
        rfm['R_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
        
        # Frequency: Higher is better
        rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        
        # Monetary: Higher is better
        rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        
        # Combined scores
        rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
        rfm['RFM_total'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
        
        # Assign RFM segments
        rfm['rfm_segment'] = rfm.apply(self._assign_rfm_segment, axis=1)
        
        # Merge with customer data
        rfm = rfm.merge(
            self.customers_df[['customer_id', 'country', 'industry', 'subscription_type', 'signup_date']],
            on='customer_id',
            how='left'
        )
        
        self.rfm_df = rfm
        
        # Print segment distribution
        segment_dist = rfm['rfm_segment'].value_counts()
        print(f"\nRFM Segment Distribution:")
        for segment, count in segment_dist.items():
            pct = count / len(rfm) * 100
            print(f"  {segment}: {count} ({pct:.1f}%)")
        
        return rfm
    
    def _assign_rfm_segment(self, row):
        """Assign RFM segment based on scores"""
        r, f, m = row['R_score'], row['F_score'], row['M_score']
        
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 4 and m >= 4:
            return 'Loyal Customers'
        elif r >= 4 and f >= 2 and m >= 2:
            return 'Potential Loyalists'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r >= 3 and f >= 2 and m >= 2:
            return 'Promising'
        elif r >= 3 and f >= 3 and m <= 3:
            return 'Need Attention'
        elif r == 2 and f <= 3:
            return 'About to Sleep'
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk'
        elif r <= 2 and f >= 4 and m >= 4:
            return 'Cant Lose'
        elif r <= 2 and f <= 2:
            return 'Hibernating'
        else:
            return 'Lost'
    
    def apply_behavioral_clustering(self, n_clusters=5):
        """Step 2: Apply clustering to group customers into behavioral segments"""
        print("\n=== Step 2: Behavioral Clustering ===")
        
        # Prepare features for clustering
        cluster_features = ['recency', 'frequency', 'monetary', 'RFM_total']
        X = self.rfm_df[cluster_features].copy()
        
        # Handle any missing or infinite values
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find optimal number of clusters using silhouette score
        silhouette_scores = []
        k_range = range(3, 8)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k}")
        
        # Final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.rfm_df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_profile = self.rfm_df.groupby('cluster').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': ['sum', 'mean'],
            'RFM_total': 'mean'
        }).reset_index()
        
        cluster_profile.columns = [
            'cluster', 'customer_count', 'avg_recency', 'avg_frequency',
            'total_revenue', 'avg_revenue', 'avg_rfm_score'
        ]
        
        # Assign behavioral labels based on cluster characteristics
        cluster_profile = cluster_profile.sort_values('avg_rfm_score', ascending=False)
        
        behavioral_labels = ['VIP Elite', 'Engaged Regulars', 'Growth Potential', 
                            'Casual Buyers', 'Dormant', 'At-Risk', 'Lost'][:optimal_k]
        
        cluster_to_label = dict(zip(
            cluster_profile['cluster'].values,
            behavioral_labels
        ))
        
        self.rfm_df['behavioral_segment'] = self.rfm_df['cluster'].map(cluster_to_label)
        cluster_profile['behavioral_segment'] = cluster_profile['cluster'].map(cluster_to_label)
        
        print("\nBehavioral Cluster Profiles:")
        print("-" * 90)
        for _, row in cluster_profile.iterrows():
            print(f"  {row['behavioral_segment']:<18} | Customers: {row['customer_count']:>5} | "
                  f"Avg Recency: {row['avg_recency']:>5.0f}d | Avg Freq: {row['avg_frequency']:>4.1f} | "
                  f"Avg Revenue: ${row['avg_revenue']:>8,.0f}")
        
        self.cluster_profile = cluster_profile
        self.silhouette_scores = list(zip(k_range, silhouette_scores))
        
        return self.rfm_df, cluster_profile
    
    def generate_segment_labels(self):
        """Step 3: Generate customer segment labels and descriptions"""
        print("\n=== Step 3: Generating Segment Labels ===")
        
        # Create comprehensive segment summary
        segment_summary = self.rfm_df.groupby('rfm_segment').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': ['sum', 'mean'],
            'RFM_total': 'mean'
        }).reset_index()
        
        segment_summary.columns = [
            'segment', 'customer_count', 'avg_recency', 'avg_frequency',
            'total_revenue', 'avg_revenue', 'avg_rfm_score'
        ]
        
        # Add descriptions and actions
        segment_summary['description'] = segment_summary['segment'].map(
            lambda x: self.segment_definitions.get(x, {}).get('description', 'N/A')
        )
        segment_summary['recommended_action'] = segment_summary['segment'].map(
            lambda x: self.segment_definitions.get(x, {}).get('action', 'N/A')
        )
        
        # Calculate percentages
        total_customers = len(self.rfm_df)
        total_revenue = self.rfm_df['monetary'].sum()
        
        segment_summary['customer_pct'] = (segment_summary['customer_count'] / total_customers * 100).round(1)
        segment_summary['revenue_pct'] = (segment_summary['total_revenue'] / total_revenue * 100).round(1)
        
        # Sort by avg_rfm_score descending
        segment_summary = segment_summary.sort_values('avg_rfm_score', ascending=False)
        
        # Save segment summary
        segment_summary.to_csv(
            os.path.join(self.output_dir, 'rfm_segment_summary.csv'),
            index=False
        )
        
        print("\nRFM Segment Summary with Actions:")
        print("-" * 100)
        for _, row in segment_summary.iterrows():
            print(f"\n{row['segment'].upper()}")
            print(f"  Customers: {row['customer_count']:,} ({row['customer_pct']}%)")
            print(f"  Revenue: ${row['total_revenue']:,.0f} ({row['revenue_pct']}%)")
            print(f"  Description: {row['description']}")
            print(f"  Action: {row['recommended_action']}")
        
        self.segment_summary = segment_summary
        
        return segment_summary
    
    def perform_cohort_analysis(self):
        """Step 4: Perform cohort retention analysis by signup month"""
        print("\n=== Step 4: Cohort Retention Analysis ===")
        
        # Prepare data
        customers = self.customers_df.copy()
        transactions = self.transactions_df[
            self.transactions_df['transaction_status'] == 'Completed'
        ].copy()
        
        # Create signup cohort
        customers['cohort'] = customers['signup_date'].dt.to_period('M')
        transactions = transactions.merge(
            customers[['customer_id', 'cohort']],
            on='customer_id',
            how='left'
        )
        
        # Transaction month
        transactions['transaction_month'] = transactions['transaction_date'].dt.to_period('M')
        
        # Cohort index (months since signup)
        transactions['cohort_index'] = (
            transactions['transaction_month'] - transactions['cohort']
        ).apply(lambda x: x.n)
        
        # Create cohort matrix - unique customers
        cohort_data = transactions.groupby(
            ['cohort', 'cohort_index']
        )['customer_id'].nunique().reset_index()
        cohort_data.columns = ['cohort', 'cohort_index', 'customers']
        
        # Pivot to matrix
        cohort_matrix = cohort_data.pivot(
            index='cohort',
            columns='cohort_index',
            values='customers'
        ).fillna(0)
        
        # Calculate cohort sizes
        cohort_sizes = cohort_matrix.iloc[:, 0]
        
        # Calculate retention rates
        retention_matrix = cohort_matrix.divide(cohort_sizes, axis=0) * 100
        
        # Average retention by month
        avg_retention = retention_matrix.mean(axis=0).reset_index()
        avg_retention.columns = ['month', 'avg_retention']
        
        print(f"Analyzed {len(cohort_matrix)} cohorts")
        print(f"Cohort period: {cohort_matrix.index.min()} to {cohort_matrix.index.max()}")
        
        # Print retention summary
        print("\nRetention Rate by Month:")
        for _, row in avg_retention.head(12).iterrows():
            print(f"  Month {int(row['month'])}: {row['avg_retention']:.1f}%")
        
        self.cohort_matrix = cohort_matrix
        self.retention_matrix = retention_matrix
        self.avg_retention = avg_retention
        
        return cohort_matrix, retention_matrix
    
    def calculate_cohort_revenue(self):
        """Step 5: Calculate retention and revenue contribution by cohort"""
        print("\n=== Step 5: Cohort Revenue Analysis ===")
        
        # Prepare data
        customers = self.customers_df.copy()
        transactions = self.transactions_df[
            self.transactions_df['transaction_status'] == 'Completed'
        ].copy()
        
        # Create signup cohort
        customers['cohort'] = customers['signup_date'].dt.to_period('M')
        transactions = transactions.merge(
            customers[['customer_id', 'cohort']],
            on='customer_id',
            how='left'
        )
        
        # Cohort revenue analysis
        cohort_revenue = transactions.groupby('cohort').agg({
            'customer_id': 'nunique',
            'amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        cohort_revenue.columns = ['cohort', 'customers', 'total_revenue', 'transactions']
        cohort_revenue['cohort'] = cohort_revenue['cohort'].astype(str)
        
        # Calculate averages
        cohort_revenue['avg_revenue_per_customer'] = (
            cohort_revenue['total_revenue'] / cohort_revenue['customers']
        )
        cohort_revenue['avg_transactions_per_customer'] = (
            cohort_revenue['transactions'] / cohort_revenue['customers']
        )
        
        # Revenue contribution
        total_revenue = cohort_revenue['total_revenue'].sum()
        cohort_revenue['revenue_contribution_pct'] = (
            cohort_revenue['total_revenue'] / total_revenue * 100
        )
        
        # Cumulative customers
        cohort_revenue['cumulative_customers'] = cohort_revenue['customers'].cumsum()
        
        # Calculate cohort LTV (revenue / customers)
        cohort_revenue['cohort_ltv'] = cohort_revenue['total_revenue'] / cohort_revenue['customers']
        
        # Save cohort revenue
        cohort_revenue.to_csv(
            os.path.join(self.output_dir, 'cohort_revenue_analysis.csv'),
            index=False
        )
        
        print("\nCohort Revenue Summary (Recent 12 cohorts):")
        print("-" * 90)
        for _, row in cohort_revenue.tail(12).iterrows():
            print(f"  {row['cohort']}: {row['customers']:>4} customers | "
                  f"${row['total_revenue']:>10,.0f} revenue | "
                  f"${row['avg_revenue_per_customer']:>8,.0f} per customer")
        
        # Calculate overall metrics
        print(f"\nOverall Metrics:")
        print(f"  Total Cohorts: {len(cohort_revenue)}")
        print(f"  Total Customers: {cohort_revenue['customers'].sum():,}")
        print(f"  Total Revenue: ${total_revenue:,.2f}")
        print(f"  Avg LTV: ${cohort_revenue['cohort_ltv'].mean():,.2f}")
        
        self.cohort_revenue = cohort_revenue
        
        return cohort_revenue
    
    def export_dashboard_data(self):
        """Step 6: Export segmentation and cohort datasets for dashboard usage"""
        print("\n=== Step 6: Exporting Dashboard Data ===")
        
        # 1. Customer segmentation dashboard data
        segment_dashboard = self.rfm_df[[
            'customer_id', 'recency', 'frequency', 'monetary',
            'R_score', 'F_score', 'M_score', 'RFM_score', 'RFM_total',
            'rfm_segment', 'behavioral_segment', 'country', 'industry', 'subscription_type'
        ]].copy()
        
        segment_dashboard['revenue_formatted'] = segment_dashboard['monetary'].apply(lambda x: f"${x:,.2f}")
        
        segment_path = os.path.join(self.output_dir, 'segmentation_dashboard.csv')
        segment_dashboard.to_csv(segment_path, index=False)
        print(f"  Segmentation dashboard: {segment_path}")
        
        # 2. RFM segment summary for dashboard cards
        segment_kpis = self.segment_summary[[
            'segment', 'customer_count', 'customer_pct', 'total_revenue',
            'revenue_pct', 'avg_revenue', 'avg_rfm_score'
        ]].copy()
        
        segment_kpis_path = os.path.join(self.output_dir, 'segment_kpis.csv')
        segment_kpis.to_csv(segment_kpis_path, index=False)
        print(f"  Segment KPIs: {segment_kpis_path}")
        
        # 3. Cohort retention for heatmap
        retention_export = self.retention_matrix.copy()
        retention_export.index = retention_export.index.astype(str)
        retention_export = retention_export.iloc[:, :12]  # First 12 months
        
        retention_path = os.path.join(self.output_dir, 'cohort_retention_dashboard.csv')
        retention_export.to_csv(retention_path)
        print(f"  Cohort retention: {retention_path}")
        
        # 4. Average retention curve
        retention_curve_path = os.path.join(self.output_dir, 'retention_curve.csv')
        self.avg_retention.to_csv(retention_curve_path, index=False)
        print(f"  Retention curve: {retention_curve_path}")
        
        # 5. Cohort revenue for dashboard
        revenue_path = os.path.join(self.output_dir, 'cohort_revenue_dashboard.csv')
        self.cohort_revenue.to_csv(revenue_path, index=False)
        print(f"  Cohort revenue: {revenue_path}")
        
        # 6. Behavioral cluster summary
        cluster_path = os.path.join(self.output_dir, 'behavioral_clusters.csv')
        self.cluster_profile.to_csv(cluster_path, index=False)
        print(f"  Behavioral clusters: {cluster_path}")
        
        # 7. Summary metrics
        metrics = {
            'metric': [
                'total_customers', 'total_revenue', 'avg_recency',
                'avg_frequency', 'avg_monetary', 'num_segments',
                'num_cohorts', 'avg_retention_month1', 'avg_retention_month6'
            ],
            'value': [
                len(self.rfm_df),
                self.rfm_df['monetary'].sum(),
                self.rfm_df['recency'].mean(),
                self.rfm_df['frequency'].mean(),
                self.rfm_df['monetary'].mean(),
                len(self.segment_summary),
                len(self.cohort_revenue),
                self.avg_retention[self.avg_retention['month'] == 1]['avg_retention'].values[0] if 1 in self.avg_retention['month'].values else 0,
                self.avg_retention[self.avg_retention['month'] == 6]['avg_retention'].values[0] if 6 in self.avg_retention['month'].values else 0
            ]
        }
        metrics_df = pd.DataFrame(metrics)
        metrics_path = os.path.join(self.output_dir, 'segmentation_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"  Summary metrics: {metrics_path}")
        
        return segment_dashboard
    
    def create_visualizations(self):
        """Step 7: Create visualization charts for segments and cohort retention"""
        print("\n=== Step 7: Creating Visualizations ===")
        
        # 1. RFM Segment Distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Segment pie chart
        ax1 = axes[0, 0]
        segment_counts = self.rfm_df['rfm_segment'].value_counts()
        colors = sns.color_palette('Set2', len(segment_counts))
        wedges, texts, autotexts = ax1.pie(
            segment_counts.head(8), labels=segment_counts.head(8).index,
            autopct='%1.1f%%', colors=colors, startangle=90
        )
        ax1.set_title('RFM Segment Distribution', fontsize=12, fontweight='bold')
        
        # RFM score distribution
        ax2 = axes[0, 1]
        ax2.hist(self.rfm_df['RFM_total'], bins=13, color='steelblue', edgecolor='white')
        ax2.axvline(x=self.rfm_df['RFM_total'].mean(), color='red', linestyle='--',
                    label=f"Mean: {self.rfm_df['RFM_total'].mean():.1f}")
        ax2.set_xlabel('RFM Total Score', fontsize=10)
        ax2.set_ylabel('Customer Count', fontsize=10)
        ax2.set_title('Distribution of RFM Scores', fontsize=12, fontweight='bold')
        ax2.legend()
        
        # R-F-M scores heatmap
        ax3 = axes[1, 0]
        rf_matrix = self.rfm_df.groupby(['R_score', 'F_score']).size().unstack(fill_value=0)
        sns.heatmap(rf_matrix, annot=True, fmt='d', cmap='YlGnBu', ax=ax3)
        ax3.set_xlabel('Frequency Score', fontsize=10)
        ax3.set_ylabel('Recency Score', fontsize=10)
        ax3.set_title('R-F Score Distribution', fontsize=12, fontweight='bold')
        
        # Revenue by segment
        ax4 = axes[1, 1]
        segment_revenue = self.rfm_df.groupby('rfm_segment')['monetary'].sum().sort_values(ascending=True)
        colors = sns.color_palette('Blues_r', len(segment_revenue))
        ax4.barh(segment_revenue.index, segment_revenue.values, color=colors)
        ax4.set_xlabel('Total Revenue ($)', fontsize=10)
        ax4.set_title('Revenue by RFM Segment', fontsize=12, fontweight='bold')
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visuals_dir, 'rfm_segmentation.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: rfm_segmentation.png")
        
        # 2. Cohort Retention Heatmap
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Select recent cohorts for visibility
        recent_retention = self.retention_matrix.tail(12)
        if recent_retention.shape[1] > 12:
            recent_retention = recent_retention.iloc[:, :12]
        
        sns.heatmap(
            recent_retention,
            annot=True,
            fmt='.0f',
            cmap='YlGnBu',
            ax=ax,
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Retention Rate (%)'}
        )
        
        ax.set_xlabel('Months Since Signup', fontsize=12)
        ax.set_ylabel('Cohort (Signup Month)', fontsize=12)
        ax.set_title('Customer Retention by Cohort', fontsize=14, fontweight='bold')
        
        # Format y-axis labels
        y_labels = [str(label)[:7] for label in recent_retention.index]
        ax.set_yticklabels(y_labels, rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visuals_dir, 'cohort_retention_heatmap.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: cohort_retention_heatmap.png")
        
        # 3. Retention Curve
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        ax1.plot(self.avg_retention['month'], self.avg_retention['avg_retention'],
                 'b-o', linewidth=2, markersize=8)
        ax1.fill_between(self.avg_retention['month'], self.avg_retention['avg_retention'],
                         alpha=0.3)
        ax1.set_xlabel('Months Since Signup', fontsize=10)
        ax1.set_ylabel('Average Retention Rate (%)', fontsize=10)
        ax1.set_title('Customer Retention Curve', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)
        
        # Cohort revenue trend
        ax2 = axes[1]
        cohort_rev = self.cohort_revenue.tail(24)
        ax2.bar(range(len(cohort_rev)), cohort_rev['total_revenue'],
                color=sns.color_palette('Blues', len(cohort_rev)))
        ax2.set_xticks(range(0, len(cohort_rev), 3))
        ax2.set_xticklabels(cohort_rev['cohort'].iloc[::3], rotation=45, ha='right')
        ax2.set_xlabel('Cohort', fontsize=10)
        ax2.set_ylabel('Total Revenue ($)', fontsize=10)
        ax2.set_title('Revenue by Cohort', fontsize=12, fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visuals_dir, 'retention_and_revenue.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: retention_and_revenue.png")
        
        # 4. Behavioral Clusters
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cluster distribution
        ax1 = axes[0]
        cluster_counts = self.rfm_df['behavioral_segment'].value_counts()
        colors = sns.color_palette('viridis', len(cluster_counts))
        ax1.bar(cluster_counts.index, cluster_counts.values, color=colors)
        ax1.set_xlabel('Behavioral Segment', fontsize=10)
        ax1.set_ylabel('Customer Count', fontsize=10)
        ax1.set_title('Customer Distribution by Behavioral Segment', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=30)
        
        # RFM 3D-like scatter (Recency vs Frequency, sized by Monetary)
        ax2 = axes[1]
        scatter = ax2.scatter(
            self.rfm_df['recency'],
            self.rfm_df['frequency'],
            c=self.rfm_df['monetary'],
            s=self.rfm_df['monetary'] / 100 + 10,
            cmap='YlOrRd',
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax2, label='Monetary ($)')
        ax2.set_xlabel('Recency (days)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Customer Distribution (R-F-M)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visuals_dir, 'behavioral_segmentation.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: behavioral_segmentation.png")
        
        print("\nAll visualizations generated successfully!")
        
        return True
    
    def generate_summary_report(self):
        """Generate comprehensive segmentation report"""
        report_date = datetime.now().strftime('%Y-%m-%d')
        
        report = f"""
================================================================================
              CUSTOMER SEGMENTATION & COHORT ANALYSIS REPORT
                          {report_date}
================================================================================

EXECUTIVE SUMMARY
-----------------
Total Customers Analyzed: {len(self.rfm_df):,}
Total Revenue: ${self.rfm_df['monetary'].sum():,.2f}
Analysis Period: Full Customer Lifetime

RFM SEGMENTATION SUMMARY
------------------------
"""
        
        for _, row in self.segment_summary.iterrows():
            report += f"""
{row['segment'].upper()}:
  • Customers: {row['customer_count']:,} ({row['customer_pct']}%)
  • Revenue: ${row['total_revenue']:,.0f} ({row['revenue_pct']}%)
  • Avg RFM Score: {row['avg_rfm_score']:.1f}
  • Action: {row['recommended_action']}
"""
        
        report += f"""
BEHAVIORAL CLUSTERING
---------------------
Number of Clusters: {len(self.cluster_profile)}
"""
        
        for _, row in self.cluster_profile.iterrows():
            report += f"  • {row['behavioral_segment']}: {row['customer_count']:,} customers (${row['avg_revenue']:,.0f} avg revenue)\n"
        
        report += f"""
COHORT RETENTION ANALYSIS
-------------------------
Total Cohorts: {len(self.cohort_revenue)}
Average Month 1 Retention: {self.avg_retention[self.avg_retention['month'] == 1]['avg_retention'].values[0]:.1f}%
Average Month 6 Retention: {self.avg_retention[self.avg_retention['month'] == 6]['avg_retention'].values[0] if 6 in self.avg_retention['month'].values else 'N/A'}%
Average Month 12 Retention: {self.avg_retention[self.avg_retention['month'] == 12]['avg_retention'].values[0] if 12 in self.avg_retention['month'].values else 'N/A'}%

KEY INSIGHTS
------------
1. Top performing segment: {self.segment_summary.iloc[0]['segment']} with ${self.segment_summary.iloc[0]['total_revenue']:,.0f} revenue
2. Largest segment: {self.segment_summary.nlargest(1, 'customer_count').iloc[0]['segment']} ({self.segment_summary.nlargest(1, 'customer_count').iloc[0]['customer_count']:,} customers)
3. Highest LTV segment: {self.segment_summary.nlargest(1, 'avg_revenue').iloc[0]['segment']} (${self.segment_summary.nlargest(1, 'avg_revenue').iloc[0]['avg_revenue']:,.0f} avg)

================================================================================
                              END OF REPORT
================================================================================
"""
        
        report_path = os.path.join(self.output_dir, 'segmentation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to {report_path}")
        
        return report
    
    def run_segmentation_analysis(self):
        """Run complete segmentation and cohort analysis pipeline"""
        print("\n" + "="*60)
        print("CUSTOMER SEGMENTATION & COHORT ANALYSIS PIPELINE")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Step 1: RFM Segmentation
        self.perform_rfm_segmentation()
        
        # Step 2: Behavioral Clustering
        self.apply_behavioral_clustering()
        
        # Step 3: Generate Segment Labels
        self.generate_segment_labels()
        
        # Step 4: Cohort Retention Analysis
        self.perform_cohort_analysis()
        
        # Step 5: Cohort Revenue Analysis
        self.calculate_cohort_revenue()
        
        # Step 6: Export Dashboard Data
        self.export_dashboard_data()
        
        # Step 7: Create Visualizations
        self.create_visualizations()
        
        # Generate summary report
        self.generate_summary_report()
        
        # Save full RFM data
        full_path = os.path.join(self.output_dir, 'customer_segmentation_full.csv')
        self.rfm_df.to_csv(full_path, index=False)
        print(f"\nFull segmentation data saved to {full_path}")
        
        print("\n" + "="*60)
        print("SEGMENTATION & COHORT ANALYSIS COMPLETE")
        print("="*60)
        
        return {
            'rfm_df': self.rfm_df,
            'segment_summary': self.segment_summary,
            'cluster_profile': self.cluster_profile,
            'cohort_revenue': self.cohort_revenue,
            'retention_matrix': self.retention_matrix
        }


def main():
    """Main entry point"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'outputs')
    output_dir = os.path.join(base_dir, 'outputs', 'segmentation')
    visuals_dir = os.path.join(base_dir, 'visuals')
    
    analyzer = CustomerSegmentationAnalyzer(data_dir, output_dir, visuals_dir)
    results = analyzer.run_segmentation_analysis()
    
    return results


if __name__ == "__main__":
    main()
