"""
Cohort Analysis and RFM Analysis Module
Implements cohort retention analysis and RFM customer scoring
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns


class CohortRFMAnalyzer:
    """Cohort and RFM analysis for customer behavior insights"""
    
    def __init__(self, data_dir, output_dir, visuals_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.visuals_dir = visuals_dir
        
        self.customers_df = None
        self.transactions_df = None
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(visuals_dir, exist_ok=True)
    
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
    
    def perform_cohort_analysis(self):
        """Perform cohort retention analysis"""
        print("\n=== Performing Cohort Analysis ===")
        
        # Prepare data
        df = self.transactions_df.copy()
        customers = self.customers_df.copy()
        
        # Get only successful transactions
        df = df[df['transaction_status'] == 'Completed']
        
        # Add signup cohort
        customers['cohort'] = customers['signup_date'].dt.to_period('M')
        df = df.merge(customers[['customer_id', 'cohort']], on='customer_id', how='left')
        
        # Transaction month
        df['transaction_month'] = df['transaction_date'].dt.to_period('M')
        
        # Calculate cohort index (months since signup)
        df['cohort_index'] = (df['transaction_month'] - df['cohort']).apply(lambda x: x.n)
        
        # Create cohort table
        cohort_data = df.groupby(['cohort', 'cohort_index'])['customer_id'].nunique().reset_index()
        cohort_data.columns = ['cohort', 'cohort_index', 'customers']
        
        # Pivot to matrix
        cohort_matrix = cohort_data.pivot(
            index='cohort', 
            columns='cohort_index', 
            values='customers'
        )
        
        # Calculate retention rates
        cohort_sizes = cohort_matrix.iloc[:, 0]
        retention_matrix = cohort_matrix.divide(cohort_sizes, axis=0) * 100
        
        # Clean up
        retention_matrix = retention_matrix.fillna(0)
        
        # Save results
        cohort_matrix_str = cohort_matrix.copy()
        cohort_matrix_str.index = cohort_matrix_str.index.astype(str)
        cohort_matrix_str.to_csv(
            os.path.join(self.output_dir, 'cohort_counts.csv')
        )
        
        retention_matrix_str = retention_matrix.copy()
        retention_matrix_str.index = retention_matrix_str.index.astype(str)
        retention_matrix_str.to_csv(
            os.path.join(self.output_dir, 'cohort_retention.csv')
        )
        
        # Calculate average retention by cohort month
        avg_retention = retention_matrix.mean(axis=0).reset_index()
        avg_retention.columns = ['month', 'avg_retention']
        avg_retention.to_csv(
            os.path.join(self.output_dir, 'average_retention.csv'),
            index=False
        )
        
        print(f"\nCohort Matrix Shape: {cohort_matrix.shape}")
        print(f"Number of cohorts: {len(cohort_matrix)}")
        
        # Create visualizations
        self._plot_cohort_heatmap(retention_matrix)
        self._plot_retention_curve(avg_retention)
        
        return cohort_matrix, retention_matrix, avg_retention
    
    def _plot_cohort_heatmap(self, retention_matrix):
        """Create cohort retention heatmap"""
        # Select last 12 cohorts for visibility
        recent_cohorts = retention_matrix.tail(12)
        
        # Select first 12 months
        if recent_cohorts.shape[1] > 12:
            recent_cohorts = recent_cohorts.iloc[:, :12]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create heatmap
        sns.heatmap(
            recent_cohorts,
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
        y_labels = [str(label)[:7] for label in recent_cohorts.index]
        ax.set_yticklabels(y_labels, rotation=0)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, 'cohort_heatmap.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print("Saved cohort heatmap")
    
    def _plot_retention_curve(self, avg_retention):
        """Plot average retention curve"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(
            avg_retention['month'],
            avg_retention['avg_retention'],
            'b-o', linewidth=2, markersize=8
        )
        
        ax.fill_between(
            avg_retention['month'],
            avg_retention['avg_retention'],
            alpha=0.3
        )
        
        ax.set_xlabel('Months Since Signup', fontsize=12)
        ax.set_ylabel('Average Retention Rate (%)', fontsize=12)
        ax.set_title('Average Customer Retention Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        # Add annotations for key points
        if len(avg_retention) > 0:
            ax.annotate(
                f'{avg_retention.iloc[0]["avg_retention"]:.1f}%',
                xy=(avg_retention.iloc[0]['month'], avg_retention.iloc[0]['avg_retention']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold'
            )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, 'retention_curve.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print("Saved retention curve")
    
    def perform_rfm_analysis(self):
        """Perform RFM (Recency, Frequency, Monetary) analysis"""
        print("\n=== Performing RFM Analysis ===")
        
        # Get reference date
        reference_date = self.transactions_df['transaction_date'].max()
        
        # Filter successful transactions
        df = self.transactions_df[
            self.transactions_df['transaction_status'] == 'Completed'
        ].copy()
        
        # Calculate RFM metrics
        rfm = df.groupby('customer_id').agg({
            'transaction_date': lambda x: (reference_date - x.max()).days,  # Recency
            'transaction_id': 'count',  # Frequency
            'amount': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
        
        # Handle negative monetary values
        rfm['monetary'] = rfm['monetary'].clip(lower=0)
        
        # Calculate RFM scores (1-5 scale, 5 being best)
        # Recency: Lower is better, so reverse the scoring
        rfm['R_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
        
        # Frequency: Higher is better
        rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        
        # Monetary: Higher is better
        rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        
        # Combined RFM score
        rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
        rfm['RFM_total'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
        
        # Segment customers based on RFM scores
        rfm['segment'] = rfm.apply(self._assign_rfm_segment, axis=1)
        
        # Merge with customer info
        rfm = rfm.merge(
            self.customers_df[['customer_id', 'country', 'industry', 'subscription_type']],
            on='customer_id',
            how='left'
        )
        
        # Save results
        rfm.to_csv(
            os.path.join(self.output_dir, 'rfm_analysis.csv'),
            index=False
        )
        
        # Segment summary
        segment_summary = rfm.groupby('segment').agg({
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
        segment_summary = segment_summary.sort_values('avg_rfm_score', ascending=False)
        
        segment_summary.to_csv(
            os.path.join(self.output_dir, 'rfm_segment_summary.csv'),
            index=False
        )
        
        print("\nRFM Segment Summary:")
        print(segment_summary.to_string(index=False))
        
        # Create visualizations
        self._plot_rfm_analysis(rfm, segment_summary)
        
        return rfm, segment_summary
    
    def _assign_rfm_segment(self, row):
        """Assign customer segment based on RFM scores"""
        r, f, m = row['R_score'], row['F_score'], row['M_score']
        total = r + f + m
        
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 4 and f >= 3:
            return 'Loyal Customers'
        elif r >= 3 and f >= 1 and m >= 3:
            return 'Potential Loyalists'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r >= 3 and f >= 3 and m >= 2:
            return 'Promising'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        elif r <= 2 and f >= 4 and m >= 4:
            return "Can't Lose"
        elif r <= 2 and f <= 2:
            return 'Hibernating'
        elif r <= 3 and f <= 2:
            return 'About to Sleep'
        else:
            return 'Need Attention'
    
    def _plot_rfm_analysis(self, rfm, segment_summary):
        """Create RFM analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # RFM score distribution
        rfm['RFM_total'].hist(bins=13, ax=axes[0, 0], color='steelblue', edgecolor='white')
        axes[0, 0].set_xlabel('Total RFM Score', fontsize=10)
        axes[0, 0].set_ylabel('Count', fontsize=10)
        axes[0, 0].set_title('Distribution of RFM Scores', fontsize=12, fontweight='bold')
        
        # Segment distribution
        segment_counts = rfm['segment'].value_counts()
        colors = sns.color_palette('Set2', len(segment_counts))
        axes[0, 1].pie(
            segment_counts.head(8),
            labels=segment_counts.head(8).index,
            autopct='%1.1f%%',
            colors=colors
        )
        axes[0, 1].set_title('Customer Distribution by RFM Segment', fontsize=12, fontweight='bold')
        
        # Recency vs Frequency scatter
        scatter = axes[1, 0].scatter(
            rfm['recency'],
            rfm['frequency'],
            c=rfm['monetary'],
            cmap='YlOrRd',
            alpha=0.6,
            s=20
        )
        plt.colorbar(scatter, ax=axes[1, 0], label='Monetary ($)')
        axes[1, 0].set_xlabel('Recency (days)', fontsize=10)
        axes[1, 0].set_ylabel('Frequency', fontsize=10)
        axes[1, 0].set_title('RFM Scatter Plot', fontsize=12, fontweight='bold')
        
        # Revenue by segment
        top_segments = segment_summary.head(8).sort_values('total_revenue')
        axes[1, 1].barh(
            top_segments['segment'],
            top_segments['total_revenue'],
            color=sns.color_palette('Blues_r', len(top_segments))
        )
        axes[1, 1].set_xlabel('Total Revenue ($)', fontsize=10)
        axes[1, 1].set_title('Revenue by RFM Segment', fontsize=12, fontweight='bold')
        axes[1, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, 'rfm_analysis.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print("Saved RFM visualizations")
    
    def create_customer_lifecycle_analysis(self):
        """Analyze customer lifecycle stages"""
        print("\n=== Creating Customer Lifecycle Analysis ===")
        
        df = self.customers_df.copy()
        
        # Define lifecycle stages based on tenure and activity
        def assign_lifecycle_stage(row):
            tenure = row['customer_tenure_days']
            days_inactive = row['days_since_last_transaction']
            
            if tenure <= 90:
                return 'New (0-3 months)'
            elif tenure <= 180:
                if days_inactive <= 45:
                    return 'Growing (3-6 months)'
                else:
                    return 'Early Churn Risk'
            elif tenure <= 365:
                if days_inactive <= 45:
                    return 'Established (6-12 months)'
                else:
                    return 'Declining'
            else:
                if days_inactive <= 45:
                    return 'Mature (12+ months)'
                elif days_inactive <= 90:
                    return 'At Risk'
                else:
                    return 'Churned/Dormant'
        
        df['lifecycle_stage'] = df.apply(assign_lifecycle_stage, axis=1)
        
        # Lifecycle summary
        lifecycle_summary = df.groupby('lifecycle_stage').agg({
            'customer_id': 'count',
            'total_revenue': ['sum', 'mean'],
            'customer_ltv': 'mean',
            'customer_tenure_days': 'mean'
        }).reset_index()
        
        lifecycle_summary.columns = [
            'stage', 'customer_count', 'total_revenue', 'avg_revenue',
            'avg_ltv', 'avg_tenure_days'
        ]
        
        lifecycle_summary.to_csv(
            os.path.join(self.output_dir, 'lifecycle_analysis.csv'),
            index=False
        )
        
        # Save customer lifecycle assignments
        df[['customer_id', 'lifecycle_stage', 'customer_tenure_days', 'days_since_last_transaction']].to_csv(
            os.path.join(self.output_dir, 'customer_lifecycle.csv'),
            index=False
        )
        
        print("\nLifecycle Stage Summary:")
        print(lifecycle_summary.to_string(index=False))
        
        # Create visualization
        self._plot_lifecycle(lifecycle_summary)
        
        return lifecycle_summary
    
    def _plot_lifecycle(self, lifecycle_summary):
        """Plot lifecycle analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Customer count by stage
        sorted_data = lifecycle_summary.sort_values('customer_count', ascending=True)
        colors = sns.color_palette('viridis', len(sorted_data))
        
        axes[0].barh(sorted_data['stage'], sorted_data['customer_count'], color=colors)
        axes[0].set_xlabel('Customer Count', fontsize=10)
        axes[0].set_title('Customers by Lifecycle Stage', fontsize=12, fontweight='bold')
        
        # Revenue by stage
        sorted_revenue = lifecycle_summary.sort_values('total_revenue', ascending=True)
        axes[1].barh(sorted_revenue['stage'], sorted_revenue['total_revenue'], color=colors)
        axes[1].set_xlabel('Total Revenue ($)', fontsize=10)
        axes[1].set_title('Revenue by Lifecycle Stage', fontsize=12, fontweight='bold')
        axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, 'lifecycle_analysis.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print("Saved lifecycle visualizations")
    
    def run_cohort_rfm_analysis(self):
        """Run complete cohort and RFM analysis pipeline"""
        print("\n" + "="*60)
        print("COHORT AND RFM ANALYSIS MODULE")
        print("="*60)
        
        self.load_data()
        cohort_matrix, retention_matrix, avg_retention = self.perform_cohort_analysis()
        rfm, rfm_summary = self.perform_rfm_analysis()
        lifecycle = self.create_customer_lifecycle_analysis()
        
        print("\n=== Cohort and RFM Analysis Complete ===")
        
        return {
            'cohort_matrix': cohort_matrix,
            'retention_matrix': retention_matrix,
            'avg_retention': avg_retention,
            'rfm': rfm,
            'rfm_summary': rfm_summary,
            'lifecycle': lifecycle
        }


def main():
    """Main entry point"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'outputs')
    output_dir = os.path.join(base_dir, 'outputs', 'cohort_rfm')
    visuals_dir = os.path.join(base_dir, 'visuals')
    
    analyzer = CohortRFMAnalyzer(data_dir, output_dir, visuals_dir)
    results = analyzer.run_cohort_rfm_analysis()
    
    return results


if __name__ == "__main__":
    main()
