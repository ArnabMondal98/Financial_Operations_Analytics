"""
Customer Retention Recommendation Engine
Generates personalized retention actions based on churn risk
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class RetentionRecommendationEngine:
    """Generate retention recommendations based on churn risk and customer profile"""
    
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define retention strategies by risk level
        self.retention_strategies = {
            'High': {
                'primary_action': 'Immediate Retention Call',
                'secondary_action': 'Exclusive Discount Offer',
                'urgency': 'Critical',
                'timeline': 'Within 24-48 hours',
                'discount_range': (20, 30),
                'channel': 'Phone + Email'
            },
            'Medium': {
                'primary_action': 'Targeted Engagement Email',
                'secondary_action': 'Personalized Product Demo',
                'urgency': 'High',
                'timeline': 'Within 1 week',
                'discount_range': (10, 15),
                'channel': 'Email + In-App'
            },
            'Low': {
                'primary_action': 'Loyalty Rewards Program',
                'secondary_action': 'Feature Upgrade Offer',
                'urgency': 'Normal',
                'timeline': 'Within 30 days',
                'discount_range': (5, 10),
                'channel': 'Email + Newsletter'
            }
        }
        
        # Action templates based on customer profile
        self.action_templates = {
            'High': {
                'Premium': [
                    'Schedule VIP retention call with account manager',
                    'Offer {discount}% discount on annual renewal',
                    'Provide complimentary premium support for 3 months',
                    'Invite to exclusive executive roundtable event'
                ],
                'Enterprise': [
                    'Escalate to enterprise success team immediately',
                    'Offer {discount}% discount + dedicated support',
                    'Schedule executive business review meeting',
                    'Provide custom integration support package'
                ],
                'Professional': [
                    'Initiate proactive outreach call',
                    'Offer {discount}% discount on next renewal',
                    'Provide free upgrade trial to Enterprise tier',
                    'Send personalized value demonstration'
                ],
                'Basic': [
                    'Send win-back email with special offer',
                    'Offer {discount}% discount + free tier upgrade trial',
                    'Provide onboarding assistance call',
                    'Share success stories from similar customers'
                ]
            },
            'Medium': {
                'Premium': [
                    'Send personalized engagement email from CSM',
                    'Offer early access to new premium features',
                    'Invite to product feedback session',
                    'Share quarterly ROI report'
                ],
                'Enterprise': [
                    'Schedule quarterly business review',
                    'Send targeted feature adoption email',
                    'Offer {discount}% loyalty discount',
                    'Provide advanced training webinar access'
                ],
                'Professional': [
                    'Send re-engagement email campaign',
                    'Offer {discount}% discount on upgrade',
                    'Share relevant case studies',
                    'Provide product tips and best practices'
                ],
                'Basic': [
                    'Send targeted email with usage tips',
                    'Offer limited-time upgrade promotion',
                    'Share customer success stories',
                    'Invite to educational webinar'
                ]
            },
            'Low': {
                'Premium': [
                    'Enroll in VIP loyalty rewards program',
                    'Offer referral bonus program',
                    'Provide beta access to new features',
                    'Send quarterly appreciation gift'
                ],
                'Enterprise': [
                    'Enroll in enterprise loyalty program',
                    'Offer multi-year discount incentive',
                    'Provide co-marketing opportunities',
                    'Invite to customer advisory board'
                ],
                'Professional': [
                    'Enroll in loyalty points program',
                    'Offer anniversary discount',
                    'Provide certification program access',
                    'Send feature highlight newsletter'
                ],
                'Basic': [
                    'Enroll in basic rewards program',
                    'Offer loyalty discount on upgrade',
                    'Send monthly tips newsletter',
                    'Provide community forum access'
                ]
            }
        }
    
    def load_data(self):
        """Load churn predictions and customer data"""
        print("Loading customer and churn data...")
        
        # Load churn predictions
        self.churn_df = pd.read_csv(
            os.path.join(self.data_dir, 'churn', 'churn_predictions.csv')
        )
        
        # Load processed customers for additional features
        self.customers_df = pd.read_csv(
            os.path.join(self.data_dir, 'processed_customers.csv')
        )
        
        # Merge data
        self.data = self.churn_df.merge(
            self.customers_df[[
                'customer_id', 'transaction_frequency', 'avg_monthly_revenue',
                'days_since_last_transaction', 'payment_failure_rate',
                'successful_transactions', 'failed_transactions'
            ]],
            on='customer_id',
            how='left'
        )
        
        print(f"Loaded {len(self.data)} customers")
        return self.data
    
    def categorize_risk(self):
        """Ensure risk categorization (already done in churn prediction)"""
        print("\nVerifying risk categorization...")
        
        risk_counts = self.data['churn_risk_level'].value_counts()
        print("Risk Distribution:")
        for risk, count in risk_counts.items():
            pct = count / len(self.data) * 100
            print(f"  {risk}: {count} customers ({pct:.1f}%)")
        
        return risk_counts
    
    def calculate_customer_value_score(self):
        """Calculate customer value score for prioritization"""
        print("\nCalculating customer value scores...")
        
        # Normalize revenue to 0-100 scale
        max_revenue = self.data['total_revenue'].max()
        self.data['revenue_score'] = (self.data['total_revenue'] / max_revenue) * 100
        
        # Calculate engagement score
        max_freq = self.data['transaction_frequency'].max()
        self.data['engagement_score'] = (self.data['transaction_frequency'] / max_freq) * 100
        
        # Calculate tenure score
        max_tenure = self.data['customer_tenure_days'].max()
        self.data['tenure_score'] = (self.data['customer_tenure_days'] / max_tenure) * 100
        
        # Composite value score (weighted)
        self.data['value_score'] = (
            self.data['revenue_score'] * 0.5 +
            self.data['engagement_score'] * 0.3 +
            self.data['tenure_score'] * 0.2
        )
        
        # Priority ranking
        self.data['priority_rank'] = self.data.groupby('churn_risk_level')['value_score'].rank(
            ascending=False, method='dense'
        ).astype(int)
        
        return self.data
    
    def generate_recommendations(self):
        """Generate personalized retention recommendations"""
        print("\nGenerating retention recommendations...")
        
        recommendations = []
        
        for _, customer in self.data.iterrows():
            risk_level = customer['churn_risk_level']
            subscription = customer['subscription_type']
            
            # Get strategy based on risk level
            strategy = self.retention_strategies[risk_level]
            
            # Get action templates
            templates = self.action_templates[risk_level][subscription]
            
            # Calculate personalized discount
            discount_min, discount_max = strategy['discount_range']
            # Higher value customers get better discounts
            value_percentile = customer['value_score'] / 100
            discount = int(discount_min + (discount_max - discount_min) * value_percentile)
            
            # Format actions with discount
            actions = [action.format(discount=discount) for action in templates]
            
            # Determine specific action based on customer profile
            if customer['payment_failure_rate'] > 0.1:
                specific_action = "Address payment issues - offer alternative payment methods"
            elif customer['days_since_last_transaction'] > 90:
                specific_action = f"Re-engagement campaign - {customer['days_since_last_transaction']} days inactive"
            elif customer['transaction_frequency'] < 0.5:
                specific_action = "Increase engagement - schedule product demo"
            else:
                specific_action = actions[0]
            
            recommendation = {
                'customer_id': customer['customer_id'],
                'country': customer['country'],
                'industry': customer['industry'],
                'subscription_type': subscription,
                'monthly_fee': customer['monthly_fee'],
                'total_revenue': customer['total_revenue'],
                'churn_probability': customer['churn_probability'],
                'churn_risk_level': risk_level,
                'value_score': round(customer['value_score'], 2),
                'priority_rank': customer['priority_rank'],
                'urgency': strategy['urgency'],
                'action_timeline': strategy['timeline'],
                'contact_channel': strategy['channel'],
                'primary_action': strategy['primary_action'],
                'secondary_action': strategy['secondary_action'],
                'specific_recommendation': specific_action,
                'recommended_discount_pct': discount,
                'action_1': actions[0],
                'action_2': actions[1],
                'action_3': actions[2],
                'action_4': actions[3]
            }
            
            recommendations.append(recommendation)
        
        self.recommendations_df = pd.DataFrame(recommendations)
        
        # Sort by risk level priority and value score
        risk_order = {'High': 0, 'Medium': 1, 'Low': 2}
        self.recommendations_df['risk_sort'] = self.recommendations_df['churn_risk_level'].map(risk_order)
        self.recommendations_df = self.recommendations_df.sort_values(
            ['risk_sort', 'value_score'],
            ascending=[True, False]
        ).drop('risk_sort', axis=1)
        
        print(f"Generated {len(self.recommendations_df)} recommendations")
        
        return self.recommendations_df
    
    def create_action_summary(self):
        """Create summary of recommended actions"""
        print("\nCreating action summary...")
        
        # Summary by risk level
        risk_summary = self.recommendations_df.groupby('churn_risk_level').agg({
            'customer_id': 'count',
            'total_revenue': 'sum',
            'value_score': 'mean',
            'recommended_discount_pct': 'mean'
        }).reset_index()
        risk_summary.columns = [
            'risk_level', 'customer_count', 'total_revenue_at_risk',
            'avg_value_score', 'avg_recommended_discount'
        ]
        
        # Reorder
        risk_order = {'High': 0, 'Medium': 1, 'Low': 2}
        risk_summary['sort_order'] = risk_summary['risk_level'].map(risk_order)
        risk_summary = risk_summary.sort_values('sort_order').drop('sort_order', axis=1)
        
        # Summary by subscription type within each risk level
        sub_summary = self.recommendations_df.groupby(
            ['churn_risk_level', 'subscription_type']
        ).agg({
            'customer_id': 'count',
            'total_revenue': 'sum'
        }).reset_index()
        sub_summary.columns = ['risk_level', 'subscription_type', 'customer_count', 'revenue']
        
        return risk_summary, sub_summary
    
    def export_recommendations(self):
        """Export recommendations to CSV files"""
        print("\nExporting recommendations...")
        
        # Full recommendations
        full_path = os.path.join(self.output_dir, 'retention_recommendations_full.csv')
        self.recommendations_df.to_csv(full_path, index=False)
        print(f"  Full recommendations: {full_path}")
        
        # High-risk action list (for immediate action)
        high_risk = self.recommendations_df[
            self.recommendations_df['churn_risk_level'] == 'High'
        ].copy()
        high_risk_path = os.path.join(self.output_dir, 'high_risk_action_list.csv')
        high_risk.to_csv(high_risk_path, index=False)
        print(f"  High-risk action list: {high_risk_path}")
        
        # Medium-risk engagement list
        medium_risk = self.recommendations_df[
            self.recommendations_df['churn_risk_level'] == 'Medium'
        ].copy()
        medium_risk_path = os.path.join(self.output_dir, 'medium_risk_engagement_list.csv')
        medium_risk.to_csv(medium_risk_path, index=False)
        print(f"  Medium-risk engagement list: {medium_risk_path}")
        
        # Low-risk loyalty list
        low_risk = self.recommendations_df[
            self.recommendations_df['churn_risk_level'] == 'Low'
        ].copy()
        low_risk_path = os.path.join(self.output_dir, 'low_risk_loyalty_list.csv')
        low_risk.to_csv(low_risk_path, index=False)
        print(f"  Low-risk loyalty list: {low_risk_path}")
        
        # Action summary
        risk_summary, sub_summary = self.create_action_summary()
        
        summary_path = os.path.join(self.output_dir, 'retention_action_summary.csv')
        risk_summary.to_csv(summary_path, index=False)
        print(f"  Action summary: {summary_path}")
        
        # Executive summary report
        self._create_executive_summary(risk_summary)
        
        return full_path
    
    def _create_executive_summary(self, risk_summary):
        """Create executive summary report"""
        report_date = datetime.now().strftime('%Y-%m-%d')
        
        high_risk = self.recommendations_df[self.recommendations_df['churn_risk_level'] == 'High']
        medium_risk = self.recommendations_df[self.recommendations_df['churn_risk_level'] == 'Medium']
        low_risk = self.recommendations_df[self.recommendations_df['churn_risk_level'] == 'Low']
        
        report = f"""
================================================================================
              CUSTOMER RETENTION RECOMMENDATION REPORT
                          {report_date}
================================================================================

EXECUTIVE SUMMARY
-----------------
Total Customers Analyzed: {len(self.recommendations_df):,}
Total Revenue Under Management: ${self.recommendations_df['total_revenue'].sum():,.2f}

RISK SEGMENTATION
-----------------
HIGH RISK (Immediate Action Required)
  • Customers: {len(high_risk):,}
  • Revenue at Risk: ${high_risk['total_revenue'].sum():,.2f}
  • Recommended Actions: Retention calls + Discount offers (20-30%)
  • Timeline: Within 24-48 hours
  
MEDIUM RISK (Proactive Engagement Needed)
  • Customers: {len(medium_risk):,}
  • Revenue at Risk: ${medium_risk['total_revenue'].sum():,.2f}
  • Recommended Actions: Targeted email campaigns + Product demos
  • Timeline: Within 1 week

LOW RISK (Loyalty Building)
  • Customers: {len(low_risk):,}
  • Revenue: ${low_risk['total_revenue'].sum():,.2f}
  • Recommended Actions: Loyalty rewards + Feature upgrades
  • Timeline: Within 30 days

PRIORITY ACTIONS
----------------
Top 10 High-Value At-Risk Customers Requiring Immediate Attention:
"""
        
        top_10 = high_risk.nlargest(10, 'value_score')
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            report += f"""
{i}. {row['customer_id']} ({row['subscription_type']})
   Revenue: ${row['total_revenue']:,.2f} | Churn Prob: {row['churn_probability']:.1%}
   Action: {row['specific_recommendation']}
"""
        
        report += f"""
RESOURCE ALLOCATION
-------------------
• Retention Team Calls Needed: {len(high_risk)} (High Risk customers)
• Email Campaigns: {len(medium_risk)} recipients (Medium Risk)
• Loyalty Program Enrollments: {len(low_risk)} customers (Low Risk)

ESTIMATED IMPACT
----------------
If 50% of high-risk customers are retained:
  • Customers Saved: {len(high_risk) // 2}
  • Revenue Protected: ${high_risk['total_revenue'].sum() / 2:,.2f}

================================================================================
                            END OF REPORT
================================================================================
"""
        
        report_path = os.path.join(self.output_dir, 'retention_executive_summary.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"  Executive summary: {report_path}")
        
        return report
    
    def run_recommendation_engine(self):
        """Run the complete recommendation engine"""
        print("\n" + "="*60)
        print("CUSTOMER RETENTION RECOMMENDATION ENGINE")
        print("="*60)
        
        self.load_data()
        self.categorize_risk()
        self.calculate_customer_value_score()
        self.generate_recommendations()
        self.export_recommendations()
        
        print("\n" + "="*60)
        print("RECOMMENDATION ENGINE COMPLETE")
        print("="*60)
        
        # Print summary
        print("\n=== ACTION SUMMARY ===")
        for risk_level in ['High', 'Medium', 'Low']:
            subset = self.recommendations_df[self.recommendations_df['churn_risk_level'] == risk_level]
            strategy = self.retention_strategies[risk_level]
            print(f"\n{risk_level.upper()} RISK ({len(subset)} customers)")
            print(f"  Primary Action: {strategy['primary_action']}")
            print(f"  Timeline: {strategy['timeline']}")
            print(f"  Revenue at Stake: ${subset['total_revenue'].sum():,.2f}")
        
        return self.recommendations_df


def main():
    """Main entry point"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'outputs')
    output_dir = os.path.join(base_dir, 'outputs', 'retention')
    
    engine = RetentionRecommendationEngine(data_dir, output_dir)
    recommendations = engine.run_recommendation_engine()
    
    return recommendations


if __name__ == "__main__":
    main()
