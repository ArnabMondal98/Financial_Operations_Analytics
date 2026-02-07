"""
Main Pipeline Runner
Orchestrates all analytics modules
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from data_generator import main as generate_data
from data_preprocessing import main as preprocess_data
from revenue_forecasting import main as run_forecasting
from churn_prediction import main as run_churn_prediction
from profitability_analysis import main as run_profitability
from cohort_rfm_analysis import main as run_cohort_rfm
from executive_reporting import main as run_reporting


def run_full_pipeline():
    """Run the complete financial operations analytics pipeline"""
    print("\n" + "="*80)
    print("   FINANCIAL OPERATIONS ANALYTICS - FULL PIPELINE EXECUTION")
    print("="*80)
    
    results = {}
    
    # Step 1: Generate synthetic data
    print("\n" + "-"*60)
    print("STEP 1: DATA GENERATION")
    print("-"*60)
    customers, transactions = generate_data()
    results['data_generation'] = {
        'customers': len(customers),
        'transactions': len(transactions)
    }
    
    # Step 2: Preprocess data
    print("\n" + "-"*60)
    print("STEP 2: DATA PREPROCESSING")
    print("-"*60)
    processed_customers, processed_transactions = preprocess_data()
    results['preprocessing'] = {
        'features_created': len(processed_customers.columns)
    }
    
    # Step 3: Revenue forecasting
    print("\n" + "-"*60)
    print("STEP 3: REVENUE FORECASTING")
    print("-"*60)
    forecast_results = run_forecasting()
    results['forecasting'] = {
        'best_model': forecast_results['best_model']
    }
    
    # Step 4: Churn prediction
    print("\n" + "-"*60)
    print("STEP 4: CHURN PREDICTION")
    print("-"*60)
    churn_results = run_churn_prediction()
    results['churn'] = {
        'best_model': churn_results['best_model'],
        'high_risk_customers': len(churn_results['predictions'][
            churn_results['predictions']['churn_risk_level'] == 'High'
        ])
    }
    
    # Step 5: Profitability analysis
    print("\n" + "-"*60)
    print("STEP 5: PROFITABILITY ANALYSIS")
    print("-"*60)
    profit_results = run_profitability()
    results['profitability'] = {
        'segments_identified': len(profit_results['segment_profile'])
    }
    
    # Step 6: Cohort and RFM analysis
    print("\n" + "-"*60)
    print("STEP 6: COHORT & RFM ANALYSIS")
    print("-"*60)
    cohort_results = run_cohort_rfm()
    results['cohort_rfm'] = {
        'rfm_segments': len(cohort_results['rfm_summary'])
    }
    
    # Step 7: Executive reporting
    print("\n" + "-"*60)
    print("STEP 7: EXECUTIVE REPORTING")
    print("-"*60)
    report_results = run_reporting()
    results['reporting'] = {
        'kpis_calculated': len(report_results['kpis'])
    }
    
    # Final summary
    print("\n" + "="*80)
    print("   PIPELINE EXECUTION COMPLETE")
    print("="*80)
    
    print("\n=== EXECUTION SUMMARY ===")
    print(f"✓ Generated {results['data_generation']['customers']} customers")
    print(f"✓ Generated {results['data_generation']['transactions']} transactions")
    print(f"✓ Created {results['preprocessing']['features_created']} features")
    print(f"✓ Best forecasting model: {results['forecasting']['best_model']}")
    print(f"✓ Best churn model: {results['churn']['best_model']}")
    print(f"✓ High-risk customers identified: {results['churn']['high_risk_customers']}")
    print(f"✓ Customer segments: {results['profitability']['segments_identified']}")
    print(f"✓ RFM segments: {results['cohort_rfm']['rfm_segments']}")
    print(f"✓ Executive KPIs: {results['reporting']['kpis_calculated']}")
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    print(f"\n=== OUTPUT LOCATIONS ===")
    print(f"Data:       {os.path.join(base_dir, 'data')}")
    print(f"Outputs:    {os.path.join(base_dir, 'outputs')}")
    print(f"Models:     {os.path.join(base_dir, 'models')}")
    print(f"Visuals:    {os.path.join(base_dir, 'visuals')}")
    print(f"Dashboard:  {os.path.join(base_dir, 'dashboard')}")
    print(f"Notebooks:  {os.path.join(base_dir, 'notebooks')}")
    
    return results


if __name__ == "__main__":
    run_full_pipeline()
