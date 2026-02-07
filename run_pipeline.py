#!/usr/bin/env python3
"""
Financial Operations Analytics - Production Pipeline
=====================================================
Automated end-to-end analytics pipeline for financial operations.

This script orchestrates the complete analytics workflow:
1. Data Ingestion - Load or generate synthetic financial data
2. Preprocessing - Clean, validate, and engineer features
3. Forecasting - Revenue prediction using Prophet and ARIMA
4. Churn Prediction - ML-based customer churn modeling
5. Profitability Analysis - Customer profitability and segmentation
6. Customer Segmentation - RFM and behavioral clustering
7. Retention Engine - Personalized retention recommendations
8. Unified Dashboard - Executive reporting and Power BI exports

Usage:
    python run_pipeline.py [--generate-data] [--skip-forecasting] [--output-dir PATH]

Author: Financial Operations Analytics Team
Version: 1.0.0
"""

import os
import sys
import argparse
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def print_banner():
    """Print pipeline banner"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            FINANCIAL OPERATIONS ANALYTICS PIPELINE v1.0                      ║
║                                                                              ║
║   Revenue Forecasting | Churn Prediction | Profitability | Segmentation     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def print_step(step_num, total_steps, step_name):
    """Print step header"""
    print(f"\n{'='*70}")
    print(f"  STEP {step_num}/{total_steps}: {step_name}")
    print(f"{'='*70}")


def run_pipeline(generate_data=True, skip_forecasting=False, output_dir=None):
    """
    Run the complete financial operations analytics pipeline.
    
    Args:
        generate_data: Whether to generate synthetic data (default: True)
        skip_forecasting: Skip time-intensive forecasting step (default: False)
        output_dir: Custom output directory (optional)
    
    Returns:
        dict: Pipeline results and metrics
    """
    start_time = time.time()
    results = {}
    
    print_banner()
    print(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  - Generate Data: {generate_data}")
    print(f"  - Skip Forecasting: {skip_forecasting}")
    print(f"  - Output Directory: {output_dir or 'default'}")
    
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    outputs_dir = output_dir or os.path.join(base_dir, 'outputs')
    
    total_steps = 8 if not skip_forecasting else 7
    current_step = 0
    
    # ========================================================================
    # STEP 1: DATA INGESTION
    # ========================================================================
    current_step += 1
    print_step(current_step, total_steps, "DATA INGESTION")
    
    if generate_data:
        from data_generator import FinancialDataGenerator
        
        generator = FinancialDataGenerator(
            start_date='2021-01-01',
            end_date='2024-12-31'
        )
        customers_df, transactions_df = generator.generate_all(data_dir, num_customers=2500)
        
        results['data_ingestion'] = {
            'customers': len(customers_df),
            'transactions': len(transactions_df),
            'status': 'Generated'
        }
    else:
        import pandas as pd
        customers_df = pd.read_csv(os.path.join(data_dir, 'customers.csv'))
        transactions_df = pd.read_csv(os.path.join(data_dir, 'transactions.csv'))
        
        results['data_ingestion'] = {
            'customers': len(customers_df),
            'transactions': len(transactions_df),
            'status': 'Loaded'
        }
    
    print(f"\n✓ Data ingestion complete: {results['data_ingestion']['customers']} customers, "
          f"{results['data_ingestion']['transactions']} transactions")
    
    # ========================================================================
    # STEP 2: DATA PREPROCESSING
    # ========================================================================
    current_step += 1
    print_step(current_step, total_steps, "DATA PREPROCESSING")
    
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor(data_dir)
    processed_customers, processed_transactions = preprocessor.run_pipeline(outputs_dir)
    
    results['preprocessing'] = {
        'customer_features': len(processed_customers.columns),
        'transaction_features': len(processed_transactions.columns),
        'status': 'Complete'
    }
    
    print(f"\n✓ Preprocessing complete: {results['preprocessing']['customer_features']} customer features")
    
    # ========================================================================
    # STEP 3: REVENUE FORECASTING
    # ========================================================================
    if not skip_forecasting:
        current_step += 1
        print_step(current_step, total_steps, "REVENUE FORECASTING")
        
        from revenue_forecasting import RevenueForecaster
        
        forecaster = RevenueForecaster(
            data_dir=outputs_dir,
            output_dir=os.path.join(outputs_dir, 'forecasts'),
            visuals_dir=os.path.join(base_dir, 'visuals')
        )
        forecast_results = forecaster.run_forecasting(forecast_periods=12)
        
        results['forecasting'] = {
            'best_model': forecast_results['best_model'],
            'forecast_months': 12,
            'status': 'Complete'
        }
        
        print(f"\n✓ Forecasting complete: Best model = {results['forecasting']['best_model']}")
    
    # ========================================================================
    # STEP 4: CHURN PREDICTION
    # ========================================================================
    current_step += 1
    print_step(current_step, total_steps, "CHURN PREDICTION")
    
    from churn_prediction import ChurnPredictor
    
    predictor = ChurnPredictor(
        data_dir=outputs_dir,
        output_dir=os.path.join(outputs_dir, 'churn'),
        models_dir=os.path.join(base_dir, 'models'),
        visuals_dir=os.path.join(base_dir, 'visuals')
    )
    churn_results = predictor.run_churn_prediction()
    
    high_risk = len(churn_results['predictions'][
        churn_results['predictions']['churn_risk_level'] == 'High'
    ])
    
    results['churn'] = {
        'best_model': churn_results['best_model'],
        'high_risk_customers': high_risk,
        'status': 'Complete'
    }
    
    print(f"\n✓ Churn prediction complete: {high_risk} high-risk customers identified")
    
    # ========================================================================
    # STEP 5: PROFITABILITY ANALYSIS
    # ========================================================================
    current_step += 1
    print_step(current_step, total_steps, "PROFITABILITY ANALYSIS")
    
    from profitability_analysis import ProfitabilityAnalyzer
    
    profit_analyzer = ProfitabilityAnalyzer(
        data_dir=outputs_dir,
        output_dir=os.path.join(outputs_dir, 'profitability'),
        visuals_dir=os.path.join(base_dir, 'visuals')
    )
    profit_results = profit_analyzer.run_profitability_analysis()
    
    results['profitability'] = {
        'total_profit': profit_results['profitability_df']['gross_profit'].sum(),
        'profitable_customers': len(profit_results['profitability_df'][
            profit_results['profitability_df']['gross_profit'] > 0
        ]),
        'status': 'Complete'
    }
    
    print(f"\n✓ Profitability analysis complete: ${results['profitability']['total_profit']:,.2f} total profit")
    
    # ========================================================================
    # STEP 6: CUSTOMER SEGMENTATION
    # ========================================================================
    current_step += 1
    print_step(current_step, total_steps, "CUSTOMER SEGMENTATION")
    
    from customer_segmentation import CustomerSegmentationAnalyzer
    
    segment_analyzer = CustomerSegmentationAnalyzer(
        data_dir=outputs_dir,
        output_dir=os.path.join(outputs_dir, 'segmentation'),
        visuals_dir=os.path.join(base_dir, 'visuals')
    )
    segment_results = segment_analyzer.run_segmentation_analysis()
    
    results['segmentation'] = {
        'rfm_segments': len(segment_results['segment_summary']),
        'behavioral_clusters': len(segment_results['cluster_profile']),
        'status': 'Complete'
    }
    
    print(f"\n✓ Segmentation complete: {results['segmentation']['rfm_segments']} RFM segments")
    
    # ========================================================================
    # STEP 7: RETENTION RECOMMENDATIONS
    # ========================================================================
    current_step += 1
    print_step(current_step, total_steps, "RETENTION RECOMMENDATIONS")
    
    from retention_recommendations import RetentionRecommendationEngine
    
    retention_engine = RetentionRecommendationEngine(
        data_dir=outputs_dir,
        output_dir=os.path.join(outputs_dir, 'retention')
    )
    retention_results = retention_engine.run_recommendation_engine()
    
    results['retention'] = {
        'recommendations_generated': len(retention_results),
        'status': 'Complete'
    }
    
    print(f"\n✓ Retention engine complete: {results['retention']['recommendations_generated']} recommendations")
    
    # ========================================================================
    # STEP 8: UNIFIED DASHBOARD EXPORT
    # ========================================================================
    current_step += 1
    print_step(current_step, total_steps, "UNIFIED DASHBOARD EXPORT")
    
    from unified_analytics import UnifiedAnalyticsDashboard
    
    dashboard = UnifiedAnalyticsDashboard(
        base_dir=base_dir,
        output_dir=os.path.join(outputs_dir, 'unified'),
        visuals_dir=os.path.join(base_dir, 'visuals')
    )
    dashboard_results = dashboard.run_unified_analytics()
    
    results['dashboard'] = {
        'master_dataset_rows': len(dashboard_results['master_df']),
        'master_dataset_cols': len(dashboard_results['master_df'].columns),
        'kpis_calculated': len(dashboard_results['kpis']),
        'status': 'Complete'
    }
    
    print(f"\n✓ Dashboard export complete: {results['dashboard']['kpis_calculated']} KPIs generated")
    
    # ========================================================================
    # PIPELINE SUMMARY
    # ========================================================================
    elapsed_time = time.time() - start_time
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PIPELINE EXECUTION COMPLETE                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Execution Time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)
Completed At:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESULTS SUMMARY
---------------
✓ Data Ingestion:     {results['data_ingestion']['customers']:,} customers, {results['data_ingestion']['transactions']:,} transactions
✓ Preprocessing:      {results['preprocessing']['customer_features']} features engineered
✓ Forecasting:        {"Best model: " + results.get('forecasting', {}).get('best_model', 'Skipped') if not skip_forecasting else 'Skipped'}
✓ Churn Prediction:   {results['churn']['high_risk_customers']} high-risk customers identified
✓ Profitability:      ${results['profitability']['total_profit']:,.2f} total profit
✓ Segmentation:       {results['segmentation']['rfm_segments']} customer segments
✓ Retention:          {results['retention']['recommendations_generated']} personalized recommendations
✓ Dashboard:          {results['dashboard']['kpis_calculated']} KPIs exported

OUTPUT LOCATIONS
----------------
• Data:       {os.path.join(base_dir, 'data')}
• Outputs:    {outputs_dir}
• Models:     {os.path.join(base_dir, 'models')}
• Visuals:    {os.path.join(base_dir, 'visuals')}
• Dashboard:  {os.path.join(outputs_dir, 'unified', 'dashboard')}
    """)
    
    results['execution_time'] = elapsed_time
    results['status'] = 'Success'
    
    return results


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Financial Operations Analytics Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run full pipeline with data generation
  python run_pipeline.py --no-generate      # Use existing data files
  python run_pipeline.py --skip-forecasting # Skip time-intensive forecasting
  python run_pipeline.py --output-dir ./out # Custom output directory
        """
    )
    
    parser.add_argument(
        '--no-generate', '-n',
        action='store_true',
        help='Use existing data files instead of generating new ones'
    )
    
    parser.add_argument(
        '--skip-forecasting', '-s',
        action='store_true',
        help='Skip the forecasting step (saves time for testing)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Custom output directory path'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_pipeline(
            generate_data=not args.no_generate,
            skip_forecasting=args.skip_forecasting,
            output_dir=args.output_dir
        )
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
