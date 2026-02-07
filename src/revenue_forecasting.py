"""
Revenue Forecasting Module
Implements Prophet and ARIMA models for revenue forecasting
Includes time series decomposition for seasonality and trend analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Prophet
from prophet import Prophet

# ARIMA and Time Series Analysis
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns


class RevenueForecaster:
    """Revenue forecasting using Prophet and ARIMA"""
    
    def __init__(self, data_dir, output_dir, visuals_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.visuals_dir = visuals_dir
        self.monthly_revenue = None
        self.prophet_model = None
        self.arima_model = None
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(visuals_dir, exist_ok=True)
    
    def load_data(self):
        """Load monthly revenue data"""
        print("Loading revenue data...")
        
        self.monthly_revenue = pd.read_csv(
            os.path.join(self.data_dir, 'monthly_revenue.csv')
        )
        self.monthly_revenue['date'] = pd.to_datetime(self.monthly_revenue['date'])
        self.monthly_revenue = self.monthly_revenue.sort_values('date')
        
        print(f"Loaded {len(self.monthly_revenue)} months of revenue data")
        print(f"Date range: {self.monthly_revenue['date'].min()} to {self.monthly_revenue['date'].max()}")
        
        return self.monthly_revenue
    
    def aggregate_monthly_revenue(self):
        """Step 1: Aggregate transaction data into monthly revenue"""
        print("\n=== Step 1: Aggregating Monthly Revenue ===")
        
        # Load raw transactions if needed
        transactions_path = os.path.join(os.path.dirname(self.data_dir), 'data', 'transactions.csv')
        if os.path.exists(transactions_path):
            raw_txns = pd.read_csv(transactions_path)
            raw_txns['transaction_date'] = pd.to_datetime(raw_txns['transaction_date'])
            
            # Filter successful transactions
            successful = raw_txns[raw_txns['transaction_status'] == 'Completed']
            
            # Aggregate by month
            monthly_agg = successful.groupby(
                successful['transaction_date'].dt.to_period('M')
            ).agg({
                'amount': ['sum', 'count', 'mean'],
                'customer_id': 'nunique'
            }).reset_index()
            
            monthly_agg.columns = ['month', 'total_revenue', 'transaction_count', 
                                   'avg_transaction', 'unique_customers']
            monthly_agg['month'] = monthly_agg['month'].dt.to_timestamp()
            
            print(f"Aggregated {len(monthly_agg)} months of data")
            print(f"Total Revenue: ${monthly_agg['total_revenue'].sum():,.2f}")
            print(f"Avg Monthly Revenue: ${monthly_agg['total_revenue'].mean():,.2f}")
            
            # Save aggregation
            agg_path = os.path.join(self.output_dir, 'monthly_revenue_aggregated.csv')
            monthly_agg.to_csv(agg_path, index=False)
            print(f"Saved aggregation to {agg_path}")
            
            self.monthly_agg = monthly_agg
        
        return self.monthly_revenue
    
    def perform_time_series_decomposition(self):
        """Step 2: Decompose time series to detect seasonality and trends"""
        print("\n=== Step 2: Time Series Decomposition ===")
        
        # Prepare series for decomposition
        ts_data = self.monthly_revenue.set_index('date')['revenue']
        
        # Perform decomposition (multiplicative for revenue data with growth)
        # Need at least 2 full periods for decomposition
        if len(ts_data) >= 24:
            decomposition = seasonal_decompose(ts_data, model='multiplicative', period=12)
        else:
            decomposition = seasonal_decompose(ts_data, model='additive', period=min(12, len(ts_data)//2))
        
        # Extract components
        self.trend = decomposition.trend
        self.seasonal = decomposition.seasonal
        self.residual = decomposition.resid
        
        # Calculate seasonality indices
        seasonal_indices = self.seasonal.groupby(self.seasonal.index.month).mean()
        
        print("\nSeasonality Indices by Month:")
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month, idx in seasonal_indices.items():
            impact = "↑ Higher" if idx > 1 else "↓ Lower"
            print(f"  {month_names[month-1]}: {idx:.3f} ({impact})")
        
        # Save decomposition results
        decomp_df = pd.DataFrame({
            'date': ts_data.index,
            'observed': ts_data.values,
            'trend': self.trend.values,
            'seasonal': self.seasonal.values,
            'residual': self.residual.values
        })
        decomp_path = os.path.join(self.output_dir, 'time_series_decomposition.csv')
        decomp_df.to_csv(decomp_path, index=False)
        
        # Save seasonal indices
        seasonal_df = pd.DataFrame({
            'month': list(range(1, 13)),
            'month_name': month_names,
            'seasonal_index': [seasonal_indices.get(m, 1.0) for m in range(1, 13)]
        })
        seasonal_path = os.path.join(self.output_dir, 'seasonal_indices.csv')
        seasonal_df.to_csv(seasonal_path, index=False)
        
        # Create decomposition visualization
        self._plot_decomposition(ts_data, decomposition)
        
        # Trend analysis
        trend_clean = self.trend.dropna()
        if len(trend_clean) > 1:
            trend_growth = (trend_clean.iloc[-1] / trend_clean.iloc[0] - 1) * 100
            print(f"\nTrend Analysis:")
            print(f"  Overall trend growth: {trend_growth:.1f}%")
            print(f"  Trend start: ${trend_clean.iloc[0]:,.2f}")
            print(f"  Trend end: ${trend_clean.iloc[-1]:,.2f}")
        
        return decomposition
    
    def _plot_decomposition(self, ts_data, decomposition):
        """Create time series decomposition visualization"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # Original series
        axes[0].plot(ts_data.index, ts_data.values, 'b-', linewidth=2)
        axes[0].set_title('Original Revenue Time Series', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Revenue ($)')
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(decomposition.trend.index, decomposition.trend.values, 'g-', linewidth=2)
        axes[1].set_title('Trend Component', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Trend')
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        axes[1].grid(True, alpha=0.3)
        
        # Seasonality
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, 'orange', linewidth=2)
        axes[2].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        axes[2].set_title('Seasonal Component', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Seasonal Index')
        axes[2].grid(True, alpha=0.3)
        
        # Residuals
        axes[3].plot(decomposition.resid.index, decomposition.resid.values, 'purple', linewidth=1)
        axes[3].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        axes[3].set_title('Residual Component', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, 'time_series_decomposition.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print("Saved decomposition visualization")
    
    def prepare_prophet_data(self):
        """Prepare data for Prophet model"""
        prophet_df = self.monthly_revenue[['date', 'revenue']].copy()
        prophet_df.columns = ['ds', 'y']
        return prophet_df
    
    def train_prophet(self, forecast_periods=12):
        """Step 3 & 4: Train Prophet model and validate performance"""
        print("\n=== Step 3: Training Prophet Model ===")
        
        # Prepare data
        prophet_df = self.prepare_prophet_data()
        
        # Split data for validation
        train_size = len(prophet_df) - 3
        train_df = prophet_df.iloc[:train_size]
        test_df = prophet_df.iloc[train_size:]
        
        # Initialize and train Prophet
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )
        
        self.prophet_model.fit(train_df)
        
        # Validate on test set
        test_forecast = self.prophet_model.predict(test_df[['ds']])
        
        # Calculate metrics
        mae = mean_absolute_error(test_df['y'], test_forecast['yhat'])
        rmse = np.sqrt(mean_squared_error(test_df['y'], test_forecast['yhat']))
        mape = mean_absolute_percentage_error(test_df['y'], test_forecast['yhat']) * 100
        
        print(f"Prophet Validation Metrics:")
        print(f"  MAE: ${mae:,.2f}")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        print("\n=== Step 4: Prophet Validation Complete ===")
        
        # Retrain on full data and forecast
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )
        self.prophet_model.fit(prophet_df)
        
        # Generate future dates
        future = self.prophet_model.make_future_dataframe(periods=forecast_periods, freq='MS')
        prophet_forecast = self.prophet_model.predict(future)
        
        # Save forecast
        forecast_output = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_output.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
        forecast_output['model'] = 'Prophet'
        forecast_output.to_csv(
            os.path.join(self.output_dir, 'prophet_forecast.csv'),
            index=False
        )
        
        print(f"\n=== Step 5: Generated {forecast_periods}-Month Forecast ===")
        
        # Create visualization
        self._plot_prophet_forecast(prophet_df, prophet_forecast)
        
        return prophet_forecast, {'mae': mae, 'rmse': rmse, 'mape': mape}
    
    def check_stationarity(self, series):
        """Check if time series is stationary using ADF test"""
        result = adfuller(series.dropna())
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05
        }
    
    def train_arima(self, forecast_periods=12):
        """Train ARIMA model and generate forecasts"""
        print("\n=== Training ARIMA Model ===")
        
        # Prepare data
        revenue_series = self.monthly_revenue.set_index('date')['revenue']
        
        # Check stationarity
        stationarity = self.check_stationarity(revenue_series)
        print(f"Stationarity test - p-value: {stationarity['p_value']:.4f}")
        
        # Determine differencing
        d = 0 if stationarity['is_stationary'] else 1
        
        # Split data
        train_size = len(revenue_series) - 3
        train_series = revenue_series.iloc[:train_size]
        test_series = revenue_series.iloc[train_size:]
        
        # Grid search for best ARIMA parameters
        best_aic = float('inf')
        best_order = (1, d, 1)
        
        for p in range(0, 4):
            for q in range(0, 4):
                try:
                    model = ARIMA(train_series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue
        
        print(f"Best ARIMA order: {best_order}")
        
        # Train model with best parameters
        self.arima_model = ARIMA(train_series, order=best_order)
        fitted_model = self.arima_model.fit()
        
        # Validate
        test_forecast = fitted_model.forecast(steps=len(test_series))
        
        # Calculate metrics
        mae = mean_absolute_error(test_series, test_forecast)
        rmse = np.sqrt(mean_squared_error(test_series, test_forecast))
        mape = mean_absolute_percentage_error(test_series, test_forecast) * 100
        
        print(f"ARIMA Validation Metrics:")
        print(f"  MAE: ${mae:,.2f}")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        # Retrain on full data and forecast
        self.arima_model = ARIMA(revenue_series, order=best_order)
        fitted_model = self.arima_model.fit()
        
        # Generate forecast
        forecast = fitted_model.get_forecast(steps=forecast_periods)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Create forecast dataframe
        future_dates = pd.date_range(
            start=revenue_series.index[-1] + pd.DateOffset(months=1),
            periods=forecast_periods,
            freq='MS'
        )
        
        arima_forecast = pd.DataFrame({
            'date': future_dates,
            'forecast': forecast_mean.values,
            'lower_bound': conf_int.iloc[:, 0].values,
            'upper_bound': conf_int.iloc[:, 1].values,
            'model': 'ARIMA'
        })
        
        arima_forecast.to_csv(
            os.path.join(self.output_dir, 'arima_forecast.csv'),
            index=False
        )
        
        # Create visualization
        self._plot_arima_forecast(revenue_series, arima_forecast)
        
        return arima_forecast, {'mae': mae, 'rmse': rmse, 'mape': mape, 'order': best_order}
    
    def _plot_prophet_forecast(self, historical, forecast):
        """Create Prophet forecast visualization"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Historical data
        ax.plot(
            historical['ds'], historical['y'],
            'b-', linewidth=2, label='Historical Revenue'
        )
        
        # Forecast
        future_mask = forecast['ds'] > historical['ds'].max()
        ax.plot(
            forecast.loc[future_mask, 'ds'],
            forecast.loc[future_mask, 'yhat'],
            'r--', linewidth=2, label='Prophet Forecast'
        )
        
        # Confidence interval
        ax.fill_between(
            forecast.loc[future_mask, 'ds'],
            forecast.loc[future_mask, 'yhat_lower'],
            forecast.loc[future_mask, 'yhat_upper'],
            color='red', alpha=0.2, label='95% Confidence Interval'
        )
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Revenue ($)', fontsize=12)
        ax.set_title('Revenue Forecast - Prophet Model', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, 'prophet_forecast.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f"Saved Prophet forecast visualization")
    
    def _plot_arima_forecast(self, historical, forecast):
        """Create ARIMA forecast visualization"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Historical data
        ax.plot(
            historical.index, historical.values,
            'b-', linewidth=2, label='Historical Revenue'
        )
        
        # Forecast
        ax.plot(
            forecast['date'], forecast['forecast'],
            'g--', linewidth=2, label='ARIMA Forecast'
        )
        
        # Confidence interval
        ax.fill_between(
            forecast['date'],
            forecast['lower_bound'],
            forecast['upper_bound'],
            color='green', alpha=0.2, label='95% Confidence Interval'
        )
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Revenue ($)', fontsize=12)
        ax.set_title('Revenue Forecast - ARIMA Model', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, 'arima_forecast.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f"Saved ARIMA forecast visualization")
    
    def compare_models(self, prophet_metrics, arima_metrics):
        """Compare Prophet and ARIMA models"""
        print("\n=== Model Comparison ===")
        
        comparison = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'MAPE (%)'],
            'Prophet': [prophet_metrics['mae'], prophet_metrics['rmse'], prophet_metrics['mape']],
            'ARIMA': [arima_metrics['mae'], arima_metrics['rmse'], arima_metrics['mape']]
        })
        
        comparison.to_csv(
            os.path.join(self.output_dir, 'model_comparison.csv'),
            index=False
        )
        
        print(comparison.to_string(index=False))
        
        # Determine best model
        if prophet_metrics['mape'] < arima_metrics['mape']:
            best_model = 'Prophet'
        else:
            best_model = 'ARIMA'
        
        print(f"\nBest model based on MAPE: {best_model}")
        
        return comparison, best_model
    
    def create_combined_forecast(self):
        """Step 6: Create ensemble forecast for dashboard usage"""
        print("\n=== Step 6: Creating Dashboard-Ready Forecast ===")
        
        prophet_fc = pd.read_csv(os.path.join(self.output_dir, 'prophet_forecast.csv'))
        arima_fc = pd.read_csv(os.path.join(self.output_dir, 'arima_forecast.csv'))
        
        prophet_fc['date'] = pd.to_datetime(prophet_fc['date'])
        arima_fc['date'] = pd.to_datetime(arima_fc['date'])
        
        # Get future dates only
        historical_max = self.monthly_revenue['date'].max()
        prophet_future = prophet_fc[prophet_fc['date'] > historical_max].copy()
        arima_future = arima_fc.copy()
        
        # Merge forecasts
        combined = prophet_future[['date', 'forecast']].merge(
            arima_future[['date', 'forecast']],
            on='date',
            suffixes=('_prophet', '_arima')
        )
        
        # Ensemble (average)
        combined['forecast_ensemble'] = (
            combined['forecast_prophet'] + combined['forecast_arima']
        ) / 2
        
        combined.to_csv(
            os.path.join(self.output_dir, 'combined_forecast.csv'),
            index=False
        )
        
        # Create dashboard-ready format
        dashboard_forecast = combined.copy()
        dashboard_forecast['date'] = pd.to_datetime(dashboard_forecast['date'])
        dashboard_forecast['year'] = dashboard_forecast['date'].dt.year
        dashboard_forecast['month'] = dashboard_forecast['date'].dt.month
        dashboard_forecast['month_name'] = dashboard_forecast['date'].dt.strftime('%B')
        dashboard_forecast['quarter'] = dashboard_forecast['date'].dt.quarter
        
        dashboard_path = os.path.join(self.output_dir, 'forecast_dashboard_ready.csv')
        dashboard_forecast.to_csv(dashboard_path, index=False)
        print(f"Saved dashboard-ready forecast to {dashboard_path}")
        
        # Create comparison visualization
        self._plot_forecast_comparison(combined)
        
        print(f"Combined forecast saved")
        
        return combined
    
    def _plot_forecast_comparison(self, combined):
        """Step 7: Visualize all forecasts together"""
        print("\n=== Step 7: Generating Forecast Visualizations ===")
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Historical
        ax.plot(
            self.monthly_revenue['date'],
            self.monthly_revenue['revenue'],
            'b-', linewidth=2, label='Historical'
        )
        
        # Prophet
        ax.plot(
            combined['date'], combined['forecast_prophet'],
            'r--', linewidth=1.5, label='Prophet'
        )
        
        # ARIMA
        ax.plot(
            combined['date'], combined['forecast_arima'],
            'g--', linewidth=1.5, label='ARIMA'
        )
        
        # Ensemble
        ax.plot(
            combined['date'], combined['forecast_ensemble'],
            'purple', linewidth=2.5, label='Ensemble'
        )
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Revenue ($)', fontsize=12)
        ax.set_title('Revenue Forecast Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, 'forecast_comparison.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print("Saved forecast comparison visualization")
        
        # Create additional monthly breakdown chart
        self._plot_monthly_forecast_breakdown(combined)
    
    def _plot_monthly_forecast_breakdown(self, combined):
        """Create monthly forecast breakdown visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        combined['date'] = pd.to_datetime(combined['date'])
        combined['month_name'] = combined['date'].dt.strftime('%b %Y')
        
        # Bar chart of forecasted revenue
        colors = sns.color_palette('Blues', len(combined))
        axes[0].bar(range(len(combined)), combined['forecast_ensemble'], color=colors)
        axes[0].set_xticks(range(len(combined)))
        axes[0].set_xticklabels(combined['month_name'], rotation=45, ha='right')
        axes[0].set_ylabel('Forecasted Revenue ($)')
        axes[0].set_title('12-Month Revenue Forecast', fontsize=12, fontweight='bold')
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Cumulative forecast
        combined['cumulative'] = combined['forecast_ensemble'].cumsum()
        axes[1].plot(range(len(combined)), combined['cumulative'], 'g-o', linewidth=2, markersize=8)
        axes[1].fill_between(range(len(combined)), combined['cumulative'], alpha=0.3, color='green')
        axes[1].set_xticks(range(len(combined)))
        axes[1].set_xticklabels(combined['month_name'], rotation=45, ha='right')
        axes[1].set_ylabel('Cumulative Revenue ($)')
        axes[1].set_title('Cumulative 12-Month Forecast', fontsize=12, fontweight='bold')
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, 'forecast_monthly_breakdown.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print("Saved monthly forecast breakdown")
    
    def run_forecasting(self, forecast_periods=12):
        """Run complete forecasting pipeline with all steps"""
        print("\n" + "="*60)
        print("REVENUE FORECASTING PIPELINE")
        print("="*60)
        
        # Step 1: Load and aggregate data
        self.load_data()
        self.aggregate_monthly_revenue()
        
        # Step 2: Time series decomposition
        decomposition = self.perform_time_series_decomposition()
        
        # Steps 3-5: Train and validate Prophet
        prophet_forecast, prophet_metrics = self.train_prophet(forecast_periods)
        
        # Train ARIMA for comparison
        arima_forecast, arima_metrics = self.train_arima(forecast_periods)
        
        # Compare models
        comparison, best_model = self.compare_models(prophet_metrics, arima_metrics)
        
        # Step 6: Create dashboard-ready combined forecast
        combined = self.create_combined_forecast()
        
        # Step 8: Export all results
        self._export_all_results(prophet_metrics, arima_metrics, best_model)
        
        print("\n" + "="*60)
        print("FORECASTING PIPELINE COMPLETE")
        print("="*60)
        print(f"\nForecast period: {forecast_periods} months")
        print(f"Best performing model: {best_model}")
        
        return {
            'prophet_forecast': prophet_forecast,
            'arima_forecast': arima_forecast,
            'combined_forecast': combined,
            'comparison': comparison,
            'best_model': best_model,
            'decomposition': decomposition
        }
    
    def _export_all_results(self, prophet_metrics, arima_metrics, best_model):
        """Step 8: Export all forecast results to CSV"""
        print("\n=== Step 8: Exporting All Results ===")
        
        # Summary of all outputs
        outputs_summary = {
            'file': [
                'prophet_forecast.csv',
                'arima_forecast.csv', 
                'combined_forecast.csv',
                'forecast_dashboard_ready.csv',
                'time_series_decomposition.csv',
                'seasonal_indices.csv',
                'model_comparison.csv',
                'monthly_revenue_aggregated.csv'
            ],
            'description': [
                'Prophet model 12-month forecast with confidence intervals',
                'ARIMA model 12-month forecast with confidence intervals',
                'Ensemble forecast combining Prophet and ARIMA',
                'Dashboard-ready format with date dimensions',
                'Time series decomposition (trend, seasonal, residual)',
                'Monthly seasonality indices',
                'Model performance comparison (MAE, RMSE, MAPE)',
                'Monthly revenue aggregation from transactions'
            ]
        }
        
        summary_df = pd.DataFrame(outputs_summary)
        summary_path = os.path.join(self.output_dir, 'forecast_outputs_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Create forecast summary report
        report = f"""
REVENUE FORECAST SUMMARY
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

MODEL PERFORMANCE
-----------------
Prophet Model:
  - MAE: ${prophet_metrics['mae']:,.2f}
  - RMSE: ${prophet_metrics['rmse']:,.2f}
  - MAPE: {prophet_metrics['mape']:.2f}%

ARIMA Model (Order: {arima_metrics.get('order', 'N/A')}):
  - MAE: ${arima_metrics['mae']:,.2f}
  - RMSE: ${arima_metrics['rmse']:,.2f}
  - MAPE: {arima_metrics['mape']:.2f}%

Best Model: {best_model}

12-MONTH FORECAST SUMMARY
-------------------------
"""
        # Load combined forecast for summary
        combined = pd.read_csv(os.path.join(self.output_dir, 'combined_forecast.csv'))
        combined['date'] = pd.to_datetime(combined['date'])
        
        total_forecast = combined['forecast_ensemble'].sum()
        avg_monthly = combined['forecast_ensemble'].mean()
        
        report += f"""
Total Forecasted Revenue: ${total_forecast:,.2f}
Average Monthly Forecast: ${avg_monthly:,.2f}

Monthly Breakdown:
"""
        for _, row in combined.iterrows():
            month_str = pd.to_datetime(row['date']).strftime('%b %Y')
            report += f"  {month_str}: ${row['forecast_ensemble']:,.2f}\n"
        
        report += f"""
OUTPUT FILES
------------
Location: {self.output_dir}
"""
        for _, row in summary_df.iterrows():
            report += f"  - {row['file']}: {row['description']}\n"
        
        report_path = os.path.join(self.output_dir, 'forecast_summary_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Exported {len(outputs_summary['file'])} output files")
        print(f"Summary report: {report_path}")


def main():
    """Main entry point"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'outputs')
    output_dir = os.path.join(base_dir, 'outputs', 'forecasts')
    visuals_dir = os.path.join(base_dir, 'visuals')
    
    forecaster = RevenueForecaster(data_dir, output_dir, visuals_dir)
    results = forecaster.run_forecasting(forecast_periods=12)
    
    return results


if __name__ == "__main__":
    main()
