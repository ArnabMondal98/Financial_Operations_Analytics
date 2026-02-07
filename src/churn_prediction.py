"""
Customer Churn Prediction Module
Implements Logistic Regression and Random Forest for churn prediction
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import joblib

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns


class ChurnPredictor:
    """Customer churn prediction using ML models"""
    
    def __init__(self, data_dir, output_dir, models_dir, visuals_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.visuals_dir = visuals_dir
        
        self.customers_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_names = None
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(visuals_dir, exist_ok=True)
    
    def load_and_prepare_data(self):
        """Load and prepare data for modeling"""
        print("Loading customer data...")
        
        self.customers_df = pd.read_csv(
            os.path.join(self.data_dir, 'processed_customers.csv')
        )
        
        # Convert date columns
        date_columns = ['signup_date', 'first_transaction', 'last_transaction']
        for col in date_columns:
            if col in self.customers_df.columns:
                self.customers_df[col] = pd.to_datetime(self.customers_df[col])
        
        print(f"Loaded {len(self.customers_df)} customers")
        print(f"Churn rate: {self.customers_df['is_churned'].mean():.2%}")
        
        return self.customers_df
    
    def prepare_features(self):
        """Prepare features for ML models"""
        print("\nPreparing features...")
        
        # Select numerical features for modeling
        numerical_features = [
            'monthly_fee', 'total_transactions', 'total_revenue',
            'avg_transaction', 'successful_transactions', 'failed_transactions',
            'customer_tenure_days', 'days_since_last_transaction',
            'transaction_frequency', 'payment_failure_rate', 'avg_monthly_revenue'
        ]
        
        # Encode categorical features
        categorical_features = ['country', 'industry', 'subscription_type']
        
        # Create feature matrix
        df_features = self.customers_df.copy()
        
        # Encode categoricals
        label_encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            df_features[f'{col}_encoded'] = le.fit_transform(df_features[col].astype(str))
            label_encoders[col] = le
        
        # Final feature list
        self.feature_names = numerical_features + [f'{col}_encoded' for col in categorical_features]
        
        # Prepare X and y
        X = df_features[self.feature_names].copy()
        y = df_features['is_churned'].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"Features: {len(self.feature_names)}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n=== Training Logistic Regression ===")
        
        # Grid search for best parameters
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
            'max_iter': [1000]
        }
        
        lr = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        best_lr = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Predictions
        y_pred = best_lr.predict(self.X_test_scaled)
        y_pred_proba = best_lr.predict_proba(self.X_test_scaled)[:, 1]
        
        # Metrics
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba, 'Logistic Regression')
        
        # Save model
        joblib.dump(best_lr, os.path.join(self.models_dir, 'logistic_regression.pkl'))
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': best_lr.coef_[0],
            'abs_coefficient': np.abs(best_lr.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        feature_importance.to_csv(
            os.path.join(self.output_dir, 'lr_feature_importance.csv'),
            index=False
        )
        
        # Create visualizations
        self._plot_feature_importance(feature_importance, 'Logistic Regression')
        self._plot_roc_curve(self.y_test, y_pred_proba, 'Logistic Regression')
        
        return best_lr, metrics, y_pred_proba
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n=== Training Random Forest ===")
        
        # Grid search for best parameters
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        best_rf = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Predictions
        y_pred = best_rf.predict(self.X_test_scaled)
        y_pred_proba = best_rf.predict_proba(self.X_test_scaled)[:, 1]
        
        # Metrics
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba, 'Random Forest')
        
        # Save model
        joblib.dump(best_rf, os.path.join(self.models_dir, 'random_forest.pkl'))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': best_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(
            os.path.join(self.output_dir, 'rf_feature_importance.csv'),
            index=False
        )
        
        # Create visualizations
        self._plot_feature_importance(feature_importance, 'Random Forest', importance_col='importance')
        self._plot_roc_curve(self.y_test, y_pred_proba, 'Random Forest')
        
        return best_rf, metrics, y_pred_proba
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba, model_name):
        """Calculate classification metrics"""
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        print(f"\n{model_name} Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return metrics
    
    def _plot_feature_importance(self, importance_df, model_name, importance_col='abs_coefficient'):
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = importance_df.head(10)
        
        colors = sns.color_palette('viridis', len(top_features))
        bars = ax.barh(
            top_features['feature'],
            top_features[importance_col],
            color=colors
        )
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top 10 Features - {model_name}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, f'{model_name.lower().replace(" ", "_")}_features.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
    
    def _plot_roc_curve(self, y_true, y_pred_proba, model_name):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, f'{model_name.lower().replace(" ", "_")}_roc.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
    
    def generate_churn_predictions(self, model_type='random_forest'):
        """Generate churn predictions for all customers"""
        print("\n=== Generating Churn Predictions ===")
        
        # Load model
        if model_type == 'logistic_regression':
            model = joblib.load(os.path.join(self.models_dir, 'logistic_regression.pkl'))
        else:
            model = joblib.load(os.path.join(self.models_dir, 'random_forest.pkl'))
        
        # Prepare all customer data
        df = self.customers_df.copy()
        
        # Encode categoricals
        categorical_features = ['country', 'industry', 'subscription_type']
        for col in categorical_features:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        # Get features
        X_all = df[self.feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        X_all_scaled = self.scaler.transform(X_all)
        
        # Generate predictions
        df['churn_probability'] = model.predict_proba(X_all_scaled)[:, 1]
        df['churn_risk_level'] = pd.cut(
            df['churn_probability'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Save predictions
        output_cols = [
            'customer_id', 'country', 'industry', 'subscription_type',
            'monthly_fee', 'total_revenue', 'customer_tenure_days',
            'is_churned', 'churn_probability', 'churn_risk_level'
        ]
        
        predictions_df = df[output_cols].copy()
        predictions_df.to_csv(
            os.path.join(self.output_dir, 'churn_predictions.csv'),
            index=False
        )
        
        # Summary statistics
        risk_summary = df.groupby('churn_risk_level').agg({
            'customer_id': 'count',
            'total_revenue': 'sum',
            'churn_probability': 'mean'
        }).reset_index()
        risk_summary.columns = ['risk_level', 'customer_count', 'revenue_at_risk', 'avg_probability']
        
        risk_summary.to_csv(
            os.path.join(self.output_dir, 'churn_risk_summary.csv'),
            index=False
        )
        
        print("\nChurn Risk Summary:")
        print(risk_summary.to_string(index=False))
        
        # Create visualization
        self._plot_churn_distribution(df)
        
        return predictions_df, risk_summary
    
    def _plot_churn_distribution(self, df):
        """Plot churn probability distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(df['churn_probability'], bins=30, color='steelblue', edgecolor='white')
        axes[0].axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        axes[0].set_xlabel('Churn Probability', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Distribution of Churn Probability', fontsize=14, fontweight='bold')
        axes[0].legend()
        
        # Risk level pie chart
        risk_counts = df['churn_risk_level'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        axes[1].pie(
            risk_counts, labels=risk_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90
        )
        axes[1].set_title('Customers by Churn Risk Level', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visuals_dir, 'churn_distribution.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
    
    def compare_models(self, lr_metrics, rf_metrics):
        """Compare model performance"""
        print("\n=== Model Comparison ===")
        
        comparison = pd.DataFrame([lr_metrics, rf_metrics])
        comparison.to_csv(
            os.path.join(self.output_dir, 'churn_model_comparison.csv'),
            index=False
        )
        
        print(comparison.to_string(index=False))
        
        # Determine best model
        if rf_metrics['roc_auc'] > lr_metrics['roc_auc']:
            best_model = 'Random Forest'
        else:
            best_model = 'Logistic Regression'
        
        print(f"\nBest model based on ROC AUC: {best_model}")
        
        return comparison, best_model
    
    def run_churn_prediction(self):
        """Run complete churn prediction pipeline"""
        print("\n" + "="*60)
        print("CUSTOMER CHURN PREDICTION MODULE")
        print("="*60)
        
        self.load_and_prepare_data()
        self.prepare_features()
        
        lr_model, lr_metrics, lr_proba = self.train_logistic_regression()
        rf_model, rf_metrics, rf_proba = self.train_random_forest()
        
        comparison, best_model = self.compare_models(lr_metrics, rf_metrics)
        
        # Use best model for predictions
        model_type = 'random_forest' if best_model == 'Random Forest' else 'logistic_regression'
        predictions, risk_summary = self.generate_churn_predictions(model_type)
        
        print("\n=== Churn Prediction Complete ===")
        
        return {
            'predictions': predictions,
            'risk_summary': risk_summary,
            'comparison': comparison,
            'best_model': best_model
        }


def main():
    """Main entry point"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'outputs')
    output_dir = os.path.join(base_dir, 'outputs', 'churn')
    models_dir = os.path.join(base_dir, 'models')
    visuals_dir = os.path.join(base_dir, 'visuals')
    
    predictor = ChurnPredictor(data_dir, output_dir, models_dir, visuals_dir)
    results = predictor.run_churn_prediction()
    
    return results


if __name__ == "__main__":
    main()
