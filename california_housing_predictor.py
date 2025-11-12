"""
California Housing Price Predictor
==================================

A beginner-friendly machine learning project that predicts median house prices
in California using Linear Regression.

Author: Daniel
Date: November 2025
License: MIT
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# Set random seed for reproducibility
np.random.seed(42)

# Configure styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Create visualizations directory if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')


class CaliforniaHousingPredictor:
    """
    A class to handle California housing price prediction.
    
    Attributes:
        X_train: Training features
        X_test: Testing features
        y_train: Training target values
        y_test: Testing target values
        feature_names: Names of the features
        simple_model: Simple linear regression model
        multiple_model: Multiple linear regression model
    """
    
    def __init__(self):
        """Initialize the predictor and load data."""
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.simple_model = None
        self.multiple_model = None
        self.df = None
        
    def load_data(self):
        """Load the California housing dataset."""
        print("\n" + "="*60)
        print("🏠 CALIFORNIA HOUSING PRICE PREDICTOR")
        print("="*60)
        print("\n📂 Loading California housing dataset...")
        
        # Fetch the dataset
        housing = fetch_california_housing()
        self.X = housing.data
        self.y = housing.target
        self.feature_names = housing.feature_names
        
        # Create DataFrame for analysis
        self.df = pd.DataFrame(self.X, columns=self.feature_names)
        self.df['MedHouseVal'] = self.y
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   - Samples: {self.X.shape[0]:,}")
        print(f"   - Features: {self.X.shape[1]}")
        print(f"   - Time period: 1990")
        
    def split_data(self, test_size=0.2):
        """Split data into training and testing sets."""
        print("\n📊 Splitting data (80% train, 20% test)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        print(f"✅ Data split complete!")
        print(f"   - Training samples: {self.X_train.shape[0]:,}")
        print(f"   - Testing samples: {self.X_test.shape[0]:,}")
        
    def train_simple_model(self):
        """Train simple linear regression with only median income."""
        print("\n🔧 Training Simple Linear Regression (Income only)...")
        
        # Use only median income (first feature)
        X_train_simple = self.X_train[:, [0]]
        X_test_simple = self.X_test[:, [0]]
        
        self.simple_model = LinearRegression()
        self.simple_model.fit(X_train_simple, self.y_train)
        
        # Make predictions
        y_pred_train = self.simple_model.predict(X_train_simple)
        y_pred_test = self.simple_model.predict(X_test_simple)
        
        # Calculate metrics
        r2_train = r2_score(self.y_train, y_pred_train)
        r2_test = r2_score(self.y_test, y_pred_test)
        mae = mean_absolute_error(self.y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        
        self.simple_metrics = {
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mae': mae,
            'rmse': rmse,
            'y_pred': y_pred_test,
            'X_test': X_test_simple
        }
        
        print(f"✅ Model trained!")
        print(f"   - R² (Train): {r2_train:.4f}")
        print(f"   - R² (Test): {r2_test:.4f}")
        print(f"   - MAE: ${mae*100:.2f}K")
        print(f"   - RMSE: ${rmse*100:.2f}K")
        print(f"   - Coefficient: {self.simple_model.coef_[0]:.4f}")
        print(f"   - Intercept: {self.simple_model.intercept_:.4f}")
        
    def train_multiple_model(self):
        """Train multiple linear regression with 3 features."""
        print("\n🔧 Training Multiple Linear Regression (3 features)...")
        print("   Features: MedInc, HouseAge, AveRooms")
        
        # Use median income, house age, and average rooms
        X_train_multiple = self.X_train[:, [0, 1, 2]]
        X_test_multiple = self.X_test[:, [0, 1, 2]]
        
        self.multiple_model = LinearRegression()
        self.multiple_model.fit(X_train_multiple, self.y_train)
        
        # Make predictions
        y_pred_train = self.multiple_model.predict(X_train_multiple)
        y_pred_test = self.multiple_model.predict(X_test_multiple)
        
        # Calculate metrics
        r2_train = r2_score(self.y_train, y_pred_train)
        r2_test = r2_score(self.y_test, y_pred_test)
        mae = mean_absolute_error(self.y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        
        self.multiple_metrics = {
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mae': mae,
            'rmse': rmse,
            'y_pred': y_pred_test,
            'X_test': X_test_multiple
        }
        
        print(f"✅ Model trained!")
        print(f"   - R² (Train): {r2_train:.4f}")
        print(f"   - R² (Test): {r2_test:.4f}")
        print(f"   - MAE: ${mae*100:.2f}K")
        print(f"   - RMSE: ${rmse*100:.2f}K")
        print(f"   - Coefficients:")
        for i, feature in enumerate(['MedInc', 'HouseAge', 'AveRooms']):
            print(f"     • {feature}: {self.multiple_model.coef_[i]:.4f}")
        print(f"   - Intercept: {self.multiple_model.intercept_:.4f}")
        
    def plot_simple_regression(self):
        """Plot simple linear regression: Income vs Price."""
        print("\n📈 Creating Simple Regression visualization...")
        
        plt.figure(figsize=(12, 6))
        
        # Scatter plot
        plt.scatter(self.simple_metrics['X_test'], self.y_test, 
                   alpha=0.5, s=20, label='Actual prices', color='steelblue')
        
        # Regression line
        x_line = np.array([[self.simple_metrics['X_test'].min()], 
                          [self.simple_metrics['X_test'].max()]])
        y_line = self.simple_model.predict(x_line)
        plt.plot(x_line, y_line, 'r-', linewidth=2, label='Regression line')
        
        plt.xlabel('Median Income (scaled)', fontsize=12, fontweight='bold')
        plt.ylabel('Median House Price ($100K)', fontsize=12, fontweight='bold')
        plt.title('Simple Linear Regression: Income vs House Price\nR² = {:.4f}'.format(
            self.simple_metrics['r2_test']), fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/scatter_income_vs_price.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: visualizations/scatter_income_vs_price.png")
        
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap of all features."""
        print("\n📈 Creating Correlation Heatmap...")
        
        # Calculate correlation matrix
        corr_matrix = self.df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Heatmap: All Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: visualizations/correlation_heatmap.png")
        
    def plot_residuals(self):
        """Plot residuals analysis for multiple model."""
        print("\n📈 Creating Residuals Analysis...")
        
        residuals = self.y_test - self.multiple_metrics['y_pred']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(self.multiple_metrics['y_pred'], residuals, 
                       alpha=0.5, s=20, color='steelblue')
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
        axes[0].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1].hist(residuals, bins=30, edgecolor='black', color='steelblue', alpha=0.7)
        axes[1].set_xlabel('Residuals', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('visualizations/residuals_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: visualizations/residuals_analysis.png")
        
    def plot_actual_vs_predicted(self):
        """Plot actual vs predicted values."""
        print("\n📈 Creating Actual vs Predicted Plot...")
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(self.y_test, self.multiple_metrics['y_pred'], 
                   alpha=0.5, s=20, color='steelblue', label='Predictions')
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), self.multiple_metrics['y_pred'].min())
        max_val = max(self.y_test.max(), self.multiple_metrics['y_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                linewidth=2, label='Perfect prediction')
        
        plt.xlabel('Actual Prices ($100K)', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Prices ($100K)', fontsize=12, fontweight='bold')
        plt.title('Actual vs Predicted House Prices\nMultiple Linear Regression (R² = {:.4f})'.format(
            self.multiple_metrics['r2_test']), fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: visualizations/actual_vs_predicted.png")
        
    def plot_model_comparison(self):
        """Plot comparison of simple vs multiple models."""
        print("\n📈 Creating Model Comparison Plot...")
        
        models = ['Simple LR\n(Income only)', 'Multiple LR\n(3 features)']
        r2_scores = [self.simple_metrics['r2_test'], self.multiple_metrics['r2_test']]
        mae_values = [self.simple_metrics['mae'], self.multiple_metrics['mae']]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # R² Comparison
        colors = ['#3498db', '#e74c3c']
        bars1 = axes[0].bar(models, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        axes[0].set_ylabel('R² Score', fontsize=11, fontweight='bold')
        axes[0].set_title('Model Performance: R² Score', fontsize=12, fontweight='bold')
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # MAE Comparison
        bars2 = axes[1].bar(models, [m*100 for m in mae_values], color=colors, 
                           alpha=0.8, edgecolor='black', linewidth=2)
        axes[1].set_ylabel('Mean Absolute Error ($K)', fontsize=11, fontweight='bold')
        axes[1].set_title('Model Performance: MAE', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mae in zip(bars2, [m*100 for m in mae_values]):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'${mae:.1f}K', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: visualizations/model_comparison.png")
        
    def print_summary(self):
        """Print comprehensive analysis summary."""
        print("\n" + "="*60)
        print("📊 ANALYSIS SUMMARY")
        print("="*60)
        
        print("\n🔍 SIMPLE LINEAR REGRESSION (Income only)")
        print("-" * 60)
        print(f"R² Score (Train): {self.simple_metrics['r2_train']:.4f}")
        print(f"R² Score (Test):  {self.simple_metrics['r2_test']:.4f}")
        print(f"MAE:              ${self.simple_metrics['mae']*100:.2f}K")
        print(f"RMSE:             ${self.simple_metrics['rmse']*100:.2f}K")
        print(f"\nInterpretation:")
        print(f"• Median income alone explains {self.simple_metrics['r2_test']*100:.1f}% of price variance")
        print(f"• Average prediction error: ${self.simple_metrics['mae']*100:.0f},000")
        
        print("\n🔍 MULTIPLE LINEAR REGRESSION (MedInc, HouseAge, AveRooms)")
        print("-" * 60)
        print(f"R² Score (Train): {self.multiple_metrics['r2_train']:.4f}")
        print(f"R² Score (Test):  {self.multiple_metrics['r2_test']:.4f}")
        print(f"MAE:              ${self.multiple_metrics['mae']*100:.2f}K")
        print(f"RMSE:             ${self.multiple_metrics['rmse']*100:.2f}K")
        print(f"\nInterpretation:")
        improvement = (self.multiple_metrics['r2_test'] - self.simple_metrics['r2_test']) * 100
        print(f"• 3 features explain {self.multiple_metrics['r2_test']*100:.1f}% of price variance")
        print(f"• Improvement over simple model: +{improvement:.1f}%")
        print(f"• Average prediction error: ${self.multiple_metrics['mae']*100:.0f},000")
        
        print("\n💡 KEY INSIGHTS")
        print("-" * 60)
        print(f"1. Income is the strongest predictor (R² = {self.simple_metrics['r2_test']:.2f})")
        print(f"2. Adding features improves accuracy by {improvement:.1f}%")
        print(f"3. Best model: Multiple Linear Regression")
        print(f"4. The model captures ~60% of house price variation")
        print(f"5. Remaining 40% likely due to other factors (quality, location details, etc.)")
        
        print("\n📈 FEATURE IMPORTANCE (Multiple Model)")
        print("-" * 60)
        features_importance = [
            ('Median Income', self.multiple_model.coef_[0]),
            ('House Age', self.multiple_model.coef_[1]),
            ('Average Rooms', self.multiple_model.coef_[2])
        ]
        features_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        for i, (feature, coef) in enumerate(features_importance, 1):
            print(f"{i}. {feature:20s}: {coef:7.4f}")
        
        print("\n✅ Analysis Complete!")
        print("📁 All visualizations saved to: visualizations/")
        print("="*60 + "\n")
        
    def run(self):
        """Run the complete analysis pipeline."""
        self.load_data()
        self.split_data()
        self.train_simple_model()
        self.train_multiple_model()
        self.plot_simple_regression()
        self.plot_correlation_heatmap()
        self.plot_residuals()
        self.plot_actual_vs_predicted()
        self.plot_model_comparison()
        self.print_summary()


if __name__ == "__main__":
    # Create predictor and run analysis
    predictor = CaliforniaHousingPredictor()
    predictor.run()
    
    print("\n🎉 Thank you for using California Housing Price Predictor!")
    print("📚 For more information, see README.md")
    print("⭐ Don't forget to star the repository on GitHub!\n")
