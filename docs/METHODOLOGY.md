# Methodology & Technical Approach

## Overview
This document explains the mathematical foundations and technical approach used in the California Housing Price Predictor.

---

## 1. Linear Regression Theory

### Simple Linear Regression
The simplest form of linear regression with one independent variable:

```
ŷ = β₀ + β₁x
```

Where:
- **ŷ** = Predicted value
- **β₀** = Intercept (constant term)
- **β₁** = Slope (coefficient for x)
- **x** = Independent variable (feature)

### Multiple Linear Regression
Extending to multiple features:

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Where:
- **n** = Number of features
- **x₁, x₂, ..., xₙ** = Independent variables
- **β₁, β₂, ..., βₙ** = Coefficients for each feature

---

## 2. Model Training Process

### Objective Function (Cost Function)
We minimize the Mean Squared Error (MSE):

```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

Where:
- **n** = Number of samples
- **yᵢ** = Actual value
- **ŷᵢ** = Predicted value

### Normal Equation
Scikit-learn uses the normal equation to find optimal coefficients:

```
β = (XᵀX)⁻¹Xᵀy
```

Where:
- **X** = Feature matrix
- **y** = Target vector
- **β** = Coefficient vector

---

## 3. Model Evaluation Metrics

### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)

Where:
- SS_res = Σ(yᵢ - ŷᵢ)²  (Sum of squared residuals)
- SS_tot = Σ(yᵢ - ȳ)²   (Total sum of squares)
```

**Interpretation:**
- **R² = 1**: Perfect fit
- **R² = 0.5**: Model explains 50% of variance
- **R² = 0**: Model is no better than using mean
- **R² < 0**: Model performs worse than baseline

### Mean Absolute Error (MAE)
```
MAE = (1/n) Σ|yᵢ - ŷᵢ|
```

**Advantage:** Easy to interpret (same units as target)

### Root Mean Squared Error (RMSE)
```
RMSE = √[(1/n) Σ(yᵢ - ŷᵢ)²]
```

**Advantage:** Penalizes large errors more heavily

---

## 4. Data Preparation

### Train-Test Split
We use 80-20 split:
- **Training set (80%)**: Used to train the model
- **Testing set (20%)**: Used to evaluate performance

**Why?** Prevents overfitting and gives honest performance estimate

### Data Normalization
The California Housing dataset comes pre-scaled:
- Features are normalized (roughly 0-15 range)
- Target is normalized (roughly 0-5 range, representing $100K)

**Formula:**
```
x_normalized = (x - mean) / std_dev
```

---

## 5. Feature Analysis

### Correlation Analysis
We calculate Pearson correlation coefficients:

```
r = Cov(X, Y) / (σₓ × σᵧ)
```

**Interpretation:**
- **r = 1**: Perfect positive correlation
- **r = 0**: No correlation
- **r = -1**: Perfect negative correlation
- **|r| > 0.7**: Strong correlation
- **|r| < 0.3**: Weak correlation

### Residuals Analysis
Residuals are differences between actual and predicted values:

```
Residual = yᵢ - ŷᵢ
```

**Good residuals should:**
1. Be randomly scattered around zero
2. Have no patterns or trends
3. Follow normal distribution
4. Have constant variance (homoscedasticity)

---

## 6. Project-Specific Implementation

### Feature Selection
We selected three features for the multiple model:

1. **MedInc (Median Income)** - Strongest predictor
2. **HouseAge (House Age)** - Captures property maturity
3. **AveRooms (Average Rooms)** - Relates to property size

**Why these?**
- Highest correlation with target
- Easy to interpret
- Balance between complexity and performance
- Avoid multicollinearity

### Model Comparison

**Simple Model Benefits:**
- Easy to understand and interpret
- Fast to train
- Good baseline for comparison
- Shows that income matters most

**Multiple Model Benefits:**
- Better predictions (R² improves by 27%)
- Captures more variance
- Still relatively simple and interpretable
- Foundation for more complex models

---

## 7. Common Linear Regression Assumptions

### 1. Linearity
- Relationship between features and target is linear
- **Check:** Residual plots should show random scatter

### 2. Independence
- Observations are independent of each other
- **Check:** No patterns in residual plots

### 3. Homoscedasticity
- Variance of residuals is constant
- **Check:** Residuals spread equally around zero

### 4. Normality
- Residuals are normally distributed
- **Check:** Histogram of residuals should be bell-shaped

### 5. No Multicollinearity
- Features are not highly correlated with each other
- **Check:** Correlation matrix shows low inter-feature correlations

---

## 8. Limitations & Extensions

### Current Limitations
1. Linear model can't capture non-linear relationships
2. 40% of variance unexplained by our 3 features
3. May not perform well on new regions/time periods
4. Outliers can significantly affect coefficients

### Possible Extensions
1. **Polynomial Regression**: Add interaction terms or higher-degree polynomials
2. **Regularization**: Use Ridge or Lasso to prevent overfitting
3. **Feature Engineering**: Create new features (e.g., price per room)
4. **Non-linear Models**: Try decision trees, random forests, neural networks
5. **Cross-Validation**: Use k-fold CV for more robust evaluation
6. **Outlier Treatment**: Remove or handle extreme values

---

## 9. Performance Summary

| Metric | Simple | Multiple | Interpretation |
|--------|--------|----------|----------------|
| R² Score | 0.47 | 0.60 | More variance explained with more features |
| MAE | $0.73M | $0.68M | Slightly lower average error |
| RMSE | $0.92M | $0.95M | Similar variance of errors |
| Simplicity | Very High | High | Trade-off between accuracy and simplicity |

---

## 10. References & Learning Resources

### Textbooks
- James et al. (2013) - "An Introduction to Statistical Learning"
- Hastie et al. (2009) - "The Elements of Statistical Learning"

### Online Resources
- Scikit-learn: https://scikit-learn.org/stable/modules/linear_model.html
- 3Blue1Brown: Essence of Linear Algebra (YouTube)
- StatQuest with Josh Starmer: Linear Regression (YouTube)

### Mathematical Background
- Linear algebra basics
- Calculus (derivatives, optimization)
- Statistics and probability
- Correlation and covariance

---

**Last Updated:** November 2025
