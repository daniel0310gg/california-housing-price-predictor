# 🚀 Getting Started Guide

Welcome to the California Housing Price Predictor! This guide will get you up and running in 5 minutes.

---

## ⚡ Super Quick Start (5 minutes)

### 1. Clone the Repository
```bash
git clone https://github.com/daniel0310gg/california-housing-price-predictor.git
cd california-housing-price-predictor
```

### 2. Install Python Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run the Project
```bash
python california_housing_predictor.py
```

### 4. View Results
✅ Check the `visualizations/` folder for charts!

---

## 📋 Step-by-Step Setup (Recommended)

### Prerequisites
- **Python 3.8+** (check with `python --version`)
- **pip** (Python package manager)
- **Git** (for cloning)

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/daniel0310gg/california-housing-price-predictor.git

# Navigate into the project
cd california-housing-price-predictor
```

### Step 2: Create Virtual Environment

A virtual environment keeps dependencies isolated:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages:
- pandas (data handling)
- numpy (numerical computing)
- scikit-learn (machine learning)
- matplotlib (visualization)
- seaborn (statistical plots)

### Step 4: Verify Installation

```bash
python -c "import pandas, numpy, sklearn; print('✅ All packages installed!')"
```

### Step 5: Run the Project

```bash
python california_housing_predictor.py
```

Expect output like:
```
============================================================
🏠 CALIFORNIA HOUSING PRICE PREDICTOR
============================================================

📂 Loading California housing dataset...
✅ Dataset loaded successfully!
   - Samples: 20,640
   - Features: 8
   - Time period: 1990

📊 Splitting data (80% train, 20% test)...
✅ Data split complete!
...
```

### Step 6: View Results

All visualizations are saved to `visualizations/`:
- `scatter_income_vs_price.png`
- `correlation_heatmap.png`
- `residuals_analysis.png`
- `actual_vs_predicted.png`
- `model_comparison.png`

Open them with your image viewer!

---

## 🎯 What Should I See?

When you run the script, you'll see:

1. **Dataset Loading** - Shows number of samples and features
2. **Data Splitting** - Train/test split summary
3. **Model Training** - Two linear regression models
4. **Performance Metrics** - R², MAE, RMSE scores
5. **Analysis Summary** - Key insights and feature importance

**Expected R² Scores:**
- Simple model: ~0.47 (good starting point)
- Multiple model: ~0.60 (better predictions)

---

## 🐛 Troubleshooting

### Python Not Found
**Problem:** `python: command not found`

**Solution:**
- On Windows: Use `python3` or add Python to PATH
- On macOS: Use `python3`
- On Linux: Install Python with `sudo apt-get install python3`

### Module Not Found Error
**Problem:** `ModuleNotFoundError: No module named 'pandas'`

**Solution:**
```bash
# Activate virtual environment first!
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Then install requirements
pip install -r requirements.txt
```

### Permissions Denied
**Problem:** `PermissionError: [Errno 13] Permission denied`

**Solution:**
```bash
# Make the visualizations folder writable
chmod -R 755 visualizations/
```

### Visualization Folder Not Found
**Problem:** `FileNotFoundError: visualizations folder`

**Solution:**
- The script creates it automatically
- If not, manually create it:
  ```bash
  mkdir visualizations
  ```

---

## 📚 Understanding the Output

### Console Output

```
============================================================
🏠 CALIFORNIA HOUSING PRICE PREDICTOR
============================================================

📂 Loading California housing dataset...
✅ Dataset loaded successfully!
   - Samples: 20,640
   - Features: 8
   - Time period: 1990

📊 Splitting data (80% train, 20% test)...
✅ Data split complete!
   - Training samples: 16,512
   - Testing samples: 4,128

🔧 Training Simple Linear Regression (Income only)...
✅ Model trained!
   - R² (Train): 0.4760
   - R² (Test): 0.4744
   - MAE: $0.73K
   - RMSE: $0.92K
   - Coefficient: 0.4519
   - Intercept: 0.4682
```

**What this means:**
- R² = 0.4744 → Model explains 47% of price variation
- MAE = $0.73K → Average error is ~$73,000
- Coefficient = 0.4519 → Each unit of income increases price by ~$45K

### Visualization Files

Each PNG file shows different aspects:

1. **scatter_income_vs_price.png**
   - Shows correlation between income and price
   - Red line = model's prediction

2. **correlation_heatmap.png**
   - Shows how all features relate to each other
   - Red = strong positive correlation
   - Blue = strong negative correlation

3. **residuals_analysis.png**
   - Two plots showing prediction errors
   - Should look random (not patterned)

4. **actual_vs_predicted.png**
   - Scatter plot of predictions vs reality
   - Points on red line = perfect predictions

5. **model_comparison.png**
   - Bar charts comparing simple vs multiple model
   - Shows improvement from adding features

---

## 🔬 Next Steps: Experiments to Try

### 1. Modify Features
Edit line in `california_housing_predictor.py`:
```python
# Try different features
X_train_multiple = self.X_train[:, [0, 3, 4]]  # Different columns
```

### 2. Change Train/Test Split
Edit line in `split_data()` method:
```python
test_size=0.3  # Use 70/30 split instead of 80/20
```

### 3. Add More Features
Extend the multiple model from 3 to 5 features:
```python
X_train_multiple = self.X_train[:, [0, 1, 2, 3, 4]]  # 5 features
```

### 4. Visualize Different Relationships
Create your own plots to explore the data.

---

## 📖 Learning Resources

### Inside This Repo
- **README.md** - Full project documentation
- **docs/METHODOLOGY.md** - Mathematical foundations
- **docs/INTERPRETATION_GUIDE.md** - Understanding results

### Online Tutorials
- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html)
- [Pandas Tutorial](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)

### YouTube Channels
- 3Blue1Brown - Linear Algebra & Calculus
- StatQuest with Josh Starmer - Statistics & ML
- Corey Schafer - Python & Data Science

---

## ✅ Verification Checklist

Before you start, verify:

- [ ] Python 3.8+ installed (`python --version`)
- [ ] Git installed and able to clone
- [ ] Repository cloned successfully
- [ ] Virtual environment activated (see `(venv)` in prompt)
- [ ] All packages installed (`pip list | grep pandas`)
- [ ] Script runs without errors
- [ ] Visualizations folder created
- [ ] 5 PNG files generated in `visualizations/`

---

## 🎓 What You're Learning

By working through this project, you'll understand:

✅ How to load and explore data with pandas
✅ How to split data for training and testing
✅ How to build linear regression models
✅ How to evaluate model performance
✅ How to create publication-quality visualizations
✅ How to interpret statistical results
✅ Real-world machine learning workflow

---

## 🆘 Getting Help

### Common Issues

**Q: Script runs but no visualizations?**
A: Check the `visualizations/` folder - they might have been created!

**Q: What do the numbers mean?**
A: See `docs/INTERPRETATION_GUIDE.md` for detailed explanations

**Q: Can I modify the code?**
A: Absolutely! Try changing features, parameters, and see what happens

**Q: Can I use different data?**
A: Yes! See the data loading section - you can load your own CSV

---

## 🚀 Ready to Begin?

```bash
# Copy-paste this into your terminal:
git clone https://github.com/daniel0310gg/california-housing-price-predictor.git
cd california-housing-price-predictor
pip install -r requirements.txt
python california_housing_predictor.py
```

That's it! You're running machine learning! 🎉

---

## ❓ Have Questions?

- 📖 Check README.md for general info
- 📚 Read docs/INTERPRETATION_GUIDE.md for understanding results
- 🔬 Read docs/METHODOLOGY.md for technical details
- 💬 Open an issue on GitHub
- ⭐ Star the repo if you found it helpful!

---

**Last Updated:** November 2025

**Happy Learning!** 🚀
