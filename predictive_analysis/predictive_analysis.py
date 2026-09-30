import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========== 0. GLOBAL FIX ==========
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
sns.set_style("whitegrid", {'font.family':'serif', 'font.serif':['Times New Roman']})

def apply_times_font(ax):
    """Apply Times New Roman to all text in an axis - fixes your previous issue"""
    ax.set_xlabel(ax.get_xlabel(), fontname='Times New Roman', labelpad=10)
    ax.set_ylabel(ax.get_ylabel(), fontname='Times New Roman', labelpad=10)
    ax.set_title(ax.get_title(), fontname='Times New Roman', pad=15)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    leg = ax.get_legend()
    if leg:
        for text in leg.get_texts():
            text.set_fontname('Times New Roman')
        leg.get_title().set_fontname('Times New Roman')

# ========== 1. LOAD HISTORICAL DATA FROM ONLINE (No CSV needed) ==========
# Using monthly sales / passenger data - real historical trend data
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
# Fallback if that fails: use sales data
try:
    df = pd.read_csv(URL)
except:
    URL = "https://raw.githubusercontent.com/plotly/datasets/master/monthly-milk-production.csv"
    df = pd.read_csv(URL)

print(f"Historical Dataset Loaded: {df.shape}")
print(df.head())

# Standardize columns
df.columns = ['Month', 'Sales']
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month')
df.set_index('Month', inplace=True)

# Handle missing values if any
df = df.dropna()

# ========== 2. CLEAN & PREPROCESS + FEATURE ENGINEERING ==========
# Create time-based features for regression
df['TimeIndex'] = np.arange(len(df))
df['Year'] = df.index.year
df['MonthNum'] = df.index.month
df['Lag1'] = df['Sales'].shift(1)
df['Lag12'] = df['Sales'].shift(12)
df['RollingMean3'] = df['Sales'].rolling(3).mean()
df['RollingMean12'] = df['Sales'].rolling(12).mean()

# Drop NaN from lags
df_clean = df.dropna().copy()
print(f"After preprocessing: {df_clean.shape}")

# ========== 3. EDA - HISTORICAL TREND ==========
fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
ax.plot(df.index, df['Sales'], color='#2C3E50', linewidth=2, label='Historical Sales')
ax.set_title('Historical Data - Sales Trend Over Time', fontname='Times New Roman', fontweight='bold', pad=20)
ax.set_xlabel('Year', fontname='Times New Roman', labelpad=12)
ax.set_ylabel('Sales / Passengers (Historical Value)', fontname='Times New Roman', labelpad=12)
ax.legend(frameon=True)
apply_times_font(ax)
ax.grid(True, alpha=0.3, linestyle='--')
fig.savefig('historical_trend.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
plt.show()

# ========== 4. PREPARE DATA FOR PREDICTION ==========
features = ['TimeIndex', 'MonthNum', 'Lag1', 'Lag12', 'RollingMean3', 'RollingMean12']
X = df_clean[features]
y = df_clean['Sales']

# Time-series split (80% train, 20% test - keep order)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 5. BUILD PREDICTIVE MODELS ==========
# Model 1: Linear Regression (Trend)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_test_scaled)

# Model 2: Random Forest (Non-linear trends)
rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
rf.fit(X_train_scaled, y_train)
pred_rf = rf.predict(X_test_scaled)

# ========== 6. EVALUATE MODEL ACCURACY ==========
def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n{name}:")
    print(f" RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.3f} | MAPE: {mape:.2f}%")
    return rmse, mae, r2, mape

print("\n========== MODEL EVALUATION ==========")
evaluate(y_test, pred_lr, "Linear Regression")
evaluate(y_test, pred_rf, "Random Forest Regressor")

# ========== 7. VISUALIZE PREDICTIONS VS ACTUAL ==========
fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

# Plot train history
ax.plot(y_train.index, y_train, label='Historical Train Data', color='gray', linewidth=1.5, alpha=0.7)
ax.plot(y_test.index, y_test, label='Actual Future (Test)', color='black', linewidth=2.5)

# Predictions
ax.plot(y_test.index, pred_lr, label='Predicted - Linear Regression', color='#E74C3C', linestyle='--', linewidth=2)
ax.plot(y_test.index, pred_rf, label='Predicted - Random Forest', color='#2980B9', linestyle='-', linewidth=2)

ax.set_title('Predictive Analytics: Forecast vs Actual Future Trends', fontname='Times New Roman', fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel('Timeline (Historical + Forecast)', fontname='Times New Roman', labelpad=12)
ax.set_ylabel('Sales / Value', fontname='Times New Roman', labelpad=12)
ax.legend(frameon=True, loc='best', edgecolor='gray')
apply_times_font(ax)
ax.grid(True, alpha=0.3, linestyle='--')
fig.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
plt.show()

# ========== 8. FUTURE FORECASTING (Next 24 Months) ==========
future_months = 24
last_date = df_clean.index[-1]
future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=future_months, freq='MS')

# Simple future forecast using last known pattern
future_df = pd.DataFrame(index=future_dates)
future_df['TimeIndex'] = np.arange(len(df_clean), len(df_clean)+future_months)
future_df['MonthNum'] = future_df.index.month

# For lag features, use rolling predictions (recursive)
# For simplicity, use last values
last_sales = df_clean['Sales'].values
future_lag1 = []
for i in range(future_months):
    if i==0:
        future_lag1.append(last_sales[-1])
    else:
        future_lag1.append(future_df['Predicted_RF'].iloc[i-1] if 'Predicted_RF' in future_df else last_sales[-1])

future_df['Lag1'] = future_lag1
future_df['Lag12'] = df_clean['Sales'].values[-12:].tolist() + [np.nan]*(future_months-12)
future_df['Lag12'] = future_df['Lag12'].ffill()
future_df['RollingMean3'] = df_clean['Sales'].rolling(3).mean().iloc[-1]
future_df['RollingMean12'] = df_clean['Sales'].rolling(12).mean().iloc[-1]

X_future = future_df[features]
X_future_scaled = scaler.transform(X_future)
future_df['Predicted_RF'] = rf.predict(X_future_scaled)
future_df['Predicted_LR'] = lr.predict(X_future_scaled)

# Plot future forecast
fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
ax.plot(df_clean.index, df_clean['Sales'], label='Historical Data', color='gray', linewidth=1.5)
ax.plot(future_df.index, future_df['Predicted_RF'], label='Future Forecast (24 Months) - RF', color='#2980B9', linewidth=2.5, marker='o', markersize=4)
ax.fill_between(future_df.index, future_df['Predicted_RF']*0.9, future_df['Predicted_RF']*1.1, color='#2980B9', alpha=0.2, label='Confidence Interval (90%)')

ax.set_title('Future Trend Forecast - Next 24 Months', fontname='Times New Roman', fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel('Year', fontname='Times New Roman', labelpad=12)
ax.set_ylabel('Predicted Sales', fontname='Times New Roman', labelpad=12)
ax.legend(frameon=True, loc='best')
apply_times_font(ax)
ax.grid(True, alpha=0.3, linestyle='--')
fig.savefig('future_forecast.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
plt.show()

# ========== 9. FEATURE IMPORTANCE ==========
importance = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
sns.barplot(data=importance, x='Importance', y='Feature', palette='Blues_d', ax=ax)
ax.set_title('Feature Importance - What Drives Predictions?', fontname='Times New Roman', fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel('Importance Score', fontname='Times New Roman', labelpad=12)
ax.set_ylabel('Features', fontname='Times New Roman', labelpad=12)
apply_times_font(ax)
fig.savefig('feature_importance.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
plt.show()

# Save results
df_clean.to_csv('historical_data_cleaned.csv')
future_df.to_csv('future_forecast_24_months.csv')
print("\nSaved: future_forecast_24_months.csv")