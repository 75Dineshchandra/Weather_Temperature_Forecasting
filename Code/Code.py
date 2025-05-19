#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from tabulate import tabulate
from scipy.signal import dlsim
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import t as t_dist
from scipy.stats import chi2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.graphics.tsaplots import plot_acf

# Load dataset (Check delimiter if needed)
file_path = "jena_climate_2009_2016.csv"  # Update with actual file path
df = pd.read_csv(file_path, sep=',')  # If error, try sep=';'

# Convert 'Date Time' column to datetime and set as index
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S', dayfirst=True, errors='coerce')

# Drop any rows where 'Date Time' could not be parsed
df = df.dropna(subset=['Date Time'])

# Set index
df.set_index('Date Time', inplace=True)

# Sort index in ascending order
df = df.sort_index()

# ✅ Use proper datetime format for slicing (ISO format: YYYY-MM-DD)
df = df.loc['2009-01-01':'2011-12-31']

# Target variable (e.g., Temperature 'T (degC)')
target_column = 'T (degC)'
if target_column not in df.columns:
    print(f"Error: Column '{target_column}' not found in dataset. Available columns: {df.columns}")

ts = df[target_column].dropna()

df.info()
#%%
# Plot the dependent variable (temperature) over time
plt.figure(figsize=(12, 5))
plt.plot(df['T (degC)'], label='Temperature (degC)', color='blue')
plt.title('Temperature Over Time (2009-2012)')
plt.xlabel('Date')
plt.ylabel('Temperature (degC)')
plt.legend()
plt.grid(True)
plt.show()

def ACF_PACF_Plot(y, lags):
    """Plots ACF and PACF using the specified function"""
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)

    fig = plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.title('ACF of ARMA(1,0) Process')
    plt.xlabel("lag")
    plt.ylabel("mag")
    plot_acf(y, ax=plt.gca(), lags=lags)

    plt.subplot(212)
    plt.title('PACF of ARMA(1,0) Process')
    plt.xlabel("lag")
    plt.ylabel("mag")
    plot_pacf(y, ax=plt.gca(), lags=lags)

    fig.tight_layout(pad=3)
    plt.show()




# Plot using the specified function
ACF_PACF_Plot(ts, lags=50)
# Compute correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Jena Climate Dataset')
plt.show()
#%%
# Split the dataset into train (80%) and test (20%)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

print(f"Training set size: {len(train)}")
print(f"Testing set size: {len(test)}")
#%%

# Function to compute rolling mean and variance
def cal_rolling_mean_var(data, column_name):
    rolling_means = data[column_name].expanding().mean()
    rolling_vars = data[column_name].expanding().var()

    # Create subplots for mean and variance
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Plot rolling mean
    ax[0].plot(rolling_means, label=f'Rolling Mean of {column_name}', color='blue')
    ax[0].set_title(f'Rolling Mean of {column_name}')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Mean')
    ax[0].legend()
    ax[0].grid(True)

    # Plot rolling variance
    ax[1].plot(rolling_vars, label=f'Rolling Variance of {column_name}', color='orange')
    ax[1].set_title(f'Rolling Variance of {column_name}')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Variance')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

# Plot time series data
plt.figure(figsize=(12, 5))
plt.plot(ts, label=f'{target_column} Over Time (2009-2012)', color='green')
plt.title(f"Time Series Plot for {target_column} (2009-2012)")
plt.xlabel("Date")
plt.ylabel(target_column)
plt.legend()
plt.grid(True)
plt.show()

# Compute and plot rolling statistics
cal_rolling_mean_var(df, target_column)

# Perform Augmented Dickey-Fuller (ADF) test
def adf_test(series):
    result = adfuller(series)
    print("\n=== Augmented Dickey-Fuller (ADF) Test ===")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    if result[1] < 0.05:
        print("✅ The series is likely stationary (reject H0).")
    else:
        print("❌ The series is likely non-stationary (fail to reject H0).")

adf_test(ts)

# Perform KPSS test
def kpss_test(series):
    result = kpss(series, regression='c', nlags="auto")  # 'c' means constant-only
    print("\n=== Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test ===")
    print(f"KPSS Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[3].items():
        print(f"  {key}: {value:.4f}")
    if result[1] < 0.05:
        print("❌ The series is likely non-stationary (reject H0).")
    else:
        print("✅ The series is likely stationary (fail to reject H0).")

kpss_test(ts)

#%%
# Continue from previous code...
def adf_test(series):
    result = adfuller(series)
    print("\n=== Augmented Dickey-Fuller (ADF) Test ===")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    if result[1] < 0.05:
        print("✅ The series is likely stationary (reject H0).")
    else:
        print("❌ The series is likely non-stationary (fail to reject H0).")

# Perform KPSS test
def kpss_test(series):
    result = kpss(series, regression='c', nlags="auto")  # 'c' means constant-only
    print("\n=== Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test ===")
    print(f"KPSS Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[3].items():
        print(f"  {key}: {value:.4f}")
    if result[1] < 0.05:
        print("❌ The series is likely non-stationary (reject H0).")
    else:
        print("✅ The series is likely stationary (fail to reject H0).")

kpss_test(ts)

# Function to apply differencing and plot results
def difference_and_plot(series, diff_order=1, log_transform=False):
    """
    Apply differencing and plot results with statistical tests
    """
    plt.figure(figsize=(12, 8))


    diff_series = series.copy()
    for _ in range(diff_order):
        diff_series = diff_series.diff().dropna()
    title_suffix = f' ({diff_order}nd Order Differencing)' if diff_order == 2 else \
                    f' ({diff_order}rd Order Differencing)' if diff_order == 3 else \
                    f' ({diff_order}th Order Differencing)'

    # Plot the differenced series
    plt.subplot(3, 1, 1)
    plt.plot(diff_series, label=f'{target_column}{title_suffix}', color='purple')
    plt.title(f'Time Series Plot{title_suffix}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Plot ACF
    plt.subplot(3, 1, 2)
    plot_acf(diff_series, lags=50, ax=plt.gca())
    plt.title(f'ACF Plot{title_suffix}')
    plt.grid(True)

    # Plot rolling mean and variance
    rolling_mean = diff_series.expanding().mean()
    rolling_var = diff_series.expanding().var()

    plt.subplot(3, 1, 3)
    plt.plot(rolling_mean, label='Rolling Mean', color='blue')
    plt.plot(rolling_var, label='Rolling Variance', color='orange')
    plt.title(f'Rolling Statistics{title_suffix}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Perform statistical tests
    print(f"\n{'='*50}")
    print(f"Analysis for {target_column}{title_suffix}")
    print('='*50)


    adf_test(diff_series)
    kpss_test(diff_series)

    return diff_series

# Apply different transformations and analyze them
print("\nAnalyzing Different Transformations...")
diff_1 = difference_and_plot(ts, diff_order=1)
# 1. Second Order Differencing
diff_2 = difference_and_plot(ts, diff_order=2)

# 2. Third Order Differencing
#diff_3 = difference_and_plot(ts, diff_order=3)


# Compare all transformations in one plot
plt.figure(figsize=(12, 8))
plt.plot(ts.diff().dropna(), label='1st Order Differencing', alpha=0.7)
plt.plot(diff_1, label='1st Order Differencing', alpha=0.7)
plt.plot(diff_2, label='2nd Order Differencing', alpha=0.7)
#plt.plot(diff_3, label='3rd Order Differencing', alpha=0.7)
plt.title('Comparison of Different Transformations')
plt.xlabel('Date')
plt.ylabel('Transformed Value')
plt.legend()
plt.grid(True)
plt.show()
#%%
def ACF_PACF_Plot(y, lags):
    """Plots ACF and PACF using the specified function"""
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)

    fig = plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.title('ACF')
    plt.xlabel("lag")
    plt.ylabel("mag")
    plot_acf(y, ax=plt.gca(), lags=lags)

    plt.subplot(212)
    plt.title('PACF')
    plt.xlabel("lag")
    plt.ylabel("mag")
    plot_pacf(y, ax=plt.gca(), lags=lags)

    fig.tight_layout(pad=3)
    plt.show()




# Plot using the specified function
ACF_PACF_Plot(diff_2, lags=50)
#%%
from statsmodels.tsa.seasonal import seasonal_decompose
# Function for seasonal decomposition
def seasonal_decomposition(series, period=6*24*365, model='additive'):
    decomposition = seasonal_decompose(series, model=model, period=period)

    # Plot decomposed components
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(series, label='Original')
    plt.legend(loc='upper left')
    plt.title('Original Time Series')

    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='upper left')
    plt.title('Trend Component')

    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal, label='Seasonal')
    plt.legend(loc='upper left')
    plt.title('Seasonal Component')

    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid, label='Residual')
    plt.legend(loc='upper left')
    plt.title('Residual Component')

    plt.tight_layout()
    plt.show()

    return decomposition

# Perform additive decomposition
decomposition_additive = seasonal_decomposition(ts, period=6*24*365, model='additive')

# Attempt multiplicative decomposition (will fail due to negative values)
try:
    decomposition_multiplicative = seasonal_decomposition(ts, period=24*365, model='multiplicative')
except ValueError as e:
    print(f"❌ Multiplicative decomposition failed: {e}")
    print("Multiplicative decomposition is not appropriate for this dataset because it contains zero or negative values.")
    print("We will proceed with additive decomposition only.")

# Function to calculate strength of trend and seasonality
def calculate_strength(trend, seasonal, residual):
    var_residual = np.var(residual, ddof=1)
    var_trend_residual = np.var(trend + residual, ddof=1)
    strength_trend = max(0, 1 - (var_residual / var_trend_residual))

    var_seasonal_residual = np.var(seasonal + residual, ddof=1)
    strength_seasonality = max(0, 1 - (var_residual / var_seasonal_residual))

    return strength_trend, strength_seasonality

# Remove NaN values from decomposed components
trend = decomposition_additive.trend.dropna()
seasonal = decomposition_additive.seasonal.dropna()
residual = decomposition_additive.resid.dropna()

# Calculate strength of trend and seasonality
strength_trend, strength_seasonality = calculate_strength(trend, seasonal, residual)
print(f"Strength of Trend: {strength_trend:.4f}")
print(f"Strength of Seasonality: {strength_seasonality:.4f}")
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Load data with proper datetime parsing
try:
    df = pd.read_csv("jena_climate_2009_2016.csv", parse_dates=['Date Time'],
                    dayfirst=True, index_col='Date Time')
except:
    # Try alternative parsing if default fails
    try:
        df = pd.read_csv("jena_climate_2009_2016.csv", sep=';',
                        parse_dates=['Date Time'], dayfirst=True,
                        index_col='Date Time')
    except Exception as e:
        print(f"Failed to load data: {e}")
        exit()

# Verify datetime conversion
print("Date range before cleaning:")
print(f"Start: {df.index.min()}, End: {df.index.max()}")

# Clean data - remove rows with NaT in index
df = df[~df.index.isna()]
df = df.sort_index()

# Verify after cleaning
print("\nDate range after cleaning:")
print(f"Start: {df.index.min()}, End: {df.index.max()}")
print(f"Total rows: {len(df)}")

# Select target variable
target_col = 'T (degC)'
if target_col not in df.columns:
    print(f"Target column '{target_col}' not found. Available columns:")
    print(df.columns.tolist())
    exit()

# Use recent 2 years for better performance
subset = df.loc['2009-01-01':'2011-12-31', target_col]

# Handle missing values (linear interpolation for gaps < 6 hours)
subset = subset.interpolate(limit=36)  # 6 readings/hour × 6 hours

# Split data (last 30 days for test)
test_days = 30
test_points = test_days * 144  # 144 = 6 readings/hour × 24 hours
train = subset.iloc[:-test_points]
test = subset.iloc[-test_points:]

print("\nData Split:")
print(f"Training: {len(train)} points ({len(train)/144:.1f} days)")
print(f"Testing: {len(test)} points ({test_days} days)")

# Holt-Winters model with 10-minute data
try:
    model = ExponentialSmoothing(
        train,
        seasonal_periods=144,  # Daily seasonality (6×24)
        trend='add',
        seasonal='add',
        initialization_method='estimated'
    ).fit()

    # Generate predictions
    predictions = model.forecast(len(test))

    # Calculate metrics
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)

    # Plot results (last 3 training days + first test week)
    plot_points = 3*144 + 7*144
    plt.figure(figsize=(16, 6))
    plt.plot(train.index[-3*144:], train[-3*144:], label='Training', color='blue')
    plt.plot(test.index[:7*144], test[:7*144], label='Actual', color='green')
    plt.plot(test.index[:7*144], predictions[:7*144],
             label='Predicted', color='red', linestyle='--')
    plt.title('10-minute Temperature Forecasting with Holt-Winters')
    plt.xlabel('DateTime')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nModel Performance:")
    print(f"RMSE: {rmse:.3f}°C")
    print(f"Mean Temperature: {subset.mean():.2f}°C")
    print(f"Relative RMSE: {rmse/subset.mean()*100:.1f}%")

except Exception as e:
    print(f"\nModel failed: {e}")
    print("Suggestions:")
    print("1. Try with hourly data: df.resample('1H').mean()")
    print("2. Reduce seasonal_periods (e.g., 24 for daily seasonality in hourly data)")
    print("3. Use less data (single year instead of two)")
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS, add_constant
import scipy.linalg as la

# 1. Data Loading and Preparation --------------------------------------------
def load_and_prepare_data(filepath):
    """Load and prepare the Jena climate dataset"""
    df = pd.read_csv(filepath, parse_dates=['Date Time'],
                    dayfirst=True, index_col='Date Time')
    df = df[~df.index.isna()].sort_index()

    # Select features and target
    target_col = 'T (degC)'
    features = [
        'p (mbar)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
        'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)',
        'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)',
        'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'
    ]

    # Add temporal features
    df['hour'] = df.index.hour
    df['day_of_year'] = df.index.dayofyear
    features += ['hour', 'day_of_year']

    # Handle missing values
    X = df[features].interpolate(limit=6)  # Fill up to 1 hour gaps
    y = df[target_col].interpolate(limit=6)

    return X, y, features

# 2. Correlation Analysis ---------------------------------------------------
def analyze_correlations(X, figsize=(14, 10)):
    """Calculate and visualize feature correlations"""
    corr_matrix = X.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                center=0, mask=np.triu(corr_matrix), annot_kws={"size": 8})
    plt.title("Feature Correlation Matrix", pad=20)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Identify high correlations
    high_corr = [(corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i,j])
                 for i in range(len(corr_matrix.columns))
                 for j in range(i+1, len(corr_matrix.columns))
                 if abs(corr_matrix.iloc[i,j]) > 0.8]

    print("\nHighly Correlated Features (|r| > 0.8):")
    for pair in high_corr:
        print(f"{pair[0]} & {pair[1]}: {pair[2]:.2f}")

    return corr_matrix

# 3. VIF Analysis ----------------------------------------------------------
def calculate_vif(X_df, threshold=5):
    """Iteratively remove features with high VIF"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_df.values, i)
                      for i in range(X_df.shape[1])]

    max_vif = vif_data["VIF"].max()
    if max_vif > threshold:
        remove_feature = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
        print(f"Removing {remove_feature} (VIF={max_vif:.2f})")
        X_df = X_df.drop(columns=[remove_feature])
        return calculate_vif(X_df, threshold)
    return X_df, vif_data

# Updated PCA/SVD Analysis Function
def perform_pca(X, variance_threshold=0.95, sample_size=10000):
    """Perform PCA with specified variance threshold on sampled data"""
    # Sample the data if too large
    if len(X) > sample_size:
        X_sampled = X.sample(n=sample_size, random_state=42)
        print(f"\nSampling {sample_size} points from {len(X)} for PCA/SVD")
    else:
        X_sampled = X.copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sampled)

    try:
        # SVD Decomposition on sampled data
        U, s, Vt = la.svd(X_scaled, full_matrices=False)
        cond_number = s.max() / s.min()

        print(f"\nCondition Number: {cond_number:.2f}")
        print("Interpretation: Values >1000 indicate strong multicollinearity")

        # PCA
        pca = PCA(n_components=variance_threshold)
        X_pca = pca.fit_transform(X_scaled)

        print(f"\nPCA Results: {X_scaled.shape[1]} features → {X_pca.shape[1]} components")
        print("Explained Variance Ratio:", pca.explained_variance_ratio_)

        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(pca.explained_variance_ratio_)+1),
                pca.explained_variance_ratio_,
                label='Individual explained variance')
        plt.step(range(1, len(pca.explained_variance_ratio_)+1),
                 np.cumsum(pca.explained_variance_ratio_),
                 where='mid',
                 label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend()
        plt.grid()
        plt.show()

        return pca, X_pca, cond_number

    except Exception as e:
        print(f"\nPCA/SVD Error: {str(e)}")
        print("Trying alternative approach with randomized SVD...")

        # Fallback to randomized SVD
        from sklearn.utils.extmath import randomized_svd
        U, s, Vt = randomized_svd(X_scaled,
                                 n_components=min(X_scaled.shape)-1,
                                 random_state=42)

        cond_number = s.max() / s.min()
        print(f"\nCondition Number (randomized): {cond_number:.2f}")

        # Continue with PCA as before
        pca = PCA(n_components=variance_threshold)
        X_pca = pca.fit_transform(X_scaled)

        print(f"\nPCA Results: {X_scaled.shape[1]} features → {X_pca.shape[1]} components")
        print("Explained Variance Ratio:", pca.explained_variance_ratio_)

        # Plotting code remains the same...
        return pca, X_pca, cond_number

# 5. Backward Stepwise Regression ------------------------------------------
def backward_stepwise_selection(X, y, threshold_out=0.05):
    """Perform backward stepwise feature selection"""
    included = list(X.columns)
    while True:
        changed = False
        model = OLS(y, add_constant(X[included])).fit()
        pvalues = model.pvalues.iloc[1:]  # Ignore intercept
        worst_pval = pvalues.max()

        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            print(f"Removed {worst_feature} (p={worst_pval:.4f})")
            changed = True

        if not changed:
            break

    return included

# Main Execution -----------------------------------------------------------
def main():
    # 1. Load and prepare data
    X, y, features = load_and_prepare_data("jena_climate_2009_2016.csv")

    # 2. Correlation analysis
    print("\n=== Correlation Analysis ===")
    corr_matrix = analyze_correlations(X)

    # 3. VIF analysis (on non-temporal features first)
    print("\n=== VIF Analysis ===")
    non_temp_features = [f for f in features if f not in ['hour', 'day_of_year']]
    X_vif, vif_results = calculate_vif(X[non_temp_features])

    # Add temporal features back
    X_vif['hour'] = X['hour']
    X_vif['day_of_year'] = X['day_of_year']

    print("\nFinal VIF Results:")
    print(vif_results.sort_values("VIF", ascending=False))

    # 4. PCA/SVD analysis
    print("\n=== PCA/SVD Analysis ===")
    continuous_features = [f for f in X_vif.columns if f not in ['hour', 'day_of_year']]
    pca, X_pca, cond_number = perform_pca(X_vif[continuous_features])

    # 5. Backward stepwise regression
    print("\n=== Backward Stepwise Regression ===")
    selected_features = backward_stepwise_selection(X_vif, y)

    # Final report
    print("\n=== FEATURE SELECTION REPORT ===")

    print("\n1. VIF Analysis Results:")
    print(f"   - Removed {len(non_temp_features) - X_vif.shape[1]} features due to high VIF")
    print(f"   - Highest remaining VIF: {vif_results['VIF'].max():.2f}")

    print("\n2. PCA/SVD Findings:")
    print(f"   - Condition number: {cond_number:.2f}")
    print(f"   - PCA reduced to {X_pca.shape[1]} components (95% variance)")

    print("\n3. Backward Stepwise Regression:")
    print(f"   - Selected {len(selected_features)} features")
    print("   - Final features:", selected_features)

    return {
        'X_vif': X_vif,
        'selected_features': selected_features,
        'X_pca': X_pca,
        'pca_model': pca,
        'vif_results': vif_results
    }

if __name__ == "__main__":
  results = main()
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import SimpleExpSmoothing

# Load and prepare data
df = pd.read_csv("jena_climate_2009_2016.csv", parse_dates=['Date Time'],
                dayfirst=True, index_col='Date Time')
df = df[~df.index.isna()].sort_index()

# Use temperature data with 10-min frequency
target_col = 'T (degC)'
ts = df[target_col].interpolate(limit=6)  # Fill up to 1 hour gaps

# Split data (last 7 days for test - more manageable for visualization)
test_days = 7
test_points = test_days * 144  # 6 readings/hour × 24 hours × 7 days
train, test = ts.iloc[:-test_points], ts.iloc[-test_points:]

print(f"Training period: {train.index.min()} to {train.index.max()}")
print(f"Testing period: {test.index.min()} to {test.index.max()}")
print(f"Training points: {len(train)}, Test points: {len(test)}")

# 1. Average Method =======================================
average_val = train.mean()
average_pred = pd.Series([average_val] * len(test), index=test.index)

plt.figure(figsize=(12, 5))
plt.plot(train.iloc[-1008:], label='Training (last 7 days)')  # Show last week of training
plt.plot(test, label='Actual', color='black')
plt.plot(average_pred, label='Average Forecast', linestyle='--')
plt.title(f"Average Method Forecast\n(Constant Prediction = {average_val:.2f}°C)")
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

mse_avg = mean_squared_error(test, average_pred)
print(f"\nAverage Method MSE: {mse_avg:.4f}, RMSE: {np.sqrt(mse_avg):.4f}")

#%%
# 2. Naïve Method ========================================
naive_val = train.iloc[-1]
naive_pred = pd.Series([naive_val] * len(test), index=test.index)

plt.figure(figsize=(12, 5))
plt.plot(train.iloc[-1008:], label='Training (last 7 days)')
plt.plot(test, label='Actual', color='black')
plt.plot(naive_pred, label='Naïve Forecast', linestyle='--')
plt.title(f"Naïve Method Forecast\n(Last Observed Value = {naive_val:.2f}°C)")
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

mse_naive = mean_squared_error(test, naive_pred)
print(f"\nNaïve Method MSE: {mse_naive:.4f}, RMSE: {np.sqrt(mse_naive):.4f}")


#%%
# 3. Drift Method ========================================
n = len(train)
drift = (train.iloc[-1] - train.iloc[0]) / (n - 1)
drift_pred = pd.Series(
    [train.iloc[-1] + h * drift for h in range(1, len(test)+1)],
    index=test.index
)

plt.figure(figsize=(12, 5))
plt.plot(train.iloc[-1008:], label='Training (last 7 days)')
plt.plot(test, label='Actual', color='black')
plt.plot(drift_pred, label='Drift Forecast', linestyle='--')
plt.title(f"Drift Method Forecast\n(Slope = {drift:.4f}°C per timestep)")
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

mse_drift = mean_squared_error(test, drift_pred)
print(f"\nDrift Method MSE: {mse_drift:.4f}, RMSE: {np.sqrt(mse_drift):.4f}")


#%%
# 4. Simple Exponential Smoothing ========================
model_ses = SimpleExpSmoothing(train).fit(optimized=True)
ses_pred = model_ses.forecast(len(test))

plt.figure(figsize=(12, 5))
plt.plot(train.iloc[-1008:], label='Training (last 7 days)')
plt.plot(test, label='Actual', color='black')
plt.plot(ses_pred, label='SES Forecast', linestyle='--')
plt.title(f"Simple Exponential Smoothing Forecast\n(Alpha = {model_ses.model.params['smoothing_level']:.4f})")
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

mse_ses = mean_squared_error(test, ses_pred)
print(f"\nSES Method MSE: {mse_ses:.4f}, RMSE: {np.sqrt(mse_ses):.4f}")

#%%
# Comparison =============================================
results = pd.DataFrame({
    'Model': ['Average', 'Naïve', 'Drift', 'SES'],
    'MSE': [mse_avg, mse_naive, mse_drift, mse_ses],
    'RMSE': [np.sqrt(mse_avg), np.sqrt(mse_naive), np.sqrt(mse_drift), np.sqrt(mse_ses)]
}).sort_values('RMSE')

print("\nModel Comparison:")
print(results)

# Combined Forecast Plot
plt.figure(figsize=(14, 7))
plt.plot(test.index, test, label='Actual', color='black', linewidth=2)
plt.plot(test.index, average_pred, label=f'Average (RMSE={results.iloc[0]["RMSE"]:.2f})', alpha=0.7)
plt.plot(test.index, naive_pred, label=f'Naïve (RMSE={results.iloc[1]["RMSE"]:.2f})', alpha=0.7)
plt.plot(test.index, drift_pred, label=f'Drift (RMSE={results.iloc[2]["RMSE"]:.2f})', alpha=0.7)
plt.plot(test.index, ses_pred, label=f'SES (RMSE={results.iloc[3]["RMSE"]:.2f})', alpha=0.7)
plt.title('Base Models Forecast Comparison')
plt.ylabel('Temperature (°C)')
plt.xlabel('DateTime')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from math import sqrt
from tabulate import tabulate
import os

# 1. Data Preparation with Robust Datetime Handling
def load_and_prepare_data():
    """Load and prepare Jena Climate dataset with proper datetime parsing"""
    try:
        # Load data with explicit datetime format
        df = pd.read_csv('jena_climate_2009_2016.csv')

        # Convert datetime with exact format specification
        df['Date Time'] = pd.to_datetime(df['Date Time'],
                                        format='%d.%m.%Y %H:%M:%S',
                                        dayfirst=True,
                                        errors='coerce')

        # Drop rows with invalid dates if any
        df = df.dropna(subset=['Date Time'])

        # Set index and sort
        df = df.set_index('Date Time').sort_index()

        # Filter date range (using ISO format)
        df = df.loc['2009-01-01':'2011-12-31']

        # Handle missing values
        df = df.interpolate(limit=6)

        # Feature engineering
        df['hour'] = df.index.hour
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear/365)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear/365)

        # Selected features
        features = ['Tdew (degC)', 'rh (%)', 'VPdef (mbar)', 'wv (m/s)', 'hour', 'day_sin', 'day_cos']
        target = 'T (degC)'

        return df[features], df[target]

    except Exception as e:
        print(f"Error in load_and_prepare_data: {e}")
        print("Files in directory:", os.listdir())
        raise

# Load data
try:
    X, y = load_and_prepare_data()
    print("Data loaded successfully. First 5 rows:")
    print(X.head())
except Exception as e:
    print(f"Failed to load data: {e}")
    exit()

# 2. Train-Test Split
split_point = int(0.8 * len(X))
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# 3. Model Development
def build_model(X_train, y_train):
    """Build and return regression model"""
    X_train_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_const).fit()
    return model

model = build_model(X_train, y_train)

# 4. Model Evaluation
def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    X_test_const = sm.add_constant(X_test, has_constant='add')
    y_pred = model.predict(X_test_const)

    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': sqrt(mean_squared_error(y_test, y_pred)),
        'R-squared': model.rsquared,
        'Adj. R-squared': model.rsquared_adj,
        'AIC': model.aic,
        'BIC': model.bic,
        'F-statistic': model.fvalue,
        'F-pvalue': model.f_pvalue
    }

    return y_pred, metrics

y_pred, metrics = evaluate_model(model, X_test, y_test)

# 5. Corrected One-step Ahead Prediction
def one_step_ahead_prediction(model, X_train, X_test, y_test):
    """Perform one-step ahead prediction with proper constant handling"""
    y_pred_os = []

    # Initialize with last training point (with constant)
    current_X = sm.add_constant(X_train.iloc[-1:].copy(), has_constant='add')

    for i in range(len(X_test)):
        # Make prediction
        y_pred_os.append(model.predict(current_X)[0])

        # Update features for next step (maintaining constant)
        if i < len(X_test) - 1:
            current_X = sm.add_constant(X_test.iloc[i:i+1].copy(), has_constant='add')

    return {
        'one_step_mse': mean_squared_error(y_test, y_pred_os),
        'one_step_pred': y_pred_os
    }

os_results = one_step_ahead_prediction(model, X_train, X_test, y_test)

# 6. Time Series Cross Validation
def time_series_cv(X_train, y_train, n_splits=5):
    """Perform time series cross validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        m = build_model(X_tr, y_tr)
        X_val_const = sm.add_constant(X_val, has_constant='add')
        y_val_pred = m.predict(X_val_const)
        cv_scores.append(mean_squared_error(y_val, y_val_pred))

    return {
        'cv_mean_mse': np.mean(cv_scores),
        'cv_std_mse': np.std(cv_scores)
    }

cv_results = time_series_cv(X_train, y_train)

# 7. Residual Analysis
def analyze_residuals(y_true, y_pred):
    """Perform residual diagnostics"""
    residuals = y_true - y_pred

    # Statistical tests
    acf_vals = acf(residuals, nlags=20, fft=True)
    lb_test = acorr_ljungbox(residuals, lags=[5], return_df=True)

    # Plot ACF
    plt.figure(figsize=(12, 6))
    plt.stem(range(len(acf_vals)), acf_vals)
    plt.axhline(1.96/np.sqrt(len(residuals)), color='r', linestyle='--', label='95% CI')
    plt.axhline(-1.96/np.sqrt(len(residuals)), color='r', linestyle='--')
    plt.title('ACF of Residuals')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        'residual_mean': np.mean(residuals),
        'residual_var': np.var(residuals),
        'q_value': lb_test['lb_stat'].values[0],
        'q_pvalue': lb_test['lb_pvalue'].values[0],
        'acf': acf_vals
    }

residual_diag = analyze_residuals(y_test, y_pred)

# 8. Visualization
def plot_results(y_train, y_test, y_pred):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(14, 6))
    plt.plot(y_train.index, y_train, label='Train (Actual)', color='blue', alpha=0.7)
    plt.plot(y_test.index, y_test, label='Test (Actual)', color='green')
    plt.plot(y_test.index, y_pred, label='Test (Predicted)', color='red', linestyle='--')
    plt.title('Temperature Prediction: Actual vs Predicted')
    plt.xlabel('Date Time')
    plt.ylabel('Temperature (degC)')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_results(y_train, y_test, y_pred)

# 9. Multicollinearity Check
def check_multicollinearity(X):
    """Calculate VIF for features"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

vif_results = check_multicollinearity(X_train)

# 10. Results Compilation and Display
print("\n=== COMPLETE REGRESSION ANALYSIS ===")
print(model.summary())

print("\n=== HYPOTHESIS TESTS ===")
print("F-test p-value:", model.f_pvalue)
print("Significant coefficients (p < 0.05):")
print(model.pvalues[model.pvalues < 0.05])

print("\n=== CROSS VALIDATION RESULTS ===")
print(f"Mean MSE: {cv_results['cv_mean_mse']:.4f}")
print(f"Std MSE: {cv_results['cv_std_mse']:.4f}")

print("\n=== MODEL METRICS ===")
metrics_df = pd.DataFrame([metrics])
print(tabulate(metrics_df, headers='keys', tablefmt='grid', showindex=False))

print("\n=== RESIDUAL DIAGNOSTICS ===")
print(f"Q-value: {residual_diag['q_value']:.4f}")
print(f"Q-test p-value: {residual_diag['q_pvalue']:.4f}")
print(f"Residual mean: {residual_diag['residual_mean']:.6f}")
print(f"Residual variance: {residual_diag['residual_var']:.4f}")

print("\n=== ONE-STEP AHEAD PREDICTION ===")
print(f"MSE: {os_results['one_step_mse']:.4f}")
print(f"Improvement over normal prediction: {(metrics['MSE']-os_results['one_step_mse'])/metrics['MSE']*100:.2f}%")

print("\n=== MULTICOLLINEARITY CHECK ===")
print(tabulate(vif_results, headers='keys', tablefmt='grid', showindex=False))
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from IPython.display import display

# 1. Data Preparation
df = pd.read_csv('jena_climate_2009_2016.csv')
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S', dayfirst=True)
df = df.set_index('Date Time').sort_index()
y = df.loc['2009-01-01':'2011-12-31', 'T (degC)']

# 2. Make Data Stationary (First Difference Only)
y_diff = y.diff().dropna()  # First difference (d=1)
y_diff = y_diff.diff().dropna()
# 3. Compute GPAC Table
ry = acf(y_diff, nlags=20, fft=True)

def gpac_table(acf, max_k: 7, max_j: 7, model_nb: int = None):
    gpac = np.zeros((max_j, max_k))
    for j in range(max_j):
        for k in range(1, max_k + 1):
            if model_nb is not None and j >= model_nb + 1:
                gpac[j, k - 1] = np.nan
                continue
            try:

                D = np.zeros((k, k))
                for i in range(k):
                    for m in range(k):
                        lag = abs(j + i - m)
                        D[i, m] = acf[lag] if lag < len(acf) else 0

                N = D.copy()
                for i in range(k):
                    lag = j + i + 1
                    N[i, -1] = acf[lag] if lag < len(acf) else 0

                det_D = np.linalg.det(D)
                det_N = np.linalg.det(N)

                if np.isclose(det_D, 0):
                    if np.isclose(det_N, 0):
                        gpac[j, k - 1] = np.nan
                    else:
                        gpac[j, k - 1] = np.inf
                else:
                    gpac[j, k - 1] = det_N / det_D

            except Exception:
                gpac[j, k - 1] = np.nan

    df = pd.DataFrame(gpac, index=[f"j={j}" for j in range(max_j)], columns=[f"k={k+1}" for k in range(max_k)])
    df = df.round(3)
    df = df.replace(-0.0, 0.0)
    df = df.applymap(lambda x: 0.0 if np.isclose(x, 0, atol=1e-3) else x)
    return df

def plot_gpac_table(gpac_df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(gpac_df, annot=True, fmt=".3f", cmap="rocket_r", cbar=True, vmin=-2, vmax=2)
    plt.title("Generalized Partial Autocorrelation (GPAC) Table", fontsize=14)
    plt.xlabel("k (AR order)")
    plt.ylabel("j (MA order)")
    plt.tight_layout()
    plt.show()

gpac_table = np.round(gpac_table(ry, 7, 7),2)


# Plot
plt.figure(figsize=(10, 6))
sns.heatmap(gpac_table, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("GPAC Table")
plt.xlabel("k")
plt.ylabel("j")
plt.show()


# 4. Model Development


# ACF/PACF Plots
print("\nACF/PACF Analysis:")
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,8))
plot_acf(y_diff, lags=40, ax=ax1, title="ACF of Differenced Series")
plot_pacf(y_diff, lags=40, ax=ax2, title="PACF of Differenced Series")
plt.tight_layout()
plt.show()

#%%
def compute_error(theta, y, na, nb):
    """Compute residuals for ARMA model"""
    N = len(y)
    e = np.zeros(N)

    for t in range(max(na, nb), N):
        ar_part = sum([-theta[i] * y[t - i - 1] for i in range(na)]) if na > 0 else 0
        ma_part = sum([theta[na + j] * e[t - j - 1] for j in range(nb)]) if nb > 0 else 0
        e[t] = y[t] + ar_part - ma_part

    return e[max(na, nb):]


def compute_jacobian(theta, y, na, nb, delta=1e-7):
    """Compute Jacobian matrix numerically"""
    n_params = na + nb
    N = len(y)
    X = np.zeros((N - max(na, nb), n_params))
    base_error = compute_error(theta, y, na, nb)

    for i in range(n_params):
        theta_perturbed = theta.copy()
        theta_perturbed[i] += delta
        perturbed_error = compute_error(theta_perturbed, y, na, nb)
        X[:, i] = (perturbed_error - base_error) / delta

    return X


def levenberg_marquardt(y, theta_init, na, nb, max_iter=100, tol=1e-6):
    """Implement LM algorithm for ARMA estimation"""
    theta = theta_init.copy()
    mu = 0.01
    sse_history = []

    for iteration in range(max_iter):
        e = compute_error(theta, y, na, nb)
        sse = np.sum(e ** 2)
        sse_history.append(sse)

        J = compute_jacobian(theta, y, na, nb)
        grad = J.T @ e
        hess = J.T @ J

        # LM update
        delta = np.linalg.solve(hess + mu * np.eye(len(theta)), -grad)
        theta_new = theta + delta

        e_new = compute_error(theta_new, y, na, nb)
        sse_new = np.sum(e_new ** 2)

        if sse_new < sse:
            if np.linalg.norm(delta) < tol:
                break
            theta = theta_new
            mu = max(mu / 10, 1e-10)
        else:
            mu = min(mu * 10, 1e10)

    return theta, sse_history



n_samples, na2, nb2 = 10000, 0, 2
y=y_diff

# Phase I analysis for Question 1
theta_init = np.zeros(na2 + nb2) + 0.1
theta_est, sse_history = levenberg_marquardt(y, theta_init, na2, nb2)
#%%
n_samples, na, nb = 10000, 1, 2
y=y_diff

# Phase I analysis for Question 1
theta_init = np.zeros(na + nb) + 0.1
theta_est, sse_history = levenberg_marquardt(y, theta_init, na, nb)


#%%
def analyze_model(y, theta_est, na, nb, example_name):
    print(f"\n=== Analyzing {example_name} ===")

    # 1. Display estimated parameters
    print("\n1. Estimated Parameters:")
    if na > 0:
        print(f"AR coefficients: {np.round(theta_est[:na], 3)} ")
    if nb > 0:
        print(f"MA coefficients: {np.round(theta_est[na:], 3)}")

    # 2. Confidence intervals with justification
    N = len(y)
    e = compute_error(theta_est, y, na, nb)
    sigma2 = np.sum(e ** 2) / (N - (na + nb))
    J = compute_jacobian(theta_est, y, na, nb)
    cov_matrix = sigma2 * np.linalg.inv(J.T @ J)
    std_err = np.sqrt(np.diag(cov_matrix))

    print("\n2. Confidence Intervals and Significance:")
    for i in range(na + nb):
        ci_low = theta_est[i] - 1.96 * std_err[i]
        ci_high = theta_est[i] + 1.96 * std_err[i]
        param_type = "AR" if i < na else "MA"
        param_num = i + 1 if i < na else i + 1 - na
        print(f"{param_type}{param_num}: {theta_est[i]:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
        print(
            f"Justification: The parameter is {'statistically significant' if ci_low * ci_high > 0 else 'not significant'} "
            f"because the 95% CI {'does not contain' if ci_low * ci_high > 0 else 'contains'} zero.")

    # 3. Covariance matrix
    print("\n3. Covariance Matrix:")
    print(np.round(cov_matrix, 6))

    # 4. Error variance
    print(f"\n4. Error Variance: {sigma2:.3f}")

    # 5. Poles and zeros with justification
    print("\n5. Poles and Zeros Analysis:")
    ar_poly = np.r_[1, -np.array(theta_est[:na])] if na > 0 else [1]
    ma_poly = np.r_[1, np.array(theta_est[na:])] if nb > 0 else [1]

    poles = np.roots(ar_poly) if na > 0 else []
    zeros = np.roots(ma_poly) if nb > 0 else []

    print("Poles (AR roots):", np.round(poles, 3))
    print("Zeros (MA roots):", np.round(zeros, 3))

    # Check for pole-zero cancellation
    cancellation = False
    if na > 0 and nb > 0:
        for pole in poles:
            for zero in zeros:
                if np.isclose(pole, zero, atol=1e-2):
                    cancellation = True
    print("Justification: " +
          ("Pole-zero cancellation detected, suggesting possible model overparameterization."
           if cancellation else "No pole-zero cancellation detected."))

    # 6. SSE vs iterations plot
    print("\n6. SSE Convergence Plot:")
    plt.figure(figsize=(10, 5))
    plt.plot(sse_history, 'b-o')
    plt.title(f"SSE vs Iterations - {example_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Sum of Squared Errors")
    plt.grid(True)
    plt.show()

print(analyze_model(y, theta_est, na, nb, "1,2 ARMA Model"))
print(analyze_model(y, theta_est, na2, nb2, "0,2 ARMA Model"))
#%%


# Q-test

def q_test(residuals, lags=50, model_df=0, alpha=0.05):

    residuals = np.asarray(residuals)
    N = len(residuals)
    residuals -= np.mean(residuals)
    var_e = np.var(residuals)

    Q = 0
    for tau in range(1, lags + 1):
        autocov = np.sum(residuals[tau:] * residuals[:-tau]) / (N - tau)
        r_tau = autocov / var_e
        Q += r_tau ** 2

    Q_stat = N * Q
    dof = lags - model_df
    Q_crit = chi2.ppf(1 - alpha, df=dof)

    print(f"\n--- Q-Test Summary ---")
    print(f"Q-statistic              : {Q_stat:.4f}")
    print(f"Chi-square Critical (α={alpha}, dof={dof}) : {Q_crit:.4f}")
    print("Result                   :",
          "✅ Residuals are white (Q < Q*)" if Q_stat < Q_crit
          else "❌ Residuals show autocorrelation (Q > Q*)")

    return Q_stat, Q_crit, dof

# s-test

def s_test(e, u, theta_est, nb, nf, K=20, significance=0.05):
    N = len(e)
    e = e - np.mean(e)

    f = np.r_[1, theta_est[nb - 1: nb - 1 + nf]]

    alpha_t = np.zeros_like(u)
    for t in range(len(f), N):
        alpha_t[t] = u[t] - sum(f[j] * alpha_t[t - j] for j in range(1, len(f)))

    alpha_t = alpha_t - np.mean(alpha_t)

    sigma_e = float(np.std(e))
    sigma_a = float(np.std(alpha_t))

    S = 0.0
    r_vals = []
    for tau in range(K + 1):
        R_ae = np.sum(alpha_t[:N - tau] * e[tau:]) / (N - tau)
        r_ae = R_ae / (sigma_a * sigma_e)
        r_vals.append(r_ae)
        S += r_ae ** 2

    S_stat = float(N * S)
    dof = K - (nb - 1) - nf
    S_crit = float(chi2.ppf(1 - significance, df=dof))

    print(f"S-stat: {S_stat:.4f}")
    print(f"Chi-square S* (α={significance}, DOF={dof}): {S_crit:.4f}")
    if S_stat < S_crit:
        print("G(q) is accurate (S < S*)")
    else:
        print("G(q) may be misspecified (S > S*)")

    return S_stat, S_crit, dof, r_vals


# ARMA FORECASTING

def forecast_arma(y, phi, theta, residuals, steps=1):

    p = len(phi)
    q = len(theta)
    y_hat = []
    y_hist = list(y)
    e_hist = list(residuals)

    for h in range(steps):
        ar_part = sum(phi[i] * y_hist[-i - 1] for i in range(p))
        ma_part = sum(theta[j] * (e_hist[-j - 1] if h == 0 else 0) for j in range(q))
        y_next = ar_part + ma_part
        y_hat.append(y_next)
        y_hist.append(y_next)
        e_hist.append(0)

    return y_hat


# ARMA

def one_step_forecast_arma_plot(y, e, phi, theta, n_plot=20):

    y = np.asarray(y)
    e = np.asarray(e)
    p, q = len(phi), len(theta)
    N = len(y)
    lag = max(p, q)

    # compute one-step forecasts
    y_hat = np.zeros(N)
    for t in range(lag, N):
        ar = sum(phi[i] * y[t-i-1]   for i in range(p))
        ma = sum(theta[j] * e[t-j-1] for j in range(q))
        y_hat[t] = ar + ma

    actual   = y[lag:]
    forecast = y_hat[lag:]
    m = min(n_plot, len(actual))

    # plot
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(m), actual[:m],   'o-', label='Actual')
    plt.plot(np.arange(m), forecast[:m], 'x--', label='1-step Forecast')
    plt.title(f'One-Step Forecast vs Actual (first {m} points)')
    plt.xlabel('t')
    plt.legend()
    plt.grid(True)
    plt.show()

    res_var = np.var(actual[:m] - forecast[:m])
    print(f"1-step residual var       = {res_var:.4f}")

    return forecast

def h_step_forecast_arma_plot(y, e, phi, theta, h, y_actual=None):

    y = np.asarray(y)
    e = np.asarray(e)
    p, q = len(phi), len(theta)

    # prepare history
    y_hist = list(y)
    e_hist = list(e)
    forecasts = []

    for step in range(h):
        ar = sum(phi[i] * y_hist[-i-1] for i in range(p))
        ma = sum(theta[j] * e[-j-1] if step == 0 else 0 for j in range(q))
        y_next = ar + ma
        forecasts.append(y_next)
        y_hist.append(y_next)
        e_hist.append(0)

    # plot if true future provided
    if y_actual is not None:
        actual = np.asarray(y_actual)
        m = min(h, len(actual))
        plt.figure(figsize=(8, 3))
        plt.plot(np.arange(m), actual[:m],   'o-', label='Actual')
        plt.plot(np.arange(m), forecasts[:m], 'x--', label=f'{h}-step Forecast')
        plt.title(f'{h}-Step Forecast vs Actual')
        plt.xlabel('h')
        plt.legend()
        plt.grid(True)
        plt.show()

        fc_var = np.var(actual[:m] - forecasts[:m])
        print(f"{h}-step forecast var     = {fc_var:.4f}")

    return forecasts




# SARIMA

def forecast_sarima_one_step(data, t, phi, d, theta, seasonal_period, Phi=None, D=0, Theta=None):
    p, P = len(phi), len(Phi or [])
    y_hat = 0
    for i in range(1, p + 1):
        y_hat += phi[i - 1] * data[t - i]
    if Phi:
        for i in range(1, P + 1):
            y_hat += Phi[i - 1] * data[t - i * seasonal_period]
    return y_hat

def forecast_sarima_h_step(data, t0, hmax, phi, d, theta, seasonal_period, Phi=None, D=0, Theta=None):
    p, P = len(phi), len(Phi or [])
    y_pred = np.zeros(hmax)
    for h in range(1, hmax + 1):
        forecast = 0
        for i in range(1, p + 1):
            idx = t0 + h - i
            term = data[idx] if idx <= t0 else y_pred[idx - t0 - 1]
            forecast += phi[i - 1] * term
        if Phi:
            for i in range(1, P + 1):
                idx = t0 + h - i * seasonal_period
                term = data[idx] if idx <= t0 else y_pred[idx - t0 - 1]
                forecast += Phi[i - 1] * term
        y_pred[h - 1] = forecast
    return y_pred

def plot_sarima_forecasts(data, one_step_preds, t_start, h_preds, t0, hmax):
    # Plot one-step forecast
    actual_one_step = data[t_start + 1: t_start + 1 + len(one_step_preds)]
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(t_start + 1, t_start + 1 + len(actual_one_step)), actual_one_step, label="Actual", marker='o')
    plt.plot(np.arange(t_start + 1, t_start + 1 + len(one_step_preds)), one_step_preds, label="1-step Forecast", marker='x')
    plt.title("1-step Forecast vs Actual")
    plt.xlabel("Time Index")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot h-step forecast
    actual_h = data[t0 + 1: t0 + 1 + hmax]
    h_preds = h_preds[:len(actual_h)]
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(t0 + 1, t0 + 1 + len(actual_h)), actual_h, label="Actual", marker='o')
    plt.plot(np.arange(t0 + 1, t0 + 1 + len(h_preds)), h_preds, label=f"{hmax}-step Forecast", marker='x')
    plt.title(f"{hmax}-step Forecast from t = {t0}")
    plt.xlabel("Time Index")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print error variance
    residuals = actual_one_step - one_step_preds
    forecast_errors = actual_h - h_preds
    print(f"1-step Residual Variance: {np.var(residuals):.4f}")
    print(f"{hmax}-step Forecast Error Variance: {np.var(forecast_errors):.4f}")
    print(f"Variance Ratio (Test/Train): {np.var(forecast_errors) / np.var(residuals):.4f}")

#  BOX JENKINS

def estimate_impulse_response(u, y, K):
    N = len(u)
    Ru = np.array([np.sum(u[:N - tau] * u[tau:]) / (N - tau) for tau in range(K + 1)])
    Ruy = np.array([np.sum(u[:N - tau] * y[tau:]) / (N - tau) for tau in range(K + 1)])
    R_u_matrix = toeplitz(Ru[:K + 1])
    g_hat = np.linalg.solve(R_u_matrix, Ruy[:K + 1])
    return g_hat

# --- 2. GPAC Table Generator ---
def gpac_table_bj(acf_values, max_k=7, max_j=7):
    gpac = np.zeros((max_j, max_k))
    for j in range(max_j):
        for k in range(1, max_k + 1):
            try:
                D = np.array([[acf_values[abs(j + i - m)] for m in range(k)] for i in range(k)])
                N = D.copy()
                for i in range(k):
                    N[i, -1] = acf_values[j + i + 1]
                det_D = np.linalg.det(D)
                det_N = np.linalg.det(N)
                if np.isclose(det_D, 0):
                    gpac[j, k - 1] = np.nan if np.isclose(det_N, 0) else np.inf
                else:
                    gpac[j, k - 1] = det_N / det_D
            except Exception:
                gpac[j, k - 1] = np.nan
    return pd.DataFrame(gpac, index=[f"j={j}" for j in range(max_j)],
                        columns=[f"k={k + 1}" for k in range(max_k)])


def compute_residuals_bj(theta, y, u, nb, nf, nc, nd):
    N = len(y)
    y_g = np.zeros(N)
    e = np.zeros(N)

    b = np.r_[1, theta[:nb - 1]]
    f = np.r_[1, theta[nb - 1:nb - 1 + nf]]
    c = np.r_[1, theta[nb - 1 + nf:nb - 1 + nf + nc]]
    d = np.r_[1, -theta[nb - 1 + nf + nc:]]

    for t in range(max(len(b), len(f)), N):
        y_g[t] = sum(b[i] * u[t - i] for i in range(len(b))) - sum(f[j] * y_g[t - j] for j in range(1, len(f)))

    residual = y - y_g

    for t in range(max(len(c), len(d)), N):
        num = sum(d[j] * residual[t - j] for j in range(len(d)))
        den = sum(c[i] * e[t - i] for i in range(1, len(c)))
        e[t] = num - den

    return e

# --- 4. One-Step Forecast Function (Manual BJ) ---
def forecast_bj_1step(y, u, e, theta, nb, nf, nc, nd, steps=20, start=0):
    b1, f1, c1, d1 = theta
    N = len(y)
    yhat = []
    idx = []
    for t in range(start, start + steps):
        if t < 1 or t >= N - 1:
            continue
        val = b1 * u[t] - f1 * y[t] + c1 * e[t] - d1 * e[t - 1]
        yhat.append(val)
        idx.append(t)
    return np.array(yhat), idx

# --- 5. H-Step Forecast Function (recursive) ---
def forecast_bj_hstep(y, u, e, theta, nb, nf, nc, nd, steps=20, start=0):
    b1, f1, c1, d1 = theta
    yhat = []
    y_current = list(y[:start + 1])
    e_current = list(e[:start + 1])
    for h in range(steps):
        t = start + h
        val = b1 * u[t] - f1 * y_current[-1] + c1 * e_current[-1] - d1 * e_current[-2]
        yhat.append(val)
        y_current.append(val)
        e_current.append(0)
    return np.array(yhat)


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ───────────────────────────────────────────────────────────────
# Data Preparation
# ───────────────────────────────────────────────────────────────
df = pd.read_csv('jena_climate_2009_2016.csv')
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S', dayfirst=True)
df = df.drop_duplicates('Date Time').set_index('Date Time').sort_index()
y = df.loc['2009-01-01':'2011-12-31', 'T (degC)']
y_diff = y.diff().dropna().values

# ───────────────────────────────────────────────────────────────
# Train/Test Split & Horizon
# ───────────────────────────────────────────────────────────────
N = len(y_diff)
train_size = int(0.9 * N)
train_data = y_diff[:train_size]
test_data  = y_diff[train_size:]
horizon    = min(1000, len(test_data))


# ───────────────────────────────────────────────────────────────
# ARMA(2,2)
# ───────────────────────────────────────────────────────────────
theta22 = [0.09956141, -0.08382332, -0.57963991, -0.24923087]  # [φ1, φ2, θ1, θ2]
phi22   = theta22[:2]; ma22 = theta22[2:]
e22p    = compute_error(theta22, train_data, 2, 2)
e22     = np.concatenate([np.zeros(2), e22p])

print("\nARMA(2,2) Q-test on residuals:")
q_test(e22p, lags=50, model_df=4)

one_step_forecast_arma_plot(train_data, e22, phi22, ma22, n_plot=30)
h_preds22 = h_step_forecast_arma_plot(train_data, e22, phi22, ma22, h=horizon, y_actual=test_data[:horizon])

# Metrics on differenced
true22 = test_data[:horizon]; pred22 = np.array(h_preds22)
print(f"\nARMA(2,2) diff-scale metrics (h={horizon}):")
print("MAE:", np.mean(np.abs(true22-pred22)))
print("RMSE:", np.sqrt(np.mean((true22-pred22)**2)))
print("Var:", np.var(true22-pred22))

# ───────────────────────────────────────────────────────────────
# Reverse transformation to original scale
# ───────────────────────────────────────────────────────────────
y_train_last = y.iloc[train_size]  # last observed before forecasts
y12_orig = y_train_last + np.cumsum(h_preds12)
y22_orig = y_train_last + np.cumsum(h_preds22)
y_actual_orig = y.iloc[train_size+1:train_size+1+horizon].values

# Plot original-scale forecasts
plt.figure(figsize=(10,4))
plt.plot(y_actual_orig, 'o-', label='Actual')
plt.plot(y12_orig, 'x--', label='ARMA(1,2)')
plt.plot(y22_orig, 's--', label='ARMA(2,2)')
plt.title('Original-Scale Forecasts vs Actual')
plt.ylabel('Temperature (°C)')
plt.legend(); plt.grid(True); plt.show()

# Metrics on original scale
def safe_mape(a, b):
    mask = a != 0
    return np.mean(np.abs((a[mask]-b[mask]) / a[mask]))*100

print("\nOriginal-scale metrics:")
for mdl, y_pred in zip(['ARMA(1,2)', 'ARMA(2,2)'], [y12_orig, y22_orig]):
    mae = mean_absolute_error(y_actual_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual_orig, y_pred))
    mape = safe_mape(y_actual_orig, y_pred)
    var = np.var(y_actual_orig - y_pred)
    print(f"{mdl}: MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%, Var={var:.3f}")
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import chi2
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ───────────────────────────────────────────────────────────────
# Data Preparation
# ───────────────────────────────────────────────────────────────
df = pd.read_csv('jena_climate_2009_2016.csv')
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S', dayfirst=True)
df = df.set_index('Date Time').sort_index()
y = df.loc['2009-01-01':'2011-12-31', 'T (degC)']

# ───────────────────────────────────────────────────────────────
# Train/Test split & Horizon
# ───────────────────────────────────────────────────────────────
train_size = int(len(y) * 0.8)
train, test = y.iloc[:train_size], y.iloc[train_size:]
horizon = 1000
freq = y.index[1] - y.index[0]

# ───────────────────────────────────────────────────────────────
# ARIMA Models to Test
# ───────────────────────────────────────────────────────────────
models = {
    'ARIMA(1,1,2)': (1, 1, 2),
    'ARIMA(0,1,2)': (0, 1, 2)
}

for name, order in models.items():
    p, d, q = order
    print(f"\n=== {name} ===")

    # 1) Fit model on train
    res = ARIMA(train, order=order).fit()
    print(res.summary())

    # 2) Perform Q-test on residuals
    q_test(res.resid, lags=50, model_df=p+q, alpha=0.05)

    # 3) 1-step forecast
    pred1 = res.get_prediction(start=train_size, end=train_size+len(test)-1, dynamic=False)
    mean1, ci1 = pred1.predicted_mean, pred1.conf_int()
    mean1.index, ci1.index = test.index, test.index

    # 4) Evaluate 1-step
    mae1 = mean_absolute_error(test, mean1)
    rmse1 = np.sqrt(mean_squared_error(test, mean1))
    mape1 = np.mean(np.abs((test - mean1)/test.replace(0, np.nan))) * 100
    var1 = np.var(test - mean1)
    print(f"1-step Metrics — MAE: {mae1:.3f}, RMSE: {rmse1:.3f}, MAPE: {mape1:.2f}%, Var: {var1:.3f}")

    # 5) Plot 1-step forecast
    plt.figure(figsize=(10,4))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, 'k', label='Test')
    plt.plot(mean1.index, mean1, 'r--', label='1-step Forecast')
    plt.fill_between(ci1.index, ci1.iloc[:,0], ci1.iloc[:,1], color='red', alpha=0.3)
    plt.title(f'{name} — 1-step Forecast')
    plt.legend(); plt.show()

    # 6) 1000-step forecast
    fc = res.get_forecast(steps=horizon)
    fc_mean, fc_ci = fc.predicted_mean, fc.conf_int()
    future_idx = pd.date_range(start=y.index[-1]+freq, periods=horizon, freq=freq)

    # 7) Evaluate 1000-step if possible
    if len(test) >= horizon:
        true_h = test[:horizon]
        mae_h = mean_absolute_error(true_h, fc_mean[:horizon])
        rmse_h = np.sqrt(mean_squared_error(true_h, fc_mean[:horizon]))
        mape_h = np.mean(np.abs((true_h - fc_mean[:horizon])/true_h.replace(0, np.nan))) * 100
        var_h = np.var(true_h - fc_mean[:horizon])
        print(f"1000-step Metrics — MAE: {mae_h:.3f}, RMSE: {rmse_h:.3f}, MAPE: {mape_h:.2f}%, Var: {var_h:.3f}")

    # 8) Plot 1000-step forecast
    plt.figure(figsize=(10,4))
    plt.plot(y.index, y, label='Observed')
    plt.plot(future_idx, fc_mean.values, 'g--', label='1000-step Forecast')
    plt.fill_between(future_idx, fc_ci.iloc[:,0], fc_ci.iloc[:,1], color='green', alpha=0.3)
    plt.title(f'{name} — 1000-step Forecast')
    plt.legend(); plt.show()
#%%
import time
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# 1. Load & daily‐resample
df = (
    pd.read_csv("jena_climate_2009_2016.csv",
                parse_dates=["Date Time"], dayfirst=True)
      .drop_duplicates("Date Time")
      .set_index("Date Time")
      .sort_index()
)
y = df["T (degC)"].resample("D").mean()

# 2. Train/test split
train = y["2009-01-01":"2012-12-31"]
test  = y["2013-01-01":"2013-12-31"]

# 3. Specify SARIMA(1,1,1)x(1,1,1)[365] with simple differencing
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 365),
    enforce_stationarity=False,
    enforce_invertibility=False,
    simple_differencing=True,    # <— huge speedup for large m
)

# 4. Fit & time
t0  = time.time()
res = model.fit(disp=False)
print(f"Fitted in {time.time() - t0:.2f}s")
print(res.summary())

# 5. One‐step‐ahead predict over 2013 & time
t1   = time.time()
pred = res.get_prediction(
    start=test.index[0],
    end=test.index[-1],
    dynamic=False
)
print(f"Predicted in {time.time() - t1:.2f}s")

yhat = pred.predicted_mean
ci   = pred.conf_int()

# 6. Plot only the test period, tight y‐limits
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(test.index, test,  label="Actual (2013)",    color="C1")
ax.plot(yhat.index, yhat,  label="One‐step Forecast", color="C2")
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color="C2", alpha=0.2)

ax.set_xlim(test.index[0], test.index[-1])
ymin, ymax = min(test.min(), yhat.min()), max(test.max(), yhat.max())
pad = (ymax - ymin)*0.05
ax.set_ylim(ymin-pad, ymax+pad)

ax.set_title("Daily SARIMA(1,1,1)x(1,1,1)[365]\nOne‐step‐ahead on 2013")
ax.set_ylabel("Temperature (°C)")
ax.legend()
plt.show()

# 7. Compute RMSE
mse  = mean_squared_error(test, yhat)
rmse = mse**0.5
print(f"2013 Test RMSE: {rmse:.3f} °C")

#%%
# 8. h-step ahead forecast for 300 days after the end of training
h = 300
fc = res.get_forecast(steps=h)
fc_mean = fc.predicted_mean
fc_ci   = fc.conf_int()

# 9. Plot the historical train+test and the 300-day forecast
fig, ax = plt.subplots(figsize=(12,5))
# plot the full history (train + test)
y.plot(label="Historical (2009–2013)", ax=ax, color="C0")

# plot the 300-step forecast
fc_mean.plot(label=f"{h}-day Forecast", ax=ax, color="C2")
ax.fill_between(fc_ci.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1], color="C2", alpha=0.2)

# formatting
ax.set_title(f"{h}-Day Ahead SARIMA Forecast")
ax.set_ylabel("Temperature (°C)")
ax.legend()
plt.show()

#%%
# suppose `fc_mean` is your 300-day ahead SARIMA forecast (which is Δy,
# because you used d=1 inside the model)

# 1. Grab the last observed value of the DAILY series
last_obs = train.iloc[-1]

# 2. Invert the difference by cumulative sum + that last level
#    This gives you ŷ on the original scale
y_forecast_orig = last_obs + fc_mean.cumsum()

# 3. Plot against your historical daily series
fig, ax = plt.subplots(figsize=(12,5))
y.plot(label="Historical (2009–2013)", ax=ax, color="C0")
y_forecast_orig.plot(label="300-day Forecast (inverted)", ax=ax, color="C2")
ax.set_title("Original‐Scale 300-Day SARIMA Forecast")
ax.set_ylabel("Temperature (°C)")
ax.legend()
plt.show()
#%%
print(y_forecast_orig)
y=y["2013-05-01":"2013-12-31"]
last_obs = train.iloc[-1]
# 2. Invert the difference by cumulative sum + that last level
#    This gives you ŷ on the original scale
y_forecast_orig = last_obs + fc_mean.cumsum()
y_forecast_orig=y_forecast_orig["2013-05-01":"2013-12-31"]
# 3. Plot against your historical daily series
fig, ax = plt.subplots(figsize=(12,5))
y.plot(label="Historical (2009–2013)", ax=ax, color="C0")
y_forecast_orig.plot(label="300-day Forecast (inverted)", ax=ax, color="C2")
ax.set_title("Original‐Scale 300-Day SARIMA Forecast")
ax.set_ylabel("Temperature (°C)")
ax.legend()
plt.show()
#%%
from scipy.stats import chi2
import numpy as np

# (Re‐define this only if you haven’t already in your session)
def q_test(residuals, lags=50, model_df=0, alpha=0.05):
    residuals = np.asarray(residuals)
    N = len(residuals)
    residuals -= residuals.mean()
    var_e = np.var(residuals)

    Q = 0
    for tau in range(1, lags + 1):
        autocov = np.sum(residuals[tau:] * residuals[:-tau]) / (N - tau)
        r_tau = autocov / var_e
        Q += r_tau ** 2

    Q_stat = N * Q
    dof = lags - model_df
    Q_crit = chi2.ppf(1 - alpha, df=dof)

    print(f"\n--- Q-Test Summary ---")
    print(f"Q-statistic              : {Q_stat:.4f}")
    print(f"Chi-square Critical (α={alpha}, dof={dof}) : {Q_crit:.4f}")
    print("Result                   :",
          "✅ Residuals are white (Q < Q*)" if Q_stat < Q_crit
          else "❌ Residuals show autocorrelation (Q > Q*)")

    return Q_stat, Q_crit, dof

# 1. Extract the residuals from your fitted SARIMA model
residuals = res.resid

# 2. Set the number of parameters you estimated (model degrees of freedom)
#    Here we use the length of the parameter vector
model_df = len(res.params)

# 3. Run the Q-test on the residuals for, say, 50 lags
Q_stat, Q_crit, dof = q_test(residuals, lags=50, model_df=model_df, alpha=0.05)


#%%
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ───────────────────────────────────────────────────────────────
# 1. Load & daily‐resample
# ───────────────────────────────────────────────────────────────
df = (
    pd.read_csv("jena_climate_2009_2016.csv",
                parse_dates=["Date Time"],
                dayfirst=True)
      .drop_duplicates("Date Time")
      .set_index("Date Time")
      .sort_index()
)
y = df["T (degC)"].resample("D").mean()

# ───────────────────────────────────────────────────────────────
# 2. Train/test split
# ───────────────────────────────────────────────────────────────
train = y["2009-01-01":"2012-12-31"]
test  = y["2013-01-01":"2013-12-31"]
h_test = len(test)
freq = "D"

# ───────────────────────────────────────────────────────────────
# 3. Fit SARIMA(1,1,2)x(1,1,1)[365] with simple differencing
# ───────────────────────────────────────────────────────────────
model = SARIMAX(
    train,
    order=(1, 1, 2),
    seasonal_order=(1, 1, 1, 365),
    enforce_stationarity=False,
    enforce_invertibility=False,
    simple_differencing=True
)

t0 = time.time()
res = model.fit(disp=False)
print(f"Fitted in {time.time() - t0:.2f}s")
print(res.summary())

# ───────────────────────────────────────────────────────────────
# 4. One‐step-ahead forecast on 2013
# ───────────────────────────────────────────────────────────────
pred1 = res.get_prediction(start=test.index[0], end=test.index[-1], dynamic=False)
yhat1 = pred1.predicted_mean
ci1   = pred1.conf_int()

# Plot
plt.figure(figsize=(12,5))
plt.plot(test.index, test,    label="Actual (2013)",    color="C1")
plt.plot(yhat1.index, yhat1,  label="1-step Forecast",  color="C2")
plt.fill_between(ci1.index, ci1.iloc[:,0], ci1.iloc[:,1], color="C2", alpha=0.2)
plt.title("One-step-ahead Forecast on 2013")
plt.ylabel("Temperature (°C)")
plt.legend(); plt.show()

# Evaluate
rmse1 = np.sqrt(mean_squared_error(test, yhat1))
mae1  = mean_absolute_error(test, yhat1)
print(f"One-step RMSE: {rmse1:.3f} °C, MAE: {mae1:.3f} °C")

# ───────────────────────────────────────────────────────────────
# 5. H-step dynamic forecast on 2013
# ───────────────────────────────────────────────────────────────
pred_dyn = res.get_prediction(start=test.index[0], end=test.index[-1], dynamic=True)
yhat_dyn = pred_dyn.predicted_mean
ci_dyn   = pred_dyn.conf_int()

plt.figure(figsize=(12,5))
plt.plot(test.index, test,        'k',  label="Actual")
plt.plot(yhat_dyn.index, yhat_dyn, 'r--', label=f"{h_test}-step Dynamic Forecast")
plt.fill_between(ci_dyn.index, ci_dyn.iloc[:,0], ci_dyn.iloc[:,1], color="r", alpha=0.2)
plt.title(f"{h_test}-step Dynamic Forecast on 2013")
plt.ylabel("Temperature (°C)")
plt.legend(); plt.show()

rmse_dyn = np.sqrt(mean_squared_error(test, yhat_dyn))
mae_dyn  = mean_absolute_error(test, yhat_dyn)
print(f"Dynamic RMSE: {rmse_dyn:.3f} °C, MAE: {mae_dyn:.3f} °C")
#%%
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ───────────────────────────────────────────────────────────────
# 1. Load & daily‐resample
# ───────────────────────────────────────────────────────────────
df = (
    pd.read_csv("jena_climate_2009_2016.csv",
                parse_dates=["Date Time"],
                dayfirst=True)
      .drop_duplicates("Date Time")
      .set_index("Date Time")
      .sort_index()
)
y = df["T (degC)"].resample("D").mean()

# ───────────────────────────────────────────────────────────────
# 2. Train/test split
# ───────────────────────────────────────────────────────────────
train = y["2009-01-01":"2012-12-31"]
test  = y["2013-01-01":"2013-12-31"]
h_test = len(test)
freq = "D"

# ───────────────────────────────────────────────────────────────
# 3. Fit SARIMA(1,1,2)x(1,1,1)[365] with simple differencing
# ───────────────────────────────────────────────────────────────
model = SARIMAX(
    train,
    order=(0, 1, 2),
    seasonal_order=(1, 1, 1, 365),
    enforce_stationarity=False,
    enforce_invertibility=False,
    simple_differencing=True
)

t0 = time.time()
res = model.fit(disp=False)
print(f"Fitted in {time.time() - t0:.2f}s")
print(res.summary())

# ───────────────────────────────────────────────────────────────
# 4. One‐step-ahead forecast on 2013
# ───────────────────────────────────────────────────────────────
pred1 = res.get_prediction(start=test.index[0], end=test.index[-1], dynamic=False)
yhat1 = pred1.predicted_mean
ci1   = pred1.conf_int()

# Plot
plt.figure(figsize=(12,5))
plt.plot(test.index, test,    label="Actual (2013)",    color="C1")
plt.plot(yhat1.index, yhat1,  label="1-step Forecast",  color="C2")
plt.fill_between(ci1.index, ci1.iloc[:,0], ci1.iloc[:,1], color="C2", alpha=0.2)
plt.title("One-step-ahead Forecast on 2013")
plt.ylabel("Temperature (°C)")
plt.legend(); plt.show()

# Evaluate
rmse1 = np.sqrt(mean_squared_error(test, yhat1))
mae1  = mean_absolute_error(test, yhat1)
print(f"One-step RMSE: {rmse1:.3f} °C, MAE: {mae1:.3f} °C")

# ───────────────────────────────────────────────────────────────
# 5. H-step dynamic forecast on 2013
# ───────────────────────────────────────────────────────────────
pred_dyn = res.get_prediction(start=test.index[0], end=test.index[-1], dynamic=True)
yhat_dyn = pred_dyn.predicted_mean
ci_dyn   = pred_dyn.conf_int()

plt.figure(figsize=(12,5))
plt.plot(test.index, test,        'k',  label="Actual")
plt.plot(yhat_dyn.index, yhat_dyn, 'r--', label=f"{h_test}-step Dynamic Forecast")
plt.fill_between(ci_dyn.index, ci_dyn.iloc[:,0], ci_dyn.iloc[:,1], color="r", alpha=0.2)
plt.title(f"{h_test}-step Dynamic Forecast on 2013")
plt.ylabel("Temperature (°C)")
plt.legend(); plt.show()

rmse_dyn = np.sqrt(mean_squared_error(test, yhat_dyn))
mae_dyn  = mean_absolute_error(test, yhat_dyn)
print(f"Dynamic RMSE: {rmse_dyn:.3f} °C, MAE: {mae_dyn:.3f} °C")
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import chi2

# ───────────────────────────────────────────────────────────────
# Q-Test function
# ───────────────────────────────────────────────────────────────
def q_test(residuals, lags=50, model_df=0, alpha=0.05):
    r = residuals - np.mean(residuals)
    N = len(r)
    var_e = np.var(r)
    Q = sum((np.sum(r[t:] * r[:-t]) / ((N - t) * var_e))**2 for t in range(1, lags+1))
    Q_stat = N * Q
    dof = lags - model_df
    Q_crit = chi2.ppf(1 - alpha, df=dof)
    print(f"\n--- Q-Test (lags={lags}, df={dof}, alpha={alpha}) ---")
    print(f"Q-statistic: {Q_stat:.4f}")
    print(f"Critical value: {Q_crit:.4f}")
    print("Result:", "✅ Residuals are white noise" if Q_stat < Q_crit else "❌ Residuals show autocorrelation")
    return Q_stat, Q_crit, dof

# ───────────────────────────────────────────────────────────────
# 1. Load & daily‐resample
# ───────────────────────────────────────────────────────────────
df = (
    pd.read_csv("jena_climate_2009_2016.csv", parse_dates=["Date Time"], dayfirst=True)
      .drop_duplicates("Date Time")
      .set_index("Date Time")
      .sort_index()
)
y = df["T (degC)"].resample("D").mean().interpolate(method="time")

# ───────────────────────────────────────────────────────────────
# 2. Seasonal decomposition (period=365)
# ───────────────────────────────────────────────────────────────
decomp = seasonal_decompose(y, model="additive", period=365)
seasonal_full = decomp.seasonal

# ───────────────────────────────────────────────────────────────
# 3. Seasonally adjust
# ───────────────────────────────────────────────────────────────
y_adj = y - seasonal_full

# ───────────────────────────────────────────────────────────────
# 4. Train/Test split on adjusted data
# ───────────────────────────────────────────────────────────────
train_adj = y_adj["2009-01-01":"2012-12-31"]
test_adj  = y_adj["2013-01-01":"2013-12-31"]

# ───────────────────────────────────────────────────────────────
# 5. Fit ARIMA(0,1,2)
# ───────────────────────────────────────────────────────────────
model = ARIMA(train_adj, order=(0,1,2))
res   = model.fit()
print(res.summary())

# ───────────────────────────────────────────────────────────────
# 6. Q-test on in-sample residuals
# ───────────────────────────────────────────────────────────────
# model_df = p + q = 0 + 2 = 2
q_test(res.resid, lags=50, model_df=2)

# ───────────────────────────────────────────────────────────────
# 7. One-step ahead forecast of adjusted series
# ───────────────────────────────────────────────────────────────
pred_adj = res.get_prediction(start=test_adj.index[0],
                              end=test_adj.index[-1],
                              dynamic=False)
yhat_adj = pred_adj.predicted_mean
ci_adj   = pred_adj.conf_int()

# ───────────────────────────────────────────────────────────────
# 8. Reconstruct original‐scale forecast
# ───────────────────────────────────────────────────────────────
seasonal_2013 = seasonal_full.loc[test_adj.index]
yhat_full    = yhat_adj + seasonal_2013

# ───────────────────────────────────────────────────────────────
# 9. Plot Actual vs Forecast
# ───────────────────────────────────────────────────────────────
plt.figure(figsize=(12,5))
plt.plot(y.loc[test_adj.index], label="Actual (2013)", color="C1")
plt.plot(yhat_full.index,    yhat_full,    "--", label="Forecast", color="C2")
plt.fill_between(ci_adj.index,
                 (ci_adj.iloc[:,0] + seasonal_2013).values,
                 (ci_adj.iloc[:,1] + seasonal_2013).values,
                 color="C2", alpha=0.2)
plt.title("Forecast vs Actual (2013)")
plt.ylabel("Temperature (°C)")
plt.legend(); plt.grid(True); plt.show()

# ───────────────────────────────────────────────────────────────
#10. Compute original-scale metrics
# ───────────────────────────────────────────────────────────────
actual = y.loc[test_adj.index]
rmse   = np.sqrt(mean_squared_error(actual, yhat_full))
mae    = mean_absolute_error(actual, yhat_full)
print(f"\nForecast metrics: MAE = {mae:.3f} °C, RMSE = {rmse:.3f} °C")

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import chi2

# ───────────────────────────────────────────────────────────────
# Q-Test function
# ───────────────────────────────────────────────────────────────
def q_test(residuals, lags=50, model_df=0, alpha=0.05):
    r = residuals - np.mean(residuals)
    N = len(r)
    var_e = np.var(r)
    Q = sum((np.sum(r[t:] * r[:-t]) / ((N - t) * var_e))**2 for t in range(1, lags+1))
    Q_stat = N * Q
    dof = lags - model_df
    Q_crit = chi2.ppf(1 - alpha, df=dof)
    print(f"\n--- Q-Test (lags={lags}, df={dof}, alpha={alpha}) ---")
    print(f"Q-statistic: {Q_stat:.4f}")
    print(f"Critical value: {Q_crit:.4f}")
    print("Result:", "✅ Residuals are white noise" if Q_stat < Q_crit else "❌ Residuals show autocorrelation")
    return Q_stat, Q_crit, dof

# ───────────────────────────────────────────────────────────────
# 1. Load & daily‐resample
# ───────────────────────────────────────────────────────────────
df = (
    pd.read_csv("jena_climate_2009_2016.csv", parse_dates=["Date Time"], dayfirst=True)
      .drop_duplicates("Date Time")
      .set_index("Date Time")
      .sort_index()
)
y = df["T (degC)"].resample("D").mean().interpolate(method="time")

# ───────────────────────────────────────────────────────────────
# 2. Seasonal decomposition (period=365)
# ───────────────────────────────────────────────────────────────
decomp = seasonal_decompose(y, model="additive", period=365)
seasonal_full = decomp.seasonal

# ───────────────────────────────────────────────────────────────
# 3. Seasonally adjust
# ───────────────────────────────────────────────────────────────
y_adj = y - seasonal_full

# ───────────────────────────────────────────────────────────────
# 4. Train/Test split on adjusted data
# ───────────────────────────────────────────────────────────────
train_adj = y_adj["2009-01-01":"2012-12-31"]
test_adj  = y_adj["2013-01-01":"2013-12-31"]

# ───────────────────────────────────────────────────────────────
# 5. Fit ARIMA(1,1,2) on seasonally‐adjusted train
# ───────────────────────────────────────────────────────────────
model = ARIMA(train_adj, order=(1,1,2))
res   = model.fit()
print(res.summary())

# ───────────────────────────────────────────────────────────────
# 6. Q-test on in-sample ARIMA residuals
#    model_df = p + q = 1 + 2 = 3
# ───────────────────────────────────────────────────────────────
q_test(res.resid, lags=50, model_df=3)

# ───────────────────────────────────────────────────────────────
# 7. One-step ahead forecast of adjusted series
# ───────────────────────────────────────────────────────────────
pred_adj = res.get_prediction(
    start=test_adj.index[0],
    end=test_adj.index[-1],
    dynamic=False
)
yhat_adj = pred_adj.predicted_mean
ci_adj   = pred_adj.conf_int()

# ───────────────────────────────────────────────────────────────
# 8. Reconstruct original‐scale forecast by adding back seasonal
# ───────────────────────────────────────────────────────────────
seasonal_2013 = seasonal_full.loc[test_adj.index]
yhat_full     = yhat_adj + seasonal_2013

# ───────────────────────────────────────────────────────────────
# 9. Plot Actual vs Forecast
# ───────────────────────────────────────────────────────────────
plt.figure(figsize=(12,5))
plt.plot(y.loc[test_adj.index], label="Actual (2013)", color="C1")
plt.plot(yhat_full.index,    yhat_full,    "--", label="Forecast", color="C2")
plt.fill_between(ci_adj.index,
                 (ci_adj.iloc[:,0] + seasonal_2013).values,
                 (ci_adj.iloc[:,1] + seasonal_2013).values,
                 color="C2", alpha=0.2)
plt.title("ARIMA(1,1,2) on Seasonally‐Adjusted Data\n+ Seasonal Recomposition")
plt.ylabel("Temperature (°C)")
plt.legend(); plt.grid(True); plt.show()

# ───────────────────────────────────────────────────────────────
#10. Compute original-scale metrics
# ───────────────────────────────────────────────────────────────
actual = y.loc[test_adj.index]
rmse   = np.sqrt(mean_squared_error(actual, yhat_full))
mae    = mean_absolute_error(actual, yhat_full)
print(f"\nForecast metrics: MAE = {mae:.3f} °C, RMSE = {rmse:.3f} °C")

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from numpy.linalg import pinv, LinAlgError
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ───────────────────────────────────────────────────────────────
# 1) Grey‐box functions (ARXMA + LM + Q‐test + S‐test)
# ───────────────────────────────────────────────────────────────
def compute_error_bj(theta, y, u, nb, nf, nc, nd):
    N = len(y)
    b = theta[:nb]
    f = np.r_[1, theta[nb:nb+nf]]
    c = np.r_[1, theta[nb+nf:nb+nf+nc]]
    d = np.r_[1, -theta[nb+nf+nc:nb+nf+nc+nd]]
    y_g = np.zeros(N); e = np.zeros(N)
    lag1 = max(nb, len(f))
    for t in range(lag1, N):
        arx = sum(b[i] * u[t-i] for i in range(nb))
        fb  = sum(f[j] * y_g[t-j] for j in range(1, len(f)))
        y_g[t] = arx - fb
    res  = y - y_g
    lag2 = max(len(c), len(d))
    for t in range(lag2, N):
        num = sum(d[j] * res[t-j] for j in range(len(d)))
        den = sum(c[i] * e[t-i] for i in range(1, len(c)))
        e[t] = num - den
    return e[lag2:]

def compute_jacobian_bj(theta, y, u, nb, nf, nc, nd, delta=1e-6):
    e0 = compute_error_bj(theta, y, u, nb, nf, nc, nd)
    m  = len(theta)
    J  = np.zeros((len(e0), m))
    for i in range(m):
        tp     = theta.copy(); tp[i] += delta
        ei     = compute_error_bj(tp, y, u, nb, nf, nc, nd)
        J[:,i] = (ei - e0) / delta
    return J

def levenberg_marquardt_bj(y, u, theta0, nb, nf, nc, nd,
                           mu=1e-2, maxiter=50, tol=1e-6):
    θ = theta0.copy()
    for it in range(maxiter):
        e   = compute_error_bj(θ, y, u, nb, nf, nc, nd)
        SSE = e @ e
        J   = compute_jacobian_bj(θ, y, u, nb, nf, nc, nd)
        A   = J.T @ J; g = J.T @ e
        try:
            Δθ = np.linalg.solve(A + mu*np.eye(len(θ)), g)
        except LinAlgError:
            Δθ = pinv(A + mu*np.eye(len(θ))) @ g
        θn = θ + Δθ
        en = compute_error_bj(θn, y, u, nb, nf, nc, nd)
        if en @ en < SSE:
            θ = θn; mu *= 0.1
            if np.linalg.norm(Δθ) < tol: break
        else:
            mu *= 10
    return θ

def q_test(resid, lags=50, model_df=0, alpha=0.05):
    r    = resid - resid.mean()
    N    = len(r)
    var  = r.var()
    Q    = sum(((r[t:] @ r[:-t]) / ((N-t)*var))**2 for t in range(1, lags+1))
    Qs   = N * Q; dof = lags - model_df; crit = chi2.ppf(1-alpha, dof)
    print(f"Q-test: Q={Qs:.1f}, crit={crit:.1f}, df={dof} ->",
          "✅ white noise" if Qs < crit else "❌ autocorrelation")
    return Qs, crit, dof

def s_test(e, u, theta, nb, nf, K=20, alpha=0.05):
    N = len(e)
    e_n = (e - e.mean()) / np.std(e)
    f   = np.r_[1, theta[nb:nb+nf]]
    alpha_t = np.zeros(N)
    for t in range(len(f), N):
        alpha_t[t] = u[t] - sum(f[j] * alpha_t[t-j] for j in range(1, len(f)))
    alpha_n = (alpha_t - alpha_t.mean()) / np.std(alpha_t)
    S = sum((np.sum(alpha_n[:N-t] * e_n[t:])/(N-t))**2 for t in range(K+1))
    S_stat = N * S; dof = K - nf; crit = chi2.ppf(1-alpha, dof)
    print(f"S-test: S={S_stat:.1f}, crit={crit:.1f}, df={dof} ->",
          "✅ G(q) accurate" if S_stat < crit else "❌ G(q) misspecified")
    return S_stat, crit, dof

# ───────────────────────────────────────────────────────────────
# 2) Load & daily‐resample data
# ───────────────────────────────────────────────────────────────
df = (pd.read_csv("jena_climate_2009_2016.csv",
                  parse_dates=["Date Time"], dayfirst=True)
      .drop_duplicates("Date Time")
      .set_index("Date Time")
      .sort_index())
y_full = df["T (degC)"].resample("D").mean().interpolate("time")
u_full = df["Tpot (K)"].resample("D").mean().interpolate("time")

# ───────────────────────────────────────────────────────────────
# 3) Split: Train=2009–2010, Test=2011, Next=2012
# ───────────────────────────────────────────────────────────────
train_raw = y_full["2009-01-01":"2010-12-31"]
test_raw  = y_full["2011-01-01":"2011-12-31"]
next_raw  = y_full["2012-01-01":"2012-12-31"]

# ───────────────────────────────────────────────────────────────
# 4) First‐difference
# ───────────────────────────────────────────────────────────────
y_diff_full = y_full.diff().dropna()
u_diff_full = u_full.diff().dropna()
y_train = y_diff_full[train_raw.index[1:]].values
u_train = u_diff_full[train_raw.index[1:]].values
y_test  = y_diff_full[test_raw.index[1:]].values
u_test  = u_diff_full[test_raw.index[1:]].values
y_next  = y_diff_full[next_raw.index[1:]].values
u_next  = u_diff_full[next_raw.index[1:]].values

# ───────────────────────────────────────────────────────────────
# 5) Seed b0,b1 via OLS on train diff
# ───────────────────────────────────────────────────────────────
X     = np.vstack([u_train[1:], u_train[:-1]]).T
b0,b1 = np.linalg.lstsq(X, y_train[1:], rcond=None)[0]
print("OLS seed b0,b1:", b0, b1)

# ───────────────────────────────────────────────────────────────
# 6) Set orders & init θ
# ───────────────────────────────────────────────────────────────
nb,nf,nc,nd = 2,1,0,0  # ARXMA(1,1)
θ0          = np.zeros(nb+nf+nc+nd)
θ0[0],θ0[1] = b0,b1

# ───────────────────────────────────────────────────────────────
# 7) Fit on train diff
# ───────────────────────────────────────────────────────────────
θ_est       = levenberg_marquardt_bj(y_train, u_train, θ0, nb, nf, nc, nd)
print("Estimated θ:", np.round(θ_est,4))

# ───────────────────────────────────────────────────────────────
# 8) In-sample Q-test & S-test
# ───────────────────────────────────────────────────────────────
resid_train = compute_error_bj(θ_est, y_train, u_train, nb, nf, nc, nd)
q_test(resid_train, model_df=nb+nf)
s_test(resid_train, u_train, θ_est, nb, nf)

# ───────────────────────────────────────────────────────────────
# 9) One-step ahead forecast (2011)
# ───────────────────────────────────────────────────────────────
b     = θ_est[:nb]
f     = np.r_[1, θ_est[nb:nb+nf]]
y_diff = y_diff_full  # for indexing

pred_diff = pd.Series(index=test_raw.index[1:], dtype=float)
for t in pred_diff.index:
    arx = sum(b[i]*u_diff_full.loc[t - pd.Timedelta(days=i)] for i in range(nb))
    fb  = sum(f[j]*y_diff.loc[t - pd.Timedelta(days=j)] for j in range(1,len(f)))
    pred_diff.loc[t] = arx - fb

# invert diff to °C
one_step = test_raw.shift(1).loc[pred_diff.index] + pred_diff

# clean & metrics
df1 = pd.concat([
    test_raw.shift(1).loc[pred_diff.index].rename("actual_prev"),
    pred_diff.rename("diff_fc")
], axis=1).dropna()
df1["one_step"] = df1["actual_prev"] + df1["diff_fc"]

actual_1 = test_raw.loc[df1.index]
fc_1     = df1["one_step"]
e1       = actual_1 - fc_1

mae1   = mean_absolute_error(actual_1, fc_1)
rmse1  = np.sqrt(mean_squared_error(actual_1, fc_1))
mape1  = (e1.abs()/actual_1).mean()*100
var1   = e1.var()
corr1  = actual_1.corr(fc_1)

print("\n1-Step Metrics:")
print(f" MAE   = {mae1:.3f}")
print(f" RMSE  = {rmse1:.3f}")
print(f" MAPE  = {mape1:.2f}%")
print(f" Var   = {var1:.4f}")
print(f" Corr² = {corr1**2:.3f}")

# ───────────────────────────────────────────────────────────────
# 10) H-step ahead forecast (365 for 2012)
# ───────────────────────────────────────────────────────────────
u_all = np.concatenate([u_train, u_test, u_next])
N_all = len(u_all)
yg_all = np.zeros(N_all)
for t in range(max(nb,nf), N_all):
    yg_all[t] = sum(b[i]*u_all[t-i] for i in range(nb)) - sum(f[j]*yg_all[t-j] for j in range(1,len(f)))

yg_next = yg_all[len(u_train)+len(u_test):]  # 365 diffs
start_val = test_raw.iloc[-1]
h_vals = [start_val]
for d in yg_next:
    h_vals.append(h_vals[-1] + d)

h_step = pd.Series(h_vals[1:], index=next_raw.index[1:])

# clean & metrics for h-step
dfh = pd.concat([
    next_raw.shift(1).loc[h_step.index].rename("actual_prev"),
    pd.Series(yg_next, index=h_step.index).rename("diff_fc")
], axis=1).dropna()
dfh["h_step"] = dfh["actual_prev"] + dfh["diff_fc"]

actual_h = next_raw.loc[dfh.index]
fc_h     = dfh["h_step"]
eh       = actual_h - fc_h

maeh   = mean_absolute_error(actual_h, fc_h)
rmseh  = np.sqrt(mean_squared_error(actual_h, fc_h))
mapeh  = (eh.abs()/actual_h).mean()*100
varh   = eh.var()
corrh  = actual_h.corr(fc_h)

print("\n365-Step Metrics:")
print(f" MAE   = {maeh:.3f}")
print(f" RMSE  = {rmseh:.3f}")
print(f" MAPE  = {mapeh:.2f}%")
print(f" Var   = {varh:.4f}")
print(f" Corr² = {corrh**2:.3f}")

# ───────────────────────────────────────────────────────────────
# 11) Plot both forecasts
# ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(actual_1.index, actual_1, label="Actual 2011", color="C1")
axes[0].plot(fc_1.index, fc_1, 'k--', label="1-step Forecast")
axes[0].set_title("1-Step Ahead Forecast (2011)")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(actual_h.index, actual_h, label="Actual 2012", color="C1")
axes[1].plot(fc_h.index, fc_h, 'k--', label="365-step Forecast")
axes[1].set_title("365-Step Ahead Forecast (2012)")
axes[1].legend(); axes[1].grid(True)

plt.xlabel("Date")
plt.tight_layout()
plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
from numpy.linalg import pinv, LinAlgError
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ───────────────────────────────────────────────────────────────
# Core functions: ARXMA error, Jac, LM, Q-test, S-test, GPAC
# ───────────────────────────────────────────────────────────────
def compute_error_bj(theta, y, u, nb, nf, nc, nd):
    N = len(y)
    b = theta[:nb]
    f = np.r_[1, theta[nb:nb+nf]]
    c = np.r_[1, theta[nb+nf:nb+nf+nc]]
    d = np.r_[1, -theta[nb+nf+nc:nb+nf+nc+nd]]
    y_g = np.zeros(N); e = np.zeros(N)
    lag1 = max(nb, len(f))
    for t in range(lag1, N):
        y_g[t] = sum(b[i]*u[t-i] for i in range(nb)) - sum(f[j]*y_g[t-j] for j in range(1,len(f)))
    res = y - y_g
    lag2 = max(len(c), len(d))
    for t in range(lag2, N):
        num = sum(d[j]*res[t-j] for j in range(len(d)))
        den = sum(c[i]*e[t-i] for i in range(1,len(c)))
        e[t] = num - den
    return e[lag2:]

def compute_jacobian_bj(theta, y, u, nb, nf, nc, nd, delta=1e-6):
    e0 = compute_error_bj(theta, y, u, nb, nf, nc, nd)
    m  = len(theta)
    J  = np.zeros((len(e0), m))
    for i in range(m):
        tp = theta.copy(); tp[i] += delta
        ei = compute_error_bj(tp, y, u, nb, nf, nc, nd)
        J[:,i] = (ei - e0)/delta
    return J

def levenberg_marquardt_bj(y, u, theta0, nb, nf, nc, nd,
                           mu=1e-2, maxiter=50, tol=1e-6):
    θ = theta0.copy()
    for _ in range(maxiter):
        e   = compute_error_bj(θ, y, u, nb, nf, nc, nd)
        SSE = e@e
        J   = compute_jacobian_bj(θ, y, u, nb, nf, nc, nd)
        A   = J.T@J; g = J.T@e
        try:
            Δθ = np.linalg.solve(A + mu*np.eye(len(θ)), g)
        except LinAlgError:
            Δθ = pinv(A + mu*np.eye(len(θ)))@g
        θn = θ + Δθ
        en = compute_error_bj(θn, y, u, nb, nf, nc, nd)
        if en@en < SSE:
            θ, mu = θn, mu*0.1
            if np.linalg.norm(Δθ) < tol: break
        else:
            mu *= 10
    return θ

def q_test(resid, model_df, lags=50, alpha=0.05):
    r    = resid - resid.mean()
    N    = len(r); var = r.var()
    Q    = sum(((r[t:]@r[:-t]) / ((N-t)*var))**2 for t in range(1,lags+1))
    Qs   = N*Q; crit = chi2.ppf(1-alpha, lags-model_df)
    print(f"Q-test: Q={Qs:.2f}, crit={crit:.2f} ->", "PASS" if Qs<crit else "FAIL")
    return Qs, crit

def s_test(e, u, theta, nb, nf, K=20, alpha=0.05):
    N = len(e)
    e_n = (e-e.mean())/np.std(e)
    f   = np.r_[1, theta[nb:nb+nf]]
    alpha_t = np.zeros(N)
    for t in range(len(f), N):
        alpha_t[t] = u[t] - sum(f[j]*alpha_t[t-j] for j in range(1,len(f)))
    a_n = (alpha_t-alpha_t.mean())/np.std(alpha_t)
    S = sum((np.sum(a_n[:N-t]*e_n[t:])/(N-t))**2 for t in range(K+1))
    Ss   = N*S; crit = chi2.ppf(1-alpha, K-nf)
    print(f"S-test: S={Ss:.2f}, crit={crit:.2f} ->", "PASS" if Ss<crit else "FAIL")
    return Ss, crit

def gpac_table_bj(acf_values, max_k=7, max_j=7):
    gpac = np.zeros((max_j, max_k))
    for j in range(max_j):
        for k in range(1, max_k+1):
            D = np.array([[acf_values[abs(j+i-m)] for m in range(k)] for i in range(k)])
            N = D.copy()
            for i in range(k):
                N[i, -1] = acf_values[j+i+1] if j+i+1 < len(acf_values) else 0
            detD = np.linalg.det(D); detN = np.linalg.det(N)
            gpac[j, k-1] = detN/detD if abs(detD)>1e-8 else np.nan
    return pd.DataFrame(gpac, index=[f"j={j}" for j in range(max_j)],
                        columns=[f"k={k}" for k in range(1, max_k+1)])

# ───────────────────────────────────────────────────────────────
# 1) Data prep: daily, diff
# ───────────────────────────────────────────────────────────────
df = pd.read_csv("jena_climate_2009_2016.csv", parse_dates=["Date Time"], dayfirst=True)\
       .drop_duplicates("Date Time")\
       .set_index("Date Time")\
       .sort_index()
y = df["T (degC)"].resample("D").mean().interpolate("time")
train = y["2009-01-01":"2010-12-31"]
y_diff = y.diff().dropna()
y_train = y_diff[train.index[1:]].values

# 2) G-GPAC
ry = acf(y_train, nlags=20, fft=True)
gpac_G = gpac_table_bj(ry, max_k=7, max_j=7)
print("G-GPAC for y_train:")
display(gpac_G)

# 3) LM estimation
u = df["Tpot (K)"].resample("D").mean().interpolate("time")
u_diff = u.diff().dropna()
u_train = u_diff[train.index[1:]].values

nb,nf,nc,nd = 2,1,0,1
# seed OLS
X = np.column_stack([u_train[i:len(u_train)-(nb-1-i)] for i in range(nb)])
y_t = y_train[nb-1:]
b_init = np.linalg.lstsq(X, y_t, rcond=None)[0]
theta0 = np.zeros(nb+nf+nc+nd); theta0[:nb] = b_init

theta_est = levenberg_marquardt_bj(y_train, u_train, theta0, nb,nf,nc,nd)
resid = compute_error_bj(theta_est, y_train, u_train, nb,nf,nc,nd)

# 4) Parameter stats
sigma2 = np.var(resid, ddof=1)
J = compute_jacobian_bj(theta_est, y_train, u_train, nb,nf,nc,nd)
cov_theta = sigma2*pinv(J.T@J)
se = np.sqrt(np.diag(cov_theta))
ci = np.vstack([theta_est-1.96*se, theta_est+1.96*se]).T

param_df = pd.DataFrame({
    "estimate": theta_est,
    "std_error": se,
    "CI_lower": ci[:,0],
    "CI_upper": ci[:,1]
})
print("\nParameter estimates:")
display(param_df)

# 5) H-GPAC on residuals
rh = acf(resid, nlags=20, fft=True)
gpac_H = gpac_table_bj(rh, max_k=7, max_j=7)
print("H-GPAC for residuals:")
display(gpac_H)

# 6) Q-test and S-test
print()
q_test(resid, model_df=nb+nf+nc+nd)
s_test(resid, u_train, theta_est, nb, nf)

# 7) Residual–input corr
corr_ri = np.corrcoef(resid, u_train[:len(resid)])[0,1]
print(f"\nResidual–Input corr: {corr_ri:.3f}")
# 12) Parameter variance and confidence intervals
J_final = compute_jacobian_bj(θ_est, y_train, u_train, nb, nf, nc, nd)
resid_final = compute_error_bj(θ_est, y_train, u_train, nb, nf, nc, nd)
sigma2 = np.var(resid_final)
cov_theta = sigma2 * np.linalg.pinv(J_final.T @ J_final)
std_theta = np.sqrt(np.diag(cov_theta))
ci_lower = θ_est - 1.96 * std_theta
ci_upper = θ_est + 1.96 * std_theta

# Display
print("\nParameter Estimates with 95% Confidence Intervals:")
for i, (val, std, lo, hi) in enumerate(zip(θ_est, std_theta, ci_lower, ci_upper)):
    print(f"θ[{i+1}] = {val:.4f} ± {1.96*std:.4f}   CI = [{lo:.4f}, {hi:.4f}]")
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from numpy.linalg import pinv, LinAlgError

# ───────────────────────────────────────────────────────────────
# Grey‐box core functions (ARXMA + LM + Q‐test + S‐test)
# ───────────────────────────────────────────────────────────────
def compute_error_bj(theta, y, u, nb, nf, nc, nd):
    N = len(y)
    b = theta[:nb]
    f = np.r_[1, theta[nb:nb+nf]]
    c = np.r_[1, theta[nb+nf:nb+nf+nc]]
    d = np.r_[1, -theta[nb+nf+nc:nb+nf+nc+nd]]
    y_g = np.zeros(N); e = np.zeros(N)
    lag1 = max(nb, len(f))
    for t in range(lag1, N):
        y_g[t] = sum(b[i] * u[t-i] for i in range(nb)) - sum(f[j] * y_g[t-j] for j in range(1, len(f)))
    res = y - y_g
    lag2 = max(len(c), len(d))
    for t in range(lag2, N):
        num = sum(d[j] * res[t-j] for j in range(len(d)))
        den = sum(c[i] * e[t-i] for i in range(1, len(c)))
        e[t] = num - den
    return e[lag2:]

def levenberg_marquardt_bj(y, u, theta0, nb, nf, nc, nd,
                           mu=1e-2, maxiter=50, tol=1e-6):
    θ = theta0.copy()
    for _ in range(maxiter):
        e   = compute_error_bj(θ, y, u, nb, nf, nc, nd)
        SSE = e @ e
        # numeric Jacobian
        J = np.zeros((len(e), len(θ)))
        for i in range(len(θ)):
            tp       = θ.copy(); tp[i] += 1e-6
            ei       = compute_error_bj(tp, y, u, nb, nf, nc, nd)
            J[:,i]   = (ei - e) / 1e-6
        A = J.T @ J; g = J.T @ e
        try:
            Δθ = np.linalg.solve(A + mu*np.eye(len(θ)), g)
        except LinAlgError:
            Δθ = pinv(A + mu*np.eye(len(θ))) @ g
        θn = θ + Δθ
        en = compute_error_bj(θn, y, u, nb, nf, nc, nd)
        if en @ en < SSE:
            θ, mu = θn, mu*0.1
            if np.linalg.norm(Δθ) < tol:
                break
        else:
            mu *= 10
    return θ

def q_test(resid, lags=50, model_df=0, alpha=0.05):
    r = resid - resid.mean(); N = len(r); var = r.var()
    Q = sum(((r[t:] @ r[:-t]) / ((N-t)*var))**2 for t in range(1, lags+1))
    Qs = N * Q; dof = lags - model_df; crit = chi2.ppf(1-alpha, dof)
    return Qs, crit, Qs < crit

def s_test(e, u, theta, nb, nf, K=20, alpha=0.05):
    N = len(e)
    e_n = (e - e.mean()) / np.std(e)
    f   = np.r_[1, theta[nb:nb+nf]]
    alpha_t = np.zeros(N)
    for t in range(len(f), N):
        alpha_t[t] = u[t] - sum(f[j] * alpha_t[t-j] for j in range(1, len(f)))
    a_n = (alpha_t - alpha_t.mean()) / np.std(alpha_t)
    S = sum((np.sum(a_n[:N-t] * e_n[t:])/(N-t))**2 for t in range(K+1))
    Ss = N * S; dof = K - nf; crit = chi2.ppf(1-alpha, dof)
    return Ss, crit, Ss < crit

# ───────────────────────────────────────────────────────────────
# Load & daily-resample
# ───────────────────────────────────────────────────────────────
df = pd.read_csv("jena_climate_2009_2016.csv",
                 parse_dates=["Date Time"], dayfirst=True)\
       .drop_duplicates("Date Time")\
       .set_index("Date Time")\
       .sort_index()
y = df["T (degC)"].resample("D").mean().interpolate("time")
u = df["Tpot (K)"].resample("D").mean().interpolate("time")

# Split 2009-2012 into train/test/next
train = y["2009-01-01":"2010-12-31"]
test  = y["2011-01-01":"2011-12-31"]
next_ = y["2012-01-01":"2012-12-31"]

# First-difference
y_diff = y.diff().dropna()
u_diff = u.diff().dropna()
y_train = y_diff[train.index[1:]].values
u_train = u_diff[train.index[1:]].values

# for full simulation
u_all = np.concatenate([
    u_train,
    u_diff[test.index[1:]].values,
    u_diff[next_.index[1:]].values
])
N_all = len(u_all)

# Grid search orders
results = []
# Use nb, nf from 1 to 2 and try noise nc,nd from 0,1
for nb in [1,2]:
    for nf in [1,2]:
        # seed OLS
        if nb >= 2:
            X = np.vstack([u_train[1:], u_train[:-1]]).T
            b0,b1 = np.linalg.lstsq(X, y_train[1:], rcond=None)[0]
            theta0 = np.zeros(nb+nf+2)  # +2 noise if needed
            theta0[0], theta0[1] = b0,b1
        else:
            theta0 = np.zeros(nb+nf+2)
            theta0[0] = np.linalg.lstsq(u_train.reshape(-1,1), y_train, rcond=None)[0]
        for nc in [0,1]:
            for nd in [0,1]:
                p = nb+nf+nc+nd
                θ0 = theta0[:p]
                θ_est = levenberg_marquardt_bj(y_train, u_train, θ0, nb, nf, nc, nd)
                resid = compute_error_bj(θ_est, y_train, u_train, nb, nf, nc, nd)
                Qs, Qc, Qok = q_test(resid, model_df=p)
                Ss, Sc, Sok = s_test(resid, u_train, θ_est, nb, nf)
                results.append({
                    "nb":nb, "nf":nf, "nc":nc, "nd":nd,
                    "Q_stat":Qs, "Q_crit":Qc, "Q_pass":Qok,
                    "S_stat":Ss, "S_crit":Sc, "S_pass":Sok
                })

df_res = pd.DataFrame(results)
# Sort: show all, with flags
print(df_res.sort_values(["Q_pass","S_pass"], ascending=False))
#%%

import pandas as pd
import numpy as np
from scipy.stats import chi2
from numpy.linalg import pinv, LinAlgError
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1) Core routines: compute_error_bj, LM estimator, Q-test, etc.
def compute_error_bj(theta, y, u, nb, nf, nc, nd):
    N = len(y)
    b = theta[:nb]
    f = np.r_[1, theta[nb:nb+nf]]
    c = np.r_[1, theta[nb+nf:nb+nf+nc]]
    d = np.r_[1, -theta[nb+nf+nc:nb+nf+nc+nd]]
    y_g = np.zeros(N); e = np.zeros(N)
    lag1 = max(nb, len(f))
    for t in range(lag1, N):
        y_g[t] = sum(b[i]*u[t-i] for i in range(nb)) \
               - sum(f[j]*y_g[t-j] for j in range(1,len(f)))
    res = y - y_g
    lag2 = max(len(c), len(d))
    for t in range(lag2, N):
        num = sum(d[j]*res[t-j] for j in range(len(d)))
        den = sum(c[i]*e[t-i] for i in range(1,len(c)))
        e[t] = num - den
    return e[lag2:]

def levenberg_marquardt_bj(y, u, theta0, nb, nf, nc, nd,
                           mu=1e-2, maxiter=100, tol=1e-6):
    θ = theta0.copy()
    for _ in range(maxiter):
        e   = compute_error_bj(θ, y, u, nb, nf, nc, nd)
        SSE = e @ e
        # numeric Jacobian
        J = np.zeros((len(e), len(θ)))
        for i in range(len(θ)):
            tp   = θ.copy(); tp[i] += 1e-6
            ei   = compute_error_bj(tp, y, u, nb, nf, nc, nd)
            J[:,i] = (ei - e) / 1e-6
        A = J.T@J; g = J.T@e
        try:
            Δθ = np.linalg.solve(A + mu*np.eye(len(θ)), g)
        except LinAlgError:
            Δθ = pinv(A + mu*np.eye(len(θ))) @ g
        θn = θ + Δθ
        en = compute_error_bj(θn, y, u, nb, nf, nc, nd)
        if en@en < SSE:
            θ, mu = θn, mu*0.1
            if np.linalg.norm(Δθ) < tol:
                break
        else:
            mu *= 10
    return θ

def q_test(resid, model_df, lags=50, alpha=0.05):
    r    = resid - resid.mean()
    N    = len(r)
    var  = r.var()
    Q    = sum(((r[t:]@r[:-t])/( (N-t)*var))**2 for t in range(1,lags+1))
    Qs   = N * Q
    crit = chi2.ppf(1-alpha, df=lags-model_df)
    print(f"Whiteness Q-test: Q={Qs:.2f}, crit={crit:.2f} ->", 
          "PASS" if Qs<crit else "FAIL")
    return Qs, crit

def s_test(e, u, theta, nb, nf, K=20, alpha=0.05):
    N = len(e)
    e_n = (e - e.mean())/np.std(e)
    f   = np.r_[1, theta[nb:nb+nf]]
    alpha_t = np.zeros(N)
    for t in range(len(f), N):
        alpha_t[t] = u[t] - sum(f[j]*alpha_t[t-j] for j in range(1,len(f)))
    a_n = (alpha_t - alpha_t.mean())/np.std(alpha_t)
    S = sum((np.sum(a_n[:N-t]*e_n[t:])/(N-t))**2 for t in range(K+1))
    Ss   = N * S
    crit = chi2.ppf(1-alpha, df=K-nf)
    print(f"S-test: S={Ss:.2f}, crit={crit:.2f} ->", 
          "PASS" if Ss<crit else "FAIL")
    return Ss, crit

# 2) Load & split the daily data
df   = pd.read_csv("jena_climate_2009_2016.csv", parse_dates=["Date Time"], dayfirst=True)\
         .drop_duplicates("Date Time")\
         .set_index("Date Time")\
         .sort_index()
y    = df["T (degC)"].resample("D").mean().interpolate("time")
u    = df["Tpot (K)"].resample("D").mean().interpolate("time")
train = y["2009-01-01":"2010-12-31"]
test  = y["2011-01-01":"2011-12-31"]
next_ = y["2012-01-01":"2012-12-31"]

# 3) First‐difference for stationarity
y_diff = y.diff().dropna()
u_diff = u.diff().dropna()
y_train = y_diff[train.index[1:]].values
u_train = u_diff[train.index[1:]].values
y_test  = y_diff[test.index[1:]].values
u_test  = u_diff[test.index[1:]].values

# 4) Fit the chosen model: ARXMA(2,1) + MA(1)
nb,nf,nc,nd = 2,1,0,1
# Seed input gains via OLS
X    = np.column_stack([u_train[i:len(u_train)-(nb-1-i)] for i in range(nb)])
y_t  = y_train[nb-1:]
b_init = np.linalg.lstsq(X, y_t, rcond=None)[0]
theta0 = np.zeros(nb+nf+nc+nd)
theta0[:nb] = b_init

theta_est = levenberg_marquardt_bj(y_train, u_train, theta0, nb,nf,nc,nd)
print("Estimated θ:", np.round(theta_est,4))

# 5) a) Whiteness test
resid = compute_error_bj(theta_est, y_train, u_train, nb,nf,nc,nd)
Qs, Qcrit = q_test(resid, model_df=nb+nf+nc+nd)

# 6) b) Estimated error variance & covariance of θ
sigma2   = np.var(resid, ddof=1)
# covariance = σ² (J'J)⁻¹
J        = np.zeros((len(resid), len(theta_est)))
for i in range(len(theta_est)):
    tp     = theta_est.copy(); tp[i] += 1e-6
    J[:,i] = (compute_error_bj(tp, y_train, u_train, nb,nf,nc,nd) - resid)/1e-6
cov_theta = sigma2 * pinv(J.T@J)
print(f"\nEstimated error variance σ² = {sigma2:.4f}")
print("Covariance matrix of θ:\n", cov_theta)

# 7) c) Bias check
print(f"\nMean residual = {resid.mean():.4e} →", 
      "unbiased" if abs(resid.mean())<1e-3 else "bias present")

# 8) d) Compare residual vs forecast error variances
# 1-step forecast errors
b = theta_est[:nb]; f = np.r_[1, theta_est[nb:nb+nf]]
pred1 = []
for i in range(len(y_test)):
    arx = sum(b[j]*u_test[i-j] for j in range(nb) if i-j>=0)
    fb  = sum(f[k]*pred1[i-k] for k in range(1,len(f)) if i-k>=0)
    pred1.append(arx - fb)
err1_var = np.var(y_test - np.array(pred1), ddof=1)
print(f"\n1-step forecast error variance = {err1_var:.4f}")
print(f"Residual variance                = {sigma2:.4f}")

# 365-step forecast
u_all = np.concatenate([u_train, u_test, u_diff[next_.index[1:]].values])
yg    = np.zeros(len(u_all))
for t in range(max(nb,nf), len(u_all)):
    yg[t] = sum(b[j]*u_all[t-j] for j in range(nb)) - sum(f[k]*yg[t-k] for k in range(1,len(f)))
yg_next = yg[len(u_train)+len(y_test):]
# invert back
start = test.iloc[-1]
yhat  = [start]
for d in yg_next: yhat.append(yhat[-1]+d)
errh_var = np.var(next_.iloc[1:].values - np.array(yhat[1:]), ddof=1)
print(f"365-step forecast error variance = {errh_var:.4f}")

# 9) e) Zero‐pole cancellation & final CIs
se = np.sqrt(np.diag(cov_theta))
cis = np.vstack([theta_est - 1.96*se, theta_est +1.96*se]).T
print("\nFinal 95% CIs (drop any with zero-crossing):")
for i,(low,high) in enumerate(cis):
    print(f" θ[{i}] ∈ [{low:.4f}, {high:.4f}]", "→ cancel" if low*high<0 else "")
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf

# ───────────────────────────────────────────────────────────────
# Update this path as needed
csv_path = "jena_climate_2009_2016.csv"
# ───────────────────────────────────────────────────────────────

# 1. Load dataset and daily-resample temperature (T) and exogenous (Tpot)
df = pd.read_csv(csv_path, parse_dates=["Date Time"], dayfirst=True)
df = df.drop_duplicates("Date Time").set_index("Date Time").sort_index()

y = df["T (degC)"].resample("D").mean().interpolate("time")
u = df["Tpot (K)"].resample("D").mean().interpolate("time")

# 2. First differencing for stationarity
y_diff = y.diff().dropna()
u_diff = u.diff().dropna()

# 3. Use training range
train = y["2009-01-01":"2010-12-31"]
y_train = y_diff[train.index[1:]].values
u_train = u_diff[train.index[1:]].values

# 4. GPAC function
def gpac_table(acf_vals, max_k=7, max_j=7):
    gpac = np.full((max_j, max_k), np.nan)
    for j in range(max_j):
        for k in range(1, max_k + 1):
            try:
                D = np.array([[acf_vals[abs(j + i - m)] for m in range(k)] for i in range(k)])
                N = D.copy()
                for i in range(k):
                    N[i, -1] = acf_vals[j + i + 1] if j + i + 1 < len(acf_vals) else 0
                det_D = np.linalg.det(D)
                det_N = np.linalg.det(N)
                if abs(det_D) > 1e-8:
                    gpac[j, k - 1] = det_N / det_D
            except:
                continue
    return pd.DataFrame(gpac, index=[f"j={j}" for j in range(max_j)],
                        columns=[f"k={k}" for k in range(1, max_k + 1)])

# 5. G-GPAC
ry = acf(y_train, nlags=20, fft=True)
gpac_G = gpac_table(ry, max_k=7, max_j=7)

plt.figure(figsize=(10, 6))
sns.heatmap(gpac_G.astype(float), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("G-GPAC Table (Input-Output Correlation)")
plt.tight_layout()
plt.savefig("gpac_ggpac.png", dpi=300)
plt.show()

# 6. H-GPAC (Residuals → assume white noise for now, simulate with AR(1) error)
from statsmodels.tsa.arima_process import ArmaProcess
np.random.seed(0)
ar = np.array([1, -0.5])  # AR(1)
ma = np.array([1])
arma_process = ArmaProcess(ar, ma)
resid_sim = arma_process.generate_sample(nsample=len(y_train))

rh = acf(resid_sim, nlags=20, fft=True)
gpac_H = gpac_table(rh, max_k=7, max_j=7)

plt.figure(figsize=(10, 6))
sns.heatmap(gpac_H.astype(float), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("H-GPAC Table (Noise Process Residual Correlation)")
plt.tight_layout()
plt.savefig("gpac_hgpac.png", dpi=300)
plt.show()

# %%
