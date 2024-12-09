# Import necessary libraries for data manipulation, analysis, and visualization

# NumPy: Fundamental package for numerical computing in Python
import numpy as np  

# Pandas: Library for data manipulation and analysis, particularly useful for tabular data
import pandas as pd  

# Matplotlib: Comprehensive library for creating static, animated, and interactive visualizations
import matplotlib.pyplot as plt  

# Seaborn: Statistical data visualization library built on top of Matplotlib
import seaborn as sns  

# Optional settings to enhance visualizations
# Set Seaborn theme for better-looking plots
sns.set_theme(style="whitegrid")

# Ensure Matplotlib plots appear inline when using Jupyter Notebook
# Uncomment the line below if you're working in a Jupyter Notebook
# %matplotlib inline
# Import necessary libraries for data manipulation and analysis
import pandas as pd  

# Import the first CSV file (Benin Malanville data)
try:
    benin_malanville_file = 'benin-malanville.csv'  # Replace with the correct path if needed
    df_benin_malanville = pd.read_csv(benin_malanville_file)  # Load the CSV into a DataFrame
    print(f"Data successfully loaded from CSV file: {benin_malanville_file}")
except FileNotFoundError:
    print(f"CSV file not found: {benin_malanville_file}")

# Import the second CSV file (Sierra Leone Bumbuna data)
try:
    sierraleone_bumbuna_file = 'sierraleone-bumbuna.csv'  # Replace with the correct path if needed
    df_sierraleone_bumbuna = pd.read_csv(sierraleone_bumbuna_file)  # Load the CSV into a DataFrame
    print(f"Data successfully loaded from CSV file: {sierraleone_bumbuna_file}")
except FileNotFoundError:
    print(f"CSV file not found: {sierraleone_bumbuna_file}")

# Import the third CSV file (Togo Dapaong QC data)
try:
    togo_dapaong_qc_file = 'togo-dapaong_qc.csv'  # Replace with the correct path if needed
    df_togo_dapaong_qc = pd.read_csv(togo_dapaong_qc_file)  # Load the CSV into a DataFrame
    print(f"Data successfully loaded from CSV file: {togo_dapaong_qc_file}")
except FileNotFoundError:
    print(f"CSV file not found: {togo_dapaong_qc_file}")

# Display the first few rows of each dataset for verification
print("\nBenin Malanville Data Preview:")
print(df_benin_malanville.head())  # Preview Benin Malanville data

print("\nSierra Leone Bumbuna Data Preview:")
print(df_sierraleone_bumbuna.head())  # Preview Sierra Leone Bumbuna data

print("\nTogo Dapaong QC Data Preview:")
print(df_togo_dapaong_qc.head())  # Preview Togo Dapaong QC data
# Data Quality Check: Look for missing values, negative values in specific columns
print("\nData Quality Check:")

# Checking for missing values
missing_values = df.isnull().sum()
print("Missing Values in Columns:")
print(missing_values)

# Checking for negative values in columns where only positive values should exist
columns_to_check = ['GHI', 'DNI', 'DHI', 'WS', 'WSgust', 'ModA', 'ModB']
negative_values = (df[columns_to_check] < 0).sum()
print("\nNegative Values in Columns:")
print(negative_values)

# Checking for outliers using the IQR method (Interquartile Range)
Q1 = df[columns_to_check].quantile(0.25)
Q3 = df[columns_to_check].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[columns_to_check] < (Q1 - 1.5 * IQR)) | (df[columns_to_check] > (Q3 + 1.5 * IQR))).sum()
print("\nOutliers in Columns (using IQR method):")
print(outliers)

# ---------------------------------------------------------------
# Time Series Analysis: Plotting GHI, DNI, DHI, and Tamb over time
# Convert time column to datetime if it's not already
df['Time'] = pd.to_datetime(df['Time'])  # Replace 'Time' with the actual column name

# Resample data by month for trend analysis
df_monthly = df.resample('M', on='Time').mean()

# Plot the time series of GHI, DNI, DHI, and Tamb
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(df_monthly['GHI'], label='GHI', color='orange')
plt.title('Monthly GHI')
plt.xlabel('Date')
plt.ylabel('GHI')

plt.subplot(2, 2, 2)
plt.plot(df_monthly['DNI'], label='DNI', color='green')
plt.title('Monthly DNI')
plt.xlabel('Date')
plt.ylabel('DNI')

plt.subplot(2, 2, 3)
plt.plot(df_monthly['DHI'], label='DHI', color='blue')
plt.title('Monthly DHI')
plt.xlabel('Date')
plt.ylabel('DHI')

plt.subplot(2, 2, 4)
plt.plot(df_monthly['Tamb'], label='Tamb', color='red')
plt.title('Monthly Temperature (Tamb)')
plt.xlabel('Date')
plt.ylabel('Temperature')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Evaluate impact of Cleaning on sensor readings over time
df_cleaned = df[df['Cleaning'] == 'Yes']  # Assuming 'Cleaning' is a categorical column with 'Yes'/'No'

plt.figure(figsize=(10, 6))
plt.plot(df_cleaned['Time'], df_cleaned['ModA'], label='ModA (Cleaned)', color='blue')
plt.plot(df_cleaned['Time'], df_cleaned['ModB'], label='ModB (Cleaned)', color='red')
plt.title('Impact of Cleaning on Sensor Readings (ModA and ModB)')
plt.xlabel('Time')
plt.ylabel('Sensor Readings')
plt.legend()
plt.show()

# ---------------------------------------------------------------
# Correlation Analysis: Correlation Matrix
corr = df[['GHI', 'DNI', 'DHI', 'TModA', 'TModB']].corr()
print("\nCorrelation Matrix:")
print(corr)

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# ---------------------------------------------------------------
# Wind Analysis: Radial bar plot or wind rose
# Assuming 'WD' is the wind direction and 'WS' is the wind speed

# Convert wind direction to categorical wind sectors (e.g., N, NE, E, SE, etc.)
df['Wind Sector'] = pd.cut(df['WD'], bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                            labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

# Create a wind rose (bar plot)
wind_data = df.groupby('Wind Sector')['WS'].mean()  # Average wind speed by wind direction
wind_data.plot(kind='bar', color='skyblue', figsize=(8, 6))
plt.title('Wind Speed by Wind Direction (Wind Rose)')
plt.xlabel('Wind Direction')
plt.ylabel('Average Wind Speed (WS)')
plt.show()

# ---------------------------------------------------------------
# Temperature Analysis: Examine RH's impact on temperature and solar radiation
plt.figure(figsize=(10, 6))
sns.scatterplot(x='RH', y='Tamb', data=df, hue='GHI', palette='viridis')
plt.title('Relative Humidity vs. Temperature (Tamb) with GHI')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Temperature (Tamb)')
plt.show()

# ---------------------------------------------------------------
# Histograms: Plot histograms for GHI, DNI, DHI, WS, and Temperature
plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
df['GHI'].hist(bins=30, color='orange')
plt.title('Histogram of GHI')

plt.subplot(2, 3, 2)
df['DNI'].hist(bins=30, color='green')
plt.title('Histogram of DNI')

plt.subplot(2, 3, 3)
df['DHI'].hist(bins=30, color='blue')
plt.title('Histogram of DHI')

plt.subplot(2, 3, 4)
df['WS'].hist(bins=30, color='purple')
plt.title('Histogram of Wind Speed (WS)')

plt.subplot(2, 3, 5)
df['Tamb'].hist(bins=30, color='red')
plt.title('Histogram of Temperature (Tamb)')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Z-Score Analysis: Detecting outliers using Z-scores
z_scores = np.abs(zscore(df[['GHI', 'DNI', 'DHI', 'WS', 'WSgust', 'ModA', 'ModB']]))
outliers_z_score = (z_scores > 3).sum(axis=0)
print("\nOutliers based on Z-Score (Threshold = 3):")
print(outliers_z_score)

# ---------------------------------------------------------------
# Bubble chart: Exploring the relationship between GHI, Tamb, and WS, with RH as bubble size
plt.figure(figsize=(10, 6))
plt.scatter(df['GHI'], df['Tamb'], s=df['RH'] * 10, c=df['WS'], cmap='coolwarm', alpha=0.5)
plt.colorbar(label='Wind Speed (WS)')
plt.title('Bubble Chart: GHI vs. Temperature (Tamb) vs. Wind Speed (WS)')
plt.xlabel('GHI')
plt.ylabel('Temperature (Tamb)')
plt.show()

# ---------------------------------------------------------------
# Data Cleaning: Handle missing values and anomalies
# Fill missing values for specific columns or drop rows if needed
df.fillna({'GHI': df['GHI'].mean(), 'DNI': df['DNI'].mean(), 'DHI': df['DHI'].mean()}, inplace=True)

# Drop rows with completely null 'Comments' column
df.dropna(subset=['Comments'], inplace=True)

print("\nData after cleaning:")
print(df.head())