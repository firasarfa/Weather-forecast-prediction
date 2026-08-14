import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_prep import load_and_preprocess_data


# Set better style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def display_data_info(df):
    """Display basic dataset information."""
    print("\n" + "="*70)
    print("DATASET INFORMATION")
    print("="*70)
    print(f"Shape: {df.shape}")
    print(f"Time Range: {df.index.min()} to {df.index.max()}")
    print(f"Duration: {(df.index.max() - df.index.min()).days} days")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:")
    print(df.describe().round(2))
    print("="*70 + "\n")

def plot_data_sample(df, n_samples=200):
    """Display first N samples of the data."""
    print("\n--- Sample of Raw Data (First 200 Hours) ---")
    print(df.head(n_samples))
    print("\n")

def plot_correlation_heatmap(df, title="Correlation Heatmap"):
    """Enhanced correlation heatmap with better styling."""
    print(f"\n--- Creating: {title} ---")
    
    # Calculate correlation
    corr = df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(corr, 
                mask=mask,
                annot=True, 
                fmt=".2f",
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
                vmin=-1, vmax=1)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_time_series_overview(df, column):
    """Comprehensive time series visualization."""
    print(f"\n--- Creating: Time Series Overview for {column} ---")
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. Full time series
    axes[0].plot(df.index, df[column], color='steelblue', linewidth=0.7)
    axes[0].set_title(f'{column} - Complete Time Series (2014-2016)', 
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Temperature (degC)')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Daily averages
    daily = df[column].resample('D').mean()
    axes[1].plot(daily.index, daily, color='forestgreen', linewidth=1.2)
    axes[1].set_title('Daily Average Temperature', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Temperature (degC)')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Monthly averages with std
    monthly = df[column].resample('M').agg(['mean', 'std'])
    axes[2].plot(monthly.index, monthly['mean'], color='orangered', 
                linewidth=2, label='Mean')
    axes[2].fill_between(monthly.index, 
                         monthly['mean'] - monthly['std'],
                         monthly['mean'] + monthly['std'],
                         alpha=0.3, color='orangered', label='+-1 Std Dev')
    axes[2].set_title('Monthly Average +- Standard Deviation', 
                     fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Temperature (degC)')
    axes[2].set_xlabel('Date')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_seasonal_patterns(df, column):
    """Analyze seasonal and hourly patterns."""
    print(f"\n--- Creating: Seasonal Patterns for {column} ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract time components
    df_temp = df.copy()
    df_temp['Hour'] = df_temp.index.hour
    df_temp['Month'] = df_temp.index.month
    df_temp['DayOfWeek'] = df_temp.index.dayofweek
    
    # 1. Hourly pattern
    hourly_avg = df_temp.groupby('Hour')[column].mean()
    axes[0, 0].plot(hourly_avg.index, hourly_avg.values, 
                   marker='o', linewidth=2, markersize=6, color='teal')
    axes[0, 0].set_title('Average Temperature by Hour of Day', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Temperature (degC)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(range(0, 24, 3))
    
    # 2. Monthly pattern
    monthly_avg = df_temp.groupby('Month')[column].mean()
    axes[0, 1].bar(monthly_avg.index, monthly_avg.values, 
                  color='coral', edgecolor='black')
    axes[0, 1].set_title('Average Temperature by Month', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Temperature (degC)')
    axes[0, 1].set_xticks(range(1, 13))
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Day of week pattern
    dow_avg = df_temp.groupby('DayOfWeek')[column].mean()
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 0].plot(dow_avg.index, dow_avg.values, 
                   marker='s', linewidth=2, markersize=8, color='purple')
    axes[1, 0].set_title('Average Temperature by Day of Week', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Day of Week')
    axes[1, 0].set_ylabel('Temperature (degC)')
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(dow_names)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Distribution histogram
    axes[1, 1].hist(df[column].dropna(), bins=50, 
                   color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Temperature Distribution', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Temperature (degC)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"Mean: {df[column].mean():.2f} degC\n"
    stats_text += f"Std: {df[column].std():.2f} degC\n"
    stats_text += f"Min: {df[column].min():.2f} degC\n"
    stats_text += f"Max: {df[column].max():.2f} degC"
    axes[1, 1].text(0.02, 0.98, stats_text,
                   transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def plot_boxplot_by_month(df, column):
    """Boxplot showing temperature distribution by month."""
    print(f"\n--- Creating: Boxplot by Month for {column} ---")
    
    df_temp = df.copy()
    df_temp['Month'] = df_temp.index.month
    
    plt.figure(figsize=(14, 6))
    # Fix seaborn warning by setting hue explicitly
    sns.boxplot(data=df_temp, x='Month', y=column, hue='Month', palette='Set2', legend=False)
    plt.title('Temperature Distribution by Month', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Temperature (degC)', fontsize=12)
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("\n" + "="*70)
    print("COMPREHENSIVE DATA VISUALIZATION")
    print("="*70)
    
    # 1. Load and visualize raw data
    df_raw = load_and_preprocess_data()
    
    if df_raw is not None and not df_raw.empty:
        # Display info and sample
        display_data_info(df_raw)
        plot_data_sample(df_raw, n_samples=200)
        
        # Correlation heatmap of all columns
        if df_raw.shape[1] > 1:
            plot_correlation_heatmap(df_raw, "Correlation Heatmap - All Variables")
        
        # Time series analysis
        plot_time_series_overview(df_raw, TARGET_COL)
        plot_seasonal_patterns(df_raw, TARGET_COL)
        plot_boxplot_by_month(df_raw, TARGET_COL)
    
    # 2. Visualize engineered features
    if get_feature_dataframe is not None:
        df_features = get_feature_dataframe()
        
        if not df_features.empty:
            print("\n" + "="*70)
            print("ENGINEERED FEATURES VISUALIZATION")
            print("="*70)
            
            display_data_info(df_features)
            plot_correlation_heatmap(df_features, 
                                    "Correlation Heatmap - LSTM Features")
            
            print("\n[SUCCESS] All visualizations completed successfully!")
        else:
            print("\n[WARNING] No feature data available.")
    else:
        print("\n[WARNING] get_feature_dataframe() not available.")