import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Dictionary mapping countries to continents
CONTINENT_MAPPING = {
    # Europe
    'Denmark': 'Europe', 'France': 'Europe', 'Germany': 'Europe', 'Italy': 'Europe',
    'Netherlands': 'Europe', 'Norway': 'Europe', 'Sweden': 'Europe', 'United Kingdom': 'Europe',
    'Spain': 'Europe', 'Portugal': 'Europe', 'Greece': 'Europe', 'Ireland': 'Europe',
    'Finland': 'Europe', 'Belgium': 'Europe', 'Austria': 'Europe', 'Switzerland': 'Europe',
    'Poland': 'Europe', 'Romania': 'Europe', 'Ukraine': 'Europe', 'Belarus': 'Europe',
    'Estonia': 'Europe', 'Latvia': 'Europe', 'Lithuania': 'Europe', 'Moldova': 'Europe',
    'Slovakia': 'Europe', 'Hungary': 'Europe', 'Croatia': 'Europe', 'Slovenia': 'Europe',
    'Bosnia And Herzegovina': 'Europe', 'Albania': 'Europe', 'Bulgaria': 'Europe',
    'Montenegro': 'Europe', 'Serbia': 'Europe', 'Åland': 'Europe',
    
    # Asia
    'China': 'Asia', 'Japan': 'Asia', 'India': 'Asia', 'Russia': 'Asia',
    'South Korea': 'Asia', 'Indonesia': 'Asia', 'Malaysia': 'Asia', 'Thailand': 'Asia',
    'Vietnam': 'Asia', 'Philippines': 'Asia', 'Myanmar': 'Asia', 'Kazakhstan': 'Asia',
    'Pakistan': 'Asia', 'Bangladesh': 'Asia', 'Nepal': 'Asia', 'Sri Lanka': 'Asia',
    'Cambodia': 'Asia', 'Laos': 'Asia', 'Mongolia': 'Asia', 'Taiwan': 'Asia',
    'Georgia': 'Asia', 'Armenia': 'Asia', 'Azerbaijan': 'Asia',
    
    # North America
    'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
    'Cuba': 'North America', 'Jamaica': 'North America', 'Haiti': 'North America',
    'Dominican Republic': 'North America', 'Guatemala': 'North America', 'Panama': 'North America',
    'Costa Rica': 'North America', 'Nicaragua': 'North America', 'Honduras': 'North America',
    
    # South America
    'Brazil': 'South America', 'Argentina': 'South America', 'Chile': 'South America',
    'Peru': 'South America', 'Colombia': 'South America', 'Venezuela': 'South America',
    'Ecuador': 'South America', 'Bolivia': 'South America', 'Paraguay': 'South America',
    'Uruguay': 'South America', 'Guyana': 'South America', 'Suriname': 'South America',
    
    # Africa
    'South Africa': 'Africa', 'Egypt': 'Africa', 'Nigeria': 'Africa', 'Kenya': 'Africa',
    'Morocco': 'Africa', 'Algeria': 'Africa', 'Tunisia': 'Africa', 'Libya': 'Africa',
    'Ghana': 'Africa', 'Zimbabwe': 'Africa', 'Angola': 'Africa', 'Mozambique': 'Africa',
    'Tanzania': 'Africa', 'Ethiopia': 'Africa', 'Uganda': 'Africa', 'Cameroon': 'Africa',
    'Madagascar': 'Africa', "Côte D'Ivoire": 'Africa', 'Sudan': 'Africa',
    
    # Oceania
    'Australia': 'Oceania', 'New Zealand': 'Oceania', 'Papua New Guinea': 'Oceania',
    'Fiji': 'Oceania', 'Solomon Islands': 'Oceania'
}

def load_and_prepare_data(temp_csv_path, emissions_csv_path):
    # Read temperature data
    print("Reading temperature data...")
    temp_df = pd.read_csv(temp_csv_path)
    
    # Convert date string to datetime
    temp_df['dt'] = pd.to_datetime(temp_df['dt'])
    # Extract year from datetime
    temp_df['Year'] = temp_df['dt'].dt.year
    
    # Map countries to continents
    print("Mapping countries to continents...")
    temp_df['Continent'] = temp_df['Country'].map(CONTINENT_MAPPING)
    
    # Print countries that don't have a continent mapping
    unmapped_countries = temp_df[temp_df['Continent'].isna()]['Country'].unique()
    if len(unmapped_countries) > 0:
        print("\nWarning: The following countries don't have a continent mapping:")
        print(sorted(unmapped_countries))
    
    # Read emissions data
    print("Reading emissions data...")
    emissions_df = pd.read_csv(emissions_csv_path)
    
    return temp_df, emissions_df

def prepare_temperature_data(temp_df):
    # Remove rows where continent mapping is missing
    temp_df = temp_df.dropna(subset=['Continent'])
    
    # Calculate continental averages
    continental_temp = temp_df.groupby(['Year', 'Continent'])['AverageTemperature'].mean().reset_index()
    
    # Calculate global average
    global_temp = temp_df.groupby('Year')['AverageTemperature'].mean().reset_index()
    global_temp['Continent'] = 'Global Average'
    
    # Combine continental and global data
    yearly_temp = pd.concat([continental_temp, global_temp])
    
    return yearly_temp

def prepare_emissions_data(emissions_df):
    # Group by Year, sum all countries' emissions
    yearly_emissions = emissions_df.groupby('Year')['Annual CO₂ emissions'].sum().reset_index()
    return yearly_emissions

# [Rest of the functions remain the same as in your current script]
def create_temperature_plot(yearly_temp):
    # Filter data from 1850 onwards
    yearly_temp = yearly_temp[yearly_temp['Year'] >= 1850].copy()
    
    plt.figure(figsize=(15, 8))
    
    # Define a color palette for continents
    colors = {
        'Europe': '#1f77b4',
        'Asia': '#ff7f0e',
        'North America': '#2ca02c',
        'South America': '#d62728',
        'Africa': '#9467bd',
        'Oceania': '#8c564b',
        'Global Average': '#000000'
    }
    
    # Get baseline temperature (1850-1900 average)
    global_data = yearly_temp[yearly_temp['Continent'] == 'Global Average']
    baseline_temp = global_data[
        (global_data['Year'] >= 1850) & 
        (global_data['Year'] <= 1900)
    ]['AverageTemperature'].mean()
    
    # Calculate threshold temperatures
    two_degree_line = baseline_temp + 2
    one_five_degree_line = baseline_temp + 1.5
    
    # Plot each continent
    for continent in yearly_temp['Continent'].unique():
        if continent != 'Global Average':
            continent_data = yearly_temp[yearly_temp['Continent'] == continent]
            plt.plot(continent_data['Year'], continent_data['AverageTemperature'], 
                    label=continent, color=colors[continent], alpha=0.5)
    
    # Plot global average with thicker black line
    global_data = yearly_temp[yearly_temp['Continent'] == 'Global Average']
    plt.plot(global_data['Year'], global_data['AverageTemperature'],
            label='Global Average', color='black', linewidth=2)
    
    # Add threshold lines
    plt.axhline(y=two_degree_line, color='red', linestyle='--', 
                label='+2°C threshold', linewidth=2)
    plt.axhline(y=one_five_degree_line, color='orange', linestyle='--', 
                label='+1.5°C threshold', linewidth=2)
    plt.axhline(y=baseline_temp, color='gray', linestyle=':', 
                label='Pre-industrial baseline (1850-1900)', alpha=0.5)
    
    plt.title('Average Temperature by Continent Over Time (1850 onwards)')
    plt.xlabel('Year')
    plt.ylabel('Average Temperature (°C)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    return plt

def create_emissions_plot(yearly_emissions):
    # Filter data from 1850 onwards
    yearly_emissions = yearly_emissions[yearly_emissions['Year'] >= 1850].copy()
    
    plt.figure(figsize=(15, 8))
    
    plt.plot(yearly_emissions['Year'], yearly_emissions['Annual CO₂ emissions'], 
            color='red', linewidth=2)
    
    plt.title('Global CO₂ Emissions Over Time (1850 onwards)')
    plt.xlabel('Year')
    plt.ylabel('Annual CO₂ Emissions')
    plt.grid(True, alpha=0.3)
    
    return plt

def create_combined_plot(yearly_temp, yearly_emissions):
    # Filter data from 1850 onwards
    yearly_temp = yearly_temp[yearly_temp['Year'] >= 1850].copy()
    yearly_emissions = yearly_emissions[yearly_emissions['Year'] >= 1850].copy()
    
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Define colors for continents
    colors = {
        'Europe': '#1f77b4',
        'Asia': '#ff7f0e',
        'North America': '#2ca02c',
        'South America': '#d62728',
        'Africa': '#9467bd',
        'Oceania': '#8c564b',
        'Global Average': '#000000'
    }
    
    # Get baseline temperature (1850-1900 average)
    global_data = yearly_temp[yearly_temp['Continent'] == 'Global Average']
    baseline_temp = global_data[
        (global_data['Year'] >= 1850) & 
        (global_data['Year'] <= 1900)
    ]['AverageTemperature'].mean()
    
    # Calculate threshold temperatures
    two_degree_line = baseline_temp + 2
    one_five_degree_line = baseline_temp + 1.5
    
    # Temperature lines on primary y-axis
    for continent in yearly_temp['Continent'].unique():
        continent_data = yearly_temp[yearly_temp['Continent'] == continent]
        if continent == 'Global Average':
            ax1.plot(continent_data['Year'], continent_data['AverageTemperature'],
                    label=continent, color=colors[continent], linewidth=2)
        else:
            ax1.plot(continent_data['Year'], continent_data['AverageTemperature'],
                    label=continent, color=colors[continent], alpha=0.5)
    
    # Add threshold lines
    ax1.axhline(y=two_degree_line, color='red', linestyle='--', 
                label='+2°C threshold', linewidth=2)
    ax1.axhline(y=one_five_degree_line, color='orange', linestyle='--', 
                label='+1.5°C threshold', linewidth=2)
    ax1.axhline(y=baseline_temp, color='gray', linestyle=':', 
                label='Pre-industrial baseline (1850-1900)', alpha=0.5)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Temperature (°C)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Emissions line on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(yearly_emissions['Year'], yearly_emissions['Annual CO₂ emissions'],
            color='red', linewidth=2, label='CO₂ Emissions')
    ax2.set_ylabel('Annual CO₂ Emissions', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Temperature by Continent and CO₂ Emissions Over Time (1850 onwards)')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.15, 1))
    
    return plt

def create_future_projections_plot(yearly_temp, yearly_emissions):
    # Filter data from 1850 onwards
    yearly_temp = yearly_temp[yearly_temp['Year'] >= 1850].copy()
    yearly_emissions = yearly_emissions[yearly_emissions['Year'] >= 1850].copy()
    
    # Prepare regression models
    global_temp = yearly_temp[yearly_temp['Continent'] == 'Global Average']
    
    # Prepare data for regression
    X_temp = global_temp[['Year']].values
    y_temp = global_temp['AverageTemperature'].values
    X_emissions = yearly_emissions[['Year']].values
    y_emissions = yearly_emissions['Annual CO₂ emissions'].values
    
    # Create and fit models
    temp_model = LinearRegression()
    temp_model.fit(X_temp, y_temp)
    emissions_model = LinearRegression()
    emissions_model.fit(X_emissions, y_emissions)
    
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Colors and baseline definitions
    colors = {
        'Europe': '#1f77b4',
        'Asia': '#ff7f0e',
        'North America': '#2ca02c',
        'South America': '#d62728',
        'Africa': '#9467bd',
        'Oceania': '#8c564b',
        'Global Average': '#000000'
    }
    
    # Get baseline temperature (1850-1900 average)
    baseline_temp = global_temp[
        (global_temp['Year'] >= 1850) & 
        (global_temp['Year'] <= 1900)
    ]['AverageTemperature'].mean()
    
    # Calculate threshold lines
    two_degree_line = baseline_temp + 2
    one_five_degree_line = baseline_temp + 1.5
    
    # Get last years and set projection period
    last_temp_year = global_temp['Year'].max()
    last_emissions_year = yearly_emissions['Year'].max()
    projection_end = 2100
    
    # Create projection years arrays
    temp_projection_years = np.array(range(last_temp_year, projection_end + 1)).reshape(-1, 1)
    emissions_projection_years = np.array(range(last_emissions_year, projection_end + 1)).reshape(-1, 1)
    
    # Get regression-based projections
    temp_regression_pred = temp_model.predict(temp_projection_years)
    emissions_regression_pred = emissions_model.predict(emissions_projection_years)
    
    # Calculate current rate projections (original method)
    recent_global_temp = global_temp[global_temp['Year'] == last_temp_year]['AverageTemperature'].iloc[0]
    recent_emissions = yearly_emissions[yearly_emissions['Year'] == last_emissions_year]['Annual CO₂ emissions'].iloc[0]
    
    last_30_years = global_temp[global_temp['Year'] >= last_temp_year - 30]
    temp_change_rate = (last_30_years['AverageTemperature'].iloc[-1] - 
                       last_30_years['AverageTemperature'].iloc[0]) / 30
    
    temp_increase_rate = temp_change_rate * 0.85
    emission_increase_rate = recent_emissions * 0.005
    
    current_rate_temps = [recent_global_temp + temp_increase_rate * (year - last_temp_year) 
                         for year in temp_projection_years.flatten()]
    current_rate_emissions = [recent_emissions + emission_increase_rate * (year - last_emissions_year) 
                            for year in emissions_projection_years.flatten()]
    
    # Plot historical temperature data
    for continent in yearly_temp['Continent'].unique():
        continent_data = yearly_temp[yearly_temp['Continent'] == continent]
        if continent == 'Global Average':
            ax1.plot(continent_data['Year'], continent_data['AverageTemperature'],
                    label='Historical Global Average', color=colors[continent], linewidth=2)
        else:
            ax1.plot(continent_data['Year'], continent_data['AverageTemperature'],
                    label=f'Historical {continent}', color=colors[continent], alpha=0.5)
    
    # Plot both types of temperature projections
    # ax1.plot(temp_projection_years, current_rate_temps, '--', color='black', 
    #          label=f'Rate-based Projection (from {last_temp_year})', linewidth=2)
    ax1.plot(temp_projection_years, temp_regression_pred, ':', color='purple', 
             label=f'Regression-based Temp Projection (from {last_temp_year})', linewidth=2)
    
    # Add threshold lines
    ax1.axhline(y=two_degree_line, color='red', linestyle='--', 
                label='+2°C threshold', linewidth=2)
    ax1.axhline(y=one_five_degree_line, color='orange', linestyle='--', 
                label='+1.5°C threshold', linewidth=2)
    ax1.axhline(y=baseline_temp, color='gray', linestyle=':', 
                label='Pre-industrial baseline (1850-1900)', alpha=0.5)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Temperature (°C)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Emissions on secondary y-axis
    ax2 = ax1.twinx()
    # Historical emissions
    ax2.plot(yearly_emissions['Year'], yearly_emissions['Annual CO₂ emissions'],
            color='red', linewidth=2, label='Historical CO₂ Emissions')
    # Both types of emissions projections
    # ax2.plot(emissions_projection_years, current_rate_emissions, '--', color='red', 
    #          label=f'Rate-based Emissions Projection (from {last_emissions_year})', alpha=0.7)
    ax2.plot(emissions_projection_years, emissions_regression_pred, ':', color='red', 
             label=f'Regression-based Emissions Projection (from {last_emissions_year})', alpha=0.7)
    
    ax2.set_ylabel('Annual CO₂ Emissions', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title(f'Historical Data and Future Projections (1850-2100)\nComparing Rate-based and Regression-based Projections')
    
    # Add vertical lines for last data years
    plt.axvline(x=last_temp_year, color='blue', linestyle='--', alpha=0.5, 
                label='Last Temperature Data')
    plt.axvline(x=last_emissions_year, color='red', linestyle='--', alpha=0.5, 
                label='Last Emissions Data')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.15, 1))
    
    return plt

def perform_regression_analysis(yearly_temp, yearly_emissions):
    
    # Prepare temperature data for global average
    global_temp = yearly_temp[yearly_temp['Continent'] == 'Global Average']
    
    # Merge temperature and emissions data
    merged_data = pd.merge(global_temp[['Year', 'AverageTemperature']], 
                          yearly_emissions[['Year', 'Annual CO₂ emissions']],
                          on='Year', how='inner')
    
    # Prepare data for regression
    X_temp = merged_data[['Year']].values
    y_temp = merged_data['AverageTemperature'].values
    X_emissions = merged_data[['Year']].values
    y_emissions = merged_data['Annual CO₂ emissions'].values
    
    # Create and fit temperature model
    temp_model = LinearRegression()
    temp_model.fit(X_temp, y_temp)
    
    # Create and fit emissions model
    emissions_model = LinearRegression()
    emissions_model.fit(X_emissions, y_emissions)
    
    # Make predictions
    temp_pred = temp_model.predict(X_temp)
    emissions_pred = emissions_model.predict(X_emissions)
    
    # Calculate metrics
    temp_r2 = r2_score(y_temp, temp_pred)
    temp_rmse = np.sqrt(mean_squared_error(y_temp, temp_pred))
    emissions_r2 = r2_score(y_emissions, emissions_pred)
    emissions_rmse = np.sqrt(mean_squared_error(y_emissions, emissions_pred))
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
    
    # Temperature regression plot
    ax1.scatter(X_temp, y_temp, color='blue', alpha=0.5, label='Actual Temperature')
    ax1.plot(X_temp, temp_pred, color='red', label='Regression Line')
    ax1.set_title('Temperature Linear Regression Analysis')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Temperature (°C)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add regression equation and metrics
    temp_eq = f'Temp = {temp_model.coef_[0]:.4f}*Year + {temp_model.intercept_:.2f}'
    ax1.text(0.05, 0.95, f'Equation: {temp_eq}\nR² = {temp_r2:.4f}\nRMSE = {temp_rmse:.4f}°C', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Emissions regression plot
    ax2.scatter(X_emissions, y_emissions, color='green', alpha=0.5, label='Actual Emissions')
    ax2.plot(X_emissions, emissions_pred, color='red', label='Regression Line')
    ax2.set_title('CO₂ Emissions Linear Regression Analysis')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Annual CO₂ Emissions')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add regression equation and metrics
    emissions_eq = f'Emissions = {emissions_model.coef_[0]:.4f}*Year + {emissions_model.intercept_:.2f}'
    ax2.text(0.05, 0.95, f'Equation: {emissions_eq}\nR² = {emissions_r2:.4f}\nRMSE = {emissions_rmse:.4f}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Return the models and metrics
    results = {
        'temperature': {
            'model': temp_model,
            'r2': temp_r2,
            'rmse': temp_rmse,
            'equation': temp_eq
        },
        'emissions': {
            'model': emissions_model,
            'r2': emissions_r2,
            'rmse': emissions_rmse,
            'equation': emissions_eq
        }
    }
    
    return plt, results

def calculate_period_statistics(data, year_column, value_column, current_year, periods=[5, 10, 20, 50]):
    """Calculate statistics for different time periods"""
    stats = {}
    
    # Calculate overall statistics
    stats['all_time'] = {
        'mean': data[value_column].mean(),
        'min': data[value_column].min(),
        'max': data[value_column].max(),
        'std': data[value_column].std(),
        'period': f"{data[year_column].min()}-{data[year_column].max()}"
    }
    
    # Calculate statistics for recent periods
    for period in periods:
        period_data = data[data[year_column] >= current_year - period]
        if not period_data.empty:
            stats[f'last_{period}_years'] = {
                'mean': period_data[value_column].mean(),
                'min': period_data[value_column].min(),
                'max': period_data[value_column].max(),
                'std': period_data[value_column].std(),
                'period': f"{period_data[year_column].min()}-{period_data[year_column].max()}"
            }
    
    return stats

def perform_enhanced_regression_analysis(yearly_temp, yearly_emissions):
    # Prepare temperature data for global average
    global_temp = yearly_temp[yearly_temp['Continent'] == 'Global Average']
    
    # Merge temperature and emissions data
    merged_data = pd.merge(global_temp[['Year', 'AverageTemperature']], 
                          yearly_emissions[['Year', 'Annual CO₂ emissions']],
                          on='Year', how='inner')
    
    current_year = merged_data['Year'].max()
    
    # Calculate statistics for both temperature and emissions
    temp_stats = calculate_period_statistics(
        merged_data, 'Year', 'AverageTemperature', current_year
    )
    emissions_stats = calculate_period_statistics(
        merged_data, 'Year', 'Annual CO₂ emissions', current_year
    )
    
    # Prepare data for regression
    X_temp = merged_data[['Year']].values
    y_temp = merged_data['AverageTemperature'].values
    X_emissions = merged_data[['Year']].values
    y_emissions = merged_data['Annual CO₂ emissions'].values
    
    # Create and fit models
    temp_model = LinearRegression()
    temp_model.fit(X_temp, y_temp)
    emissions_model = LinearRegression()
    emissions_model.fit(X_emissions, y_emissions)
    
    # Make predictions
    temp_pred = temp_model.predict(X_temp)
    emissions_pred = emissions_model.predict(X_emissions)
    
    # Calculate metrics
    temp_r2 = r2_score(y_temp, temp_pred)
    temp_rmse = np.sqrt(mean_squared_error(y_temp, temp_pred))
    emissions_r2 = r2_score(y_emissions, emissions_pred)
    emissions_rmse = np.sqrt(mean_squared_error(y_emissions, emissions_pred))
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
    
    # Temperature regression plot
    ax1.scatter(X_temp, y_temp, color='blue', alpha=0.5, label='Actual Temperature')
    ax1.plot(X_temp, temp_pred, color='red', label='Regression Line')
    ax1.set_title('Temperature Linear Regression Analysis')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Temperature (°C)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add regression equation and metrics
    temp_eq = f'Temp = {temp_model.coef_[0]:.4f}*Year + {temp_model.intercept_:.2f}'
    stats_text = f'Equation: {temp_eq}\nR² = {temp_r2:.4f}\nRMSE = {temp_rmse:.4f}°C\n\n'
    stats_text += f"All-time ({temp_stats['all_time']['period']}):\n"
    stats_text += f"Mean: {temp_stats['all_time']['mean']:.2f}°C\n"
    stats_text += f"Range: [{temp_stats['all_time']['min']:.2f}, {temp_stats['all_time']['max']:.2f}]°C\n\n"
    
    for period in [5, 10, 20, 50]:
        period_key = f'last_{period}_years'
        if period_key in temp_stats:
            stats = temp_stats[period_key]
            stats_text += f"Last {period} years ({stats['period']}):\n"
            stats_text += f"Mean: {stats['mean']:.2f}°C\n"
            stats_text += f"Range: [{stats['min']:.2f}, {stats['max']:.2f}]°C\n\n"
    
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Emissions regression plot
    ax2.scatter(X_emissions, y_emissions, color='green', alpha=0.5, label='Actual Emissions')
    ax2.plot(X_emissions, emissions_pred, color='red', label='Regression Line')
    ax2.set_title('CO₂ Emissions Linear Regression Analysis')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Annual CO₂ Emissions')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add regression equation and metrics
    emissions_eq = f'Emissions = {emissions_model.coef_[0]:.4f}*Year + {emissions_model.intercept_:.2f}'
    stats_text = f'Equation: {emissions_eq}\nR² = {emissions_r2:.4f}\nRMSE = {emissions_rmse:.4f}\n\n'
    stats_text += f"All-time ({emissions_stats['all_time']['period']}):\n"
    stats_text += f"Mean: {emissions_stats['all_time']['mean']:.2f}\n"
    stats_text += f"Range: [{emissions_stats['all_time']['min']:.2f}, {emissions_stats['all_time']['max']:.2f}]\n\n"
    
    for period in [5, 10, 20, 50]:
        period_key = f'last_{period}_years'
        if period_key in emissions_stats:
            stats = emissions_stats[period_key]
            stats_text += f"Last {period} years ({stats['period']}):\n"
            stats_text += f"Mean: {stats['mean']:.2f}\n"
            stats_text += f"Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n\n"
    
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Return the models, metrics, and statistics
    results = {
        'temperature': {
            'model': temp_model,
            'r2': temp_r2,
            'rmse': temp_rmse,
            'equation': temp_eq,
            'statistics': temp_stats
        },
        'emissions': {
            'model': emissions_model,
            'r2': emissions_r2,
            'rmse': emissions_rmse,
            'equation': emissions_eq,
            'statistics': emissions_stats
        }
    }
    
    return plt, results

def print_period_statistics(stats, metric_name, unit=""):
    print(f"\n{metric_name} Statistics:")
    print("-" * 50)
    
    # Print all-time statistics
    print(f"\nAll-time ({stats['all_time']['period']}):")
    print(f"Mean: {stats['all_time']['mean']:.2f}{unit}")
    print(f"Min:  {stats['all_time']['min']:.2f}{unit}")
    print(f"Max:  {stats['all_time']['max']:.2f}{unit}")
    print(f"Std:  {stats['all_time']['std']:.2f}{unit}")
    
    # Print statistics for recent periods
    for period in [50, 20, 10, 5]:
        period_key = f'last_{period}_years'
        if period_key in stats:
            print(f"\nLast {period} years ({stats[period_key]['period']}):")
            print(f"Mean: {stats[period_key]['mean']:.2f}{unit}")
            print(f"Min:  {stats[period_key]['min']:.2f}{unit}")
            print(f"Max:  {stats[period_key]['max']:.2f}{unit}")
            print(f"Std:  {stats[period_key]['std']:.2f}{unit}")

def main():
    # Define your CSV file paths
    temperature_csv = "tempArchive/GlobalLandTemperaturesByCountry.csv"
    emissions_csv = "GHArchive/2- annual-co-emissions-by-region.csv"
    
    # Load and prepare the data
    print("Loading data...")
    temp_df, emissions_df = load_and_prepare_data(temperature_csv, emissions_csv)
    
    # Prepare the analyzed data
    print("Preparing data for analysis...")
    yearly_temp = prepare_temperature_data(temp_df)
    yearly_emissions = prepare_emissions_data(emissions_df)
    
    # Create and save the plots
    print("Generating plots...")
    
    # Temperature plot
    print("Creating temperature plot...")
    temp_plot = create_temperature_plot(yearly_temp)
    temp_plot.savefig('./charts/temperature_trends_by_continent.png', bbox_inches='tight', dpi=300)
    temp_plot.close()
    
    # Emissions plot
    print("Creating emissions plot...")
    emissions_plot = create_emissions_plot(yearly_emissions)
    emissions_plot.savefig('./charts/emissions_trends.png', bbox_inches='tight', dpi=300)
    emissions_plot.close()
    
    # Combined plot
    print("Creating combined plot...")
    combined_plot = create_combined_plot(yearly_temp, yearly_emissions)
    combined_plot.savefig('./charts/temperature_and_emissions_by_continent.png', bbox_inches='tight', dpi=300)
    combined_plot.close()

    # Future projections plot
    print("Creating future projections plot...")
    projections_plot = create_future_projections_plot(yearly_temp, yearly_emissions)
    projections_plot.savefig('./charts/future_projections.png', bbox_inches='tight', dpi=300)
    projections_plot.close()
    
    # Basic regression analysis
    print("Performing basic regression analysis...")
    regression_plot, regression_results = perform_regression_analysis(yearly_temp, yearly_emissions)
    regression_plot.savefig('./charts/regression_analysis.png', bbox_inches='tight', dpi=300)
    regression_plot.close()
    
    # Enhanced regression analysis
    print("Performing enhanced regression analysis...")
    enhanced_plot, enhanced_results = perform_enhanced_regression_analysis(yearly_temp, yearly_emissions)
    enhanced_plot.savefig('./charts/enhanced_regression_analysis.png', bbox_inches='tight', dpi=300)
    enhanced_plot.close()
    
    # Print detailed analysis results
    print("\n============== REGRESSION ANALYSIS RESULTS ==============")
    
    # Temperature regression results
    print("\nTEMPERATURE REGRESSION ANALYSIS:")
    print("-" * 50)
    temp_model = enhanced_results['temperature']['model']
    print(f"Temperature = {temp_model.coef_[0]:.4f} ⋅ Year + {temp_model.intercept_:.2f}")
    print(f"R² Score: {enhanced_results['temperature']['r2']:.4f}")
    print(f"RMSE: {enhanced_results['temperature']['rmse']:.4f}°C")
    
    # Print temperature statistics
    print_period_statistics(enhanced_results['temperature']['statistics'], 
                          "Temperature", "°C")
    
    # Emissions regression results
    print("\nEMISSIONS REGRESSION ANALYSIS:")
    print("-" * 50)
    emissions_model = enhanced_results['emissions']['model']
    print(f"Emissions = {emissions_model.coef_[0]:.4f} ⋅ Year + {emissions_model.intercept_:.2f}")
    print(f"R² Score: {enhanced_results['emissions']['r2']:.4f}")
    print(f"RMSE: {enhanced_results['emissions']['rmse']:.4f} metric tons")
    
    # Print emissions statistics
    print_period_statistics(enhanced_results['emissions']['statistics'], 
                          "CO₂ Emissions", " metric tons")
    
    # Print simplified formulas
    print("\nFormulas in simplified notation:")
    print(f"Temperature = {temp_model.coef_[0]:.4f} ⋅ Year + {temp_model.intercept_:.2f}")
    print(f"Emissions = {emissions_model.coef_[0]:.4f} ⋅ Year + {emissions_model.intercept_:.2f}")
    
    print("\nAnalysis complete! Check the generated PNG files in charts directory.")

if __name__ == "__main__":
    main()
