import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Read the datasets
def load_and_prepare_data(temp_csv_path, emissions_csv_path):
    # Read temperature data
    temp_df = pd.read_csv(temp_csv_path)
    # Convert date string to datetime
    temp_df['dt'] = pd.to_datetime(temp_df['dt'])
    # Extract year from datetime
    temp_df['Year'] = temp_df['dt'].dt.year
    
    # Read emissions data
    emissions_df = pd.read_csv(emissions_csv_path)
    
    return temp_df, emissions_df

def prepare_temperature_data(temp_df):
    # Group by Year and Country, calculate mean temperature
    yearly_temp = temp_df.groupby(['Year', 'Country'])['AverageTemperature'].mean().reset_index()
    return yearly_temp

def prepare_emissions_data(emissions_df):
    # Group by Year, sum all countries' emissions
    yearly_emissions = emissions_df.groupby('Year')['Annual CO₂ emissions'].sum().reset_index()
    return yearly_emissions

def create_temperature_plot(yearly_temp):
    plt.figure(figsize=(15, 8))
    
    # Create line plot for each country
    for country in yearly_temp['Country'].unique():
        country_data = yearly_temp[yearly_temp['Country'] == country]
        plt.plot(country_data['Year'], country_data['AverageTemperature'], 
                label=country, alpha=0.5)
    
    plt.title('Average Temperature by Country Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Temperature (°C)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    return plt

def create_emissions_plot(yearly_emissions):
    plt.figure(figsize=(15, 8))
    
    plt.plot(yearly_emissions['Year'], yearly_emissions['Annual CO₂ emissions'], 
            color='red', linewidth=2)
    
    plt.title('Global CO₂ Emissions Over Time')
    plt.xlabel('Year')
    plt.ylabel('Annual CO₂ Emissions')
    plt.grid(True, alpha=0.3)
    
    return plt

def create_combined_plot(yearly_temp, yearly_emissions):
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Temperature lines on primary y-axis
    for country in yearly_temp['Country'].unique():
        country_data = yearly_temp[yearly_temp['Country'] == country]
        ax1.plot(country_data['Year'], country_data['AverageTemperature'], 
                label=country, alpha=0.5)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Temperature (°C)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Emissions line on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(yearly_emissions['Year'], yearly_emissions['Annual CO₂ emissions'], 
            color='red', linewidth=2, label='CO₂ Emissions')
    ax2.set_ylabel('Annual CO₂ Emissions', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Temperature and CO₂ Emissions Over Time')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.15, 1))
    
    return plt

# Example usage:
"""
# Load your data
temp_df, emissions_df = load_and_prepare_data('temperature_data.csv', 'emissions_data.csv')

# Prepare the data
yearly_temp = prepare_temperature_data(temp_df)
yearly_emissions = prepare_emissions_data(emissions_df)

# Create individual plots
temp_plot = create_temperature_plot(yearly_temp)
temp_plot.savefig('temperature_plot.png', bbox_inches='tight')
temp_plot.close()

emissions_plot = create_emissions_plot(yearly_emissions)
emissions_plot.savefig('emissions_plot.png', bbox_inches='tight')
emissions_plot.close()

# Create combined plot
combined_plot = create_combined_plot(yearly_temp, yearly_emissions)
combined_plot.savefig('combined_plot.png', bbox_inches='tight')
combined_plot.close()
"""