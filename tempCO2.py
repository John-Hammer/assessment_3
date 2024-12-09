import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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
    
    plt.title('Average Temperature by Continent Over Time')
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
    
    # Temperature lines on primary y-axis
    for continent in yearly_temp['Continent'].unique():
        continent_data = yearly_temp[yearly_temp['Continent'] == continent]
        if continent == 'Global Average':
            ax1.plot(continent_data['Year'], continent_data['AverageTemperature'],
                    label=continent, color=colors[continent], linewidth=2)
        else:
            ax1.plot(continent_data['Year'], continent_data['AverageTemperature'],
                    label=continent, color=colors[continent], alpha=0.5)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average Temperature (°C)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Emissions line on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(yearly_emissions['Year'], yearly_emissions['Annual CO₂ emissions'],
            color='red', linewidth=2, label='CO₂ Emissions')
    ax2.set_ylabel('Annual CO₂ Emissions', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Temperature by Continent and CO₂ Emissions Over Time')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.15, 1))
    
    return plt

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
    
    print("Analysis complete! Check the generated PNG files in charts directory.")

if __name__ == "__main__":
    main()