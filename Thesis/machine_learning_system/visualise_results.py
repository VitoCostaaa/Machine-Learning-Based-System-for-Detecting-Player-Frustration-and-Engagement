import pandas as pd                      # Library for handling tabular data
import seaborn as sns                   # Library for statistical data visualization
import matplotlib.pyplot as plt         # Library for creating plots
import os                               # Provides file and directory operations
import matplotlib.ticker as ticker      # Used for formatting axis ticks


def create_scientific_plots():
    
    # Path to the dataset
    file_path = 'data/player_data.csv'
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Datei {file_path} nicht gefunden.")
        return

    
    # Define column names (since CSV may not contain headers)
    column_names = [
        "player", "duration", "score", "clicks", "avg_interval", 
        "var_interval", "iti", "panic_rate", "death_altitude", "label"
    ]

    
    # Load dataset
    df = pd.read_csv(file_path, names=column_names, header=None)

    # Remove potential duplicate header row inside the data
    df = df[df['label'] != 'label']
    
    # Convert selected columns to numeric values
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['avg_interval'] = pd.to_numeric(df['avg_interval'], errors='coerce')
    df['panic_rate'] = pd.to_numeric(df['panic_rate'], errors='coerce')
    
    # Ensure output directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Set visualization style
    sns.set_theme(style="whitegrid", palette="muted") 


    # --- Plot 1: Boxplot for panic rate ---
    
    plt.figure(figsize=(10, 6))
    
    # Create boxplot comparing emotional states
    ax = sns.boxplot(
        x='player', 
        y='panic_rate', 
        hue='label', 
        data=df,
        palette={'engaged': '#2ecc71', 'frustrated': '#e74c3c'}
    )

    # Set plot labels and title
    plt.title('Behavioral Analysis: Panic rate in frustration vs. engagement')
    plt.ylabel('Clicks in the last 2 seconds')
    plt.xlabel('Test Player')
    
    # Save and display plot
    plt.savefig('plots/comparison_panic_rate.png')
    plt.show()
    plt.close()



    # --- Plot 2: Scatterplot for performance ---
    
    # Filter data for a specific player (e.g., Vito)
    player_df = df[df['player'] == "Vito"]

    plt.figure(figsize=(10, 6))

    # Create scatterplot showing relationship between click interval and score
    ax = sns.scatterplot(
    x='avg_interval', 
    y='score', 
    hue='label', 
    data=player_df, 
    s=100,
    palette={'engaged': '#2ecc71', 'frustrated': '#e74c3c'}
    
)
    
    # Adjust x-axis formatting for readability
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6)) 
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Set plot labels and title
    plt.title('Performance-Analysis: Click Interval vs. Score')
    plt.xlabel('Average Click Interval (in Seconds)')
    plt.ylabel('Score achieved')
    plt.tight_layout()

    # Save and display plot
    plt.savefig('plots/performance_analysis_Vito.png')
    plt.show()

    
    # Confirmation message
    print("All scientific plots have been saved in the ‘plots/’ folder.")

    


# Entry point of the script
if __name__ == "__main__":
    create_scientific_plots()