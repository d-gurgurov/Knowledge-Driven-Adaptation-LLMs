import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_data(file_path):
    # Read the CSV file into a pandas DataFrame
    return pd.read_csv(file_path)

def plot_comparison(data1, data2, output_file='flores_pppl_1.png', title="Model Comparison on FLORES Dataset"):
    # Merge the two dataframes on the language code
    merged_data = pd.merge(data1, data2, on='language_code', suffixes=('_model1', '_model2'))

    # Sort the merged data by the first model's average pseudo-perplexity in descending order
    merged_data = merged_data.sort_values(by='average_pseudo_perplexity_model1', ascending=False)
    
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    
    plt.plot(merged_data['language_code'], merged_data['average_pseudo_perplexity_model1'], label='mBERT', marker='o')
    plt.plot(merged_data['language_code'], merged_data['average_pseudo_perplexity_model2'], label='XLM-R', marker='o')
    
    plt.xlabel('Language Code')
    plt.ylabel('Average Pseudo-Perplexity')
    plt.title(title)
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.legend()
    plt.grid(True)
    
    # Adjust y-axis limits for better visibility
    min_value = min(merged_data['average_pseudo_perplexity_model1'].min(), merged_data['average_pseudo_perplexity_model2'].min())
    max_value = max(merged_data['average_pseudo_perplexity_model1'].max(), merged_data['average_pseudo_perplexity_model2'].max())
    max_value = 200

    # Add some padding to the y-axis
    padding = (max_value - min_value) * 0.1
    plt.ylim(min_value - padding, max_value + padding)
    
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig(output_file, format='png', dpi=300)
    print(f"Plot saved as {output_file}")
    
    # Show the plot (optional)
    plt.show()

if __name__ == "__main__":
    # Define the paths to the two files
    file1 = '/netscratch/dgurgurov/thesis/results/flores/mbert/average_pseudo_perplexities_summary.csv'
    file2 = '/netscratch/dgurgurov/thesis/results/flores/xlm-r/average_pseudo_perplexities_summary.csv'
    
    # Read the data
    data1 = read_data(file1)
    data2 = read_data(file2)
    
    # Plot the comparison
    plot_comparison(data1, data2)
