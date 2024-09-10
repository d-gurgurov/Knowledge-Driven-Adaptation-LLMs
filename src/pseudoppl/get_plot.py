import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_data(file_path):
    # Read the CSV file into a pandas DataFrame
    return pd.read_csv(file_path)

def plot_comparison(extra_model=None, output_file='conceptnet_pppl_comparison.png', title="Model Comparison on CN Dataset", **data_files):
    # Read and merge all the data files provided as arguments
    merged_data = None
    for model_name, file_path in data_files.items():
        data = read_data(file_path)
        data = data.rename(columns={"average_pseudo_perplexity": f"average_pseudo_perplexity_{model_name}"})
        
        if merged_data is None:
            merged_data = data
        else:
            merged_data = pd.merge(merged_data, data, on='language')
    
    # Sort the merged data by the first model's average pseudo-perplexity in descending order
    first_model = list(data_files.keys())[0]
    merged_data = merged_data.sort_values(by=f'average_pseudo_perplexity_{first_model}', ascending=False)  # type: ignore
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    
    for model_name in data_files.keys():
        line_style = '--' if model_name == extra_model else '-'  # Dashed line for base model
        color = 'lightgray' if model_name == extra_model else None  # Light gray color for base model
        
        plt.plot(merged_data['language'], 
                 merged_data[f'average_pseudo_perplexity_{model_name}'], 
                 label=model_name, 
                 marker='o', 
                 linestyle=line_style, 
                 color=color)
    
    plt.xlabel('Language Code')
    plt.ylabel('Average Pseudo-Perplexity')
    plt.title(title)
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.legend()
    plt.grid(True)
    
    # Adjust y-axis limits for better visibility
    min_value = merged_data[[f'average_pseudo_perplexity_{model_name}' for model_name in data_files.keys()]].min().min()
    max_value = merged_data[[f'average_pseudo_perplexity_{model_name}' for model_name in data_files.keys()]].max().max()
    max_value = 500

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

    # Define the paths to the files for each model
    model_files = {
        "mBERT": '/netscratch/dgurgurov/thesis/results/conceptnet/mbert/baseline/average_pseudo_perplexities_summary.csv',
        "mBERT_LoRA": '/netscratch/dgurgurov/thesis/results/conceptnet/mbert/lora/average_pseudo_perplexities_summary.csv',
        "mBERT_Seq_bn": '/netscratch/dgurgurov/thesis/results/conceptnet/mbert/seq_bn/average_pseudo_perplexities_summary.csv',
        "mBERT_Seq_bn_inv": '/netscratch/dgurgurov/thesis/results/conceptnet/mbert/seq_bn_inv/average_pseudo_perplexities_summary.csv',
        "XLM-R": '/netscratch/dgurgurov/thesis/results/conceptnet/xlm-r/baseline/average_pseudo_perplexities_summary.csv',
        "XLM-R_LoRA": '/netscratch/dgurgurov/thesis/results/conceptnet/xlm-r/lora/average_pseudo_perplexities_summary.csv',
        "XLM-R_Seq_bn": '/netscratch/dgurgurov/thesis/results/conceptnet/xlm-r/seq_bn/average_pseudo_perplexities_summary.csv',
        "XLM-R_Seq_bn_inv": '/netscratch/dgurgurov/thesis/results/conceptnet/xlm-r/seq_bn_inv/average_pseudo_perplexities_summary.csv'
    }
    
    # Plot the comparison
    plot_comparison(extra_model="mBERT", **model_files)
