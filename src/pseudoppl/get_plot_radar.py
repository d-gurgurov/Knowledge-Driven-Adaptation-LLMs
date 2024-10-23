import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_data(file_path):
    # Read the CSV file into a pandas DataFrame
    return pd.read_csv(file_path)

def plot_radar_charts(output_file='radar_comparison_dual.png', title="Model Comparison on conceptnet Dataset", **data_files):
    # Read and merge all the data files provided as arguments
    merged_data = None
    for model_name, file_path in data_files.items():
        data = read_data(file_path)
        data = data.rename(columns={"average_pseudo_perplexity": f"average_pseudo_perplexity_{model_name}"})
        
        if merged_data is None:
            merged_data = data
        else:
            merged_data = pd.merge(merged_data, data, on='language')
    
    # Define the number of variables (languages)
    languages = merged_data['language'].values
    num_vars = len(languages)
    
    # Compute angles for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Ensure the plot closes

    # Make the figure wider by adjusting figsize
    fig, axs = plt.subplots(1, 2, figsize=(18, 8), subplot_kw=dict(polar=True))  # Two plots side by side
    
    # Define color map and line styles for mBERT and XLM-R configurations
    color_map_mbert = {
        "mBERT": "blue",
        "mBERT_LoRA": "cyan",
        "mBERT_Seq_bn": "green",
        "mBERT_Seq_bn_inv": "lime"
    }
    color_map_xlmr = {
        "XLM-R": "red",
        "XLM-R_LoRA": "magenta",
        "XLM-R_Seq_bn": "orange",
        "XLM-R_Seq_bn_inv": "purple"
    }
    line_styles = {
        "mBERT": "-",
        "mBERT_LoRA": "--",
        "mBERT_Seq_bn": "-.",
        "mBERT_Seq_bn_inv": ":",
        "XLM-R": "-",
        "XLM-R_LoRA": "--",
        "XLM-R_Seq_bn": "-.",
        "XLM-R_Seq_bn_inv": ":"
    }

    # Normalize values for better comparison
    normalized_data = merged_data.copy()
    for model_name in color_map_mbert.keys():
        max_value = merged_data[f'average_pseudo_perplexity_{model_name}'].max()
        normalized_data[f'average_pseudo_perplexity_{model_name}'] = merged_data[f'average_pseudo_perplexity_{model_name}'] / max_value
    
    for model_name in color_map_xlmr.keys():
        max_value = merged_data[f'average_pseudo_perplexity_{model_name}'].max()
        normalized_data[f'average_pseudo_perplexity_{model_name}'] = merged_data[f'average_pseudo_perplexity_{model_name}'] / max_value

    # First radar chart for mBERT
    ax_mbert = axs[0]
    for model_name in color_map_mbert.keys():
        values = normalized_data[f'average_pseudo_perplexity_{model_name}'].values.tolist()
        values += values[:1]  # Ensure the plot closes

        ax_mbert.plot(angles, values, label=model_name, 
                      color=color_map_mbert[model_name],  # Apply unique color
                      linestyle=line_styles[model_name],  # Apply unique line style
                      linewidth=2)  # Thicker lines for better visibility
        ax_mbert.fill(angles, values, color=color_map_mbert[model_name], alpha=0.2)  # Fill with color and transparency
    
    ax_mbert.set_xticks(angles[:-1])
    ax_mbert.set_xticklabels(languages, fontsize=10)
    ax_mbert.set_title("mBERT Results (Normalized)", size=15, color='black', y=1.1)
    ax_mbert.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Second radar chart for XLM-R
    ax_xlmr = axs[1]
    for model_name in color_map_xlmr.keys():
        values = normalized_data[f'average_pseudo_perplexity_{model_name}'].values.tolist()
        values += values[:1]  # Ensure the plot closes

        ax_xlmr.plot(angles, values, label=model_name, 
                     color=color_map_xlmr[model_name],  # Apply unique color
                     linestyle=line_styles[model_name],  # Apply unique line style
                     linewidth=2)  # Thicker lines for better visibility
        ax_xlmr.fill(angles, values, color=color_map_xlmr[model_name], alpha=0.2)  # Fill with color and transparency
    
    ax_xlmr.set_xticks(angles[:-1])
    ax_xlmr.set_xticklabels(languages, fontsize=10)
    ax_xlmr.set_title("XLM-R Results (Normalized)", size=15, color='black', y=1.1)
    ax_xlmr.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Set radial limits for each chart
    ax_mbert.set_ylim(0, 1)  # Normalize between 0 and 1
    ax_xlmr.set_ylim(0, 1)    # Normalize between 0 and 1

    # Save the plot
    plt.savefig(output_file, format='png', dpi=300)
    print(f"Radar charts saved as {output_file}")
    
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
    
    # Plot the radar charts
    plot_radar_charts(**model_files)
