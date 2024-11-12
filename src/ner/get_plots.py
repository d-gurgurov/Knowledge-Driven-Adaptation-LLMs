import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the paths to the result files
data_files = {
        "mBERT_ConceptNet_Baseline": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/conceptnet_mbert_baseline_results.csv',
        "mBERT_ConceptNet_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/conceptnet_mbert_seq_bn_results.csv',
        "mBERT_ConceptNet_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/conceptnet_mbert_lora_results.csv',
        "mBERT_ConceptNet_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/conceptnet_mbert_seq_bn_inv_results.csv',
        
        "XLM-R_ConceptNet_Baseline": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/glot_xlm-r_baseline_results.csv',
        "XLM-R_ConceptNet_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/conceptnet_xlm-r_seq_bn_results.csv',
        "XLM-R_ConceptNet_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/conceptnet_xlm-r_lora_results.csv',
        "XLM-R_ConceptNet_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/conceptnet_xlm-r_seq_bn_inv_results.csv',
    
        "mBERT_Glot_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/glot_mbert_seq_bn_results.csv',
        "mBERT_Glot_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/glot_mbert_lora_results.csv',
        "mBERT_Glot_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/glot_mbert_seq_bn_inv_results.csv',
        
        "XLM-R_Glot_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/glot_xlm-r_seq_bn_results.csv',
        "XLM-R_Glot_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/glot_xlm-r_lora_results.csv',
        "XLM-R_Glot_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/glot_xlm-r_seq_bn_inv_results.csv'
}

# Load language size data (class column)
lang_size_data = pd.read_csv('/netscratch/dgurgurov/thesis/src/language_data_sizes.csv')

# Function to load and format the NER results from the csv files
def load_ner_results(data_files):
    results = {}
    for name, file_path in data_files.items():
        df = pd.read_csv(file_path)
        df = df[['Language', 'Average F1']]  # Only keep the relevant columns
        df['Model Configuration'] = name  # Add a column for the model configuration
        
        # Merge with the language class information
        df = df.merge(lang_size_data[['language_code', 'class']], left_on='Language', right_on='language_code', how='left')
        
        # Ensure the merge was successful (i.e., no NaN values in 'class')
        if df['class'].isnull().any():
            print(f"Warning: Missing 'class' values for some languages in {name}")
        
        # Sort by the 'class' column
        df = df.sort_values(by='class')  # Sort by the 'class' column
        
        results[name] = df
    return results

# Load the data and apply sorting
ner_results = load_ner_results(data_files)

# Combine the results into a single DataFrame
combined_results = pd.concat(ner_results.values(), ignore_index=True)

# Separate the results for mBERT and XLM-R
mbert_results = combined_results[combined_results['Model Configuration'].str.contains('mBERT')]
xlmr_results = combined_results[combined_results['Model Configuration'].str.contains('XLM-R')]

# Excluded languages for mBERT and XLM-R models
exclude_languages = {
        "mBERT": ["sin_Sinh", "mlt_Latn", "uig_Arab", "amh_Ethi", "bod_Tibt"],
        "XLM-R": ["ckb_Arab", "yor_Latn", "mlt_Latn", "bod_Tibt"]
    }

# Function to plot the results for mBERT or XLM-R
def plot_ner_results(results, title, save_path):
    model_name = title.split()[0]  # Extract model name (mBERT or XLM-R)
    excluded_langs = exclude_languages.get(model_name, [])
    
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")
    
    # Plotting bars for different model configurations for each language
    sns.barplot(data=results, x='Language', y='Average F1', hue='Model Configuration', alpha=0.7, errorbar=None)

    # Add asterisks to the x-axis labels for excluded languages for the current model
    ax = plt.gca()

    # Get the current x-axis labels
    current_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    
    # Modify the labels to append asterisks for the excluded languages
    modified_labels = [
        f"{label}*" if label in excluded_langs else label
        for label in current_labels
    ]
    
    # Set the modified labels back to the x-axis
    ax.set_xticklabels(modified_labels, rotation=45, ha='center', fontsize=8)

    # Change the color of the asterisks in the labels
    for tick in ax.get_xticklabels():
        label = tick.get_text()
        if '*' in label:
            # Split the label into the language part and the asterisk
            language_part, asterisk = label.split('*')
            # Redraw the label with the asterisk in a different color
            tick.set_text(language_part)
            tick.set_color('black')  # Set the color of the language part
            tick.set_fontsize(8)  # Font size of the language part
            # Use a different color for the asterisk
            ax.text(tick.get_position()[0], tick.get_position()[1], '*', color='red', fontsize=12, ha='center')

    # Identify the indices of the boundary languages
    boundary_indices = [1, 3, 14]
    
    # Add vertical lines for class boundaries after the specific languages
    for idx in boundary_indices:
        plt.axvline(x=idx + 0.5, color='gray', linestyle='--', linewidth=1)  # Adjusting for the gap


    plt.title(title, fontsize=16)
    plt.xlabel('Language', fontsize=14)
    plt.ylabel('Average F1 Score', fontsize=14)
    plt.ylim(0.1, None)
    plt.xticks(rotation=45, fontsize=8, ha='center')
    plt.legend(title='Model Configuration', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# Plot for mBERT
plot_ner_results(mbert_results, "mBERT NER Performance", './plots/mBERT_ner_comparison_bars.png')

# Plot for XLM-R
plot_ner_results(xlmr_results, "XLM-R NER Performance", './plots/XLM-R_ner_comparison_bars.png')
