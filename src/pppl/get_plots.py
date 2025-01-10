import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Load language size data
lang_size_data = pd.read_csv('/netscratch/dgurgurov/thesis/src/language_data_sizes.csv')

# Load perplexity data for CN and Glot models
glot_files = {
    "mBERT_Glot_Baseline": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/flores/mbert/baseline/average_pseudo_perplexities_summary.csv',
    "mBERT_Glot_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/flores/mbert/seq_bn/average_pseudo_perplexities_summary.csv',
    "mBERT_Glot_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/flores/mbert/lora/average_pseudo_perplexities_summary.csv',
    "mBERT_Glot_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/flores/mbert/seq_bn_inv/average_pseudo_perplexities_summary.csv',
    "XLM-R_Glot_Baseline": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/flores/xlm-r/baseline/average_pseudo_perplexities_summary.csv',
    "XLM-R_Glot_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/flores/xlm-r/seq_bn/average_pseudo_perplexities_summary.csv',
    "XLM-R_Glot_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/flores/xlm-r/lora/average_pseudo_perplexities_summary.csv',
    "XLM-R_Glot_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/flores/xlm-r/seq_bn_inv/average_pseudo_perplexities_summary.csv'
}

conceptnet_files = {
    "mBERT_ConceptNet_Baseline": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/flores/mbert/baseline/average_pseudo_perplexities_summary.csv',
    "mBERT_ConceptNet_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/conceptnet/mbert/seq_bn/average_pseudo_perplexities_summary.csv',
    "mBERT_ConceptNet_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/conceptnet/mbert/lora/average_pseudo_perplexities_summary.csv',
    "mBERT_ConceptNet_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/conceptnet/mbert/seq_bn_inv/average_pseudo_perplexities_summary.csv',
    "XLM-R_ConceptNet_Baseline": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/flores/xlm-r/baseline/average_pseudo_perplexities_summary.csv',
    "XLM-R_ConceptNet_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/conceptnet/xlm-r/seq_bn/average_pseudo_perplexities_summary.csv',
    "XLM-R_ConceptNet_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/conceptnet/xlm-r/lora/average_pseudo_perplexities_summary.csv',
    "XLM-R_ConceptNet_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/pppl/conceptnet/xlm-r/seq_bn_inv/average_pseudo_perplexities_summary.csv'
}

def load_pppl_data(file_dict):
    """Load perplexity data from multiple CSV files into a dictionary of DataFrames."""
    pppl_data = {}
    for name, file_path in file_dict.items():
        df = pd.read_csv(file_path)
        pppl_data[name] = df[['language', 'average_pseudo_perplexity']]
    return pppl_data

# Load perplexity data for CN and Glot
conceptnet_pppl_data = load_pppl_data(conceptnet_files)
glot_pppl_data = load_pppl_data(glot_files)



def plot_perplexity_comparison(pppl_data, lang_size_data, size_column, dataset_name, save_dir):
    sns.set(style="whitegrid")
    os.makedirs(save_dir, exist_ok=True)

    exclude_languages = {
        "mBERT": ["si", "mt", "ug", "am", "bo", "ku"],
        "XLM-R": ["yo", "mt", "bo"]
    }

    # Set a custom color palette with good contrast
    colors = sns.color_palette("husl", n_colors=4)
    
    sorted_lang_size = lang_size_data.sort_values(by=size_column).reset_index(drop=True)
    
    model_groups = {}
    for name, df in pppl_data.items():
        model_key = name.split('_', 1)[0]
        model_groups.setdefault(model_key, []).append((name, df))
    
    for model_key, configurations in model_groups.items():
        plt.figure(figsize=(15, 10))
        excluded_langs = exclude_languages.get(model_key, [])
        
        # Calculate bar positions
        n_configs = len(configurations)
        bar_width = 0.8 / n_configs
        
        for idx, (name, df) in enumerate(configurations):
            # Merge with language class data
            df = df.merge(sorted_lang_size[['language', "class"]], on='language')
            
            # Create a boolean mask for excluded languages
            is_excluded = df['language'].isin(excluded_langs)
            
            # Sort the DataFrame: excluded languages first, then others by class
            df['sort_key'] = (~is_excluded).astype(int)  # 0 for excluded, 1 for others
            df = df.sort_values(['sort_key', 'class']).reset_index(drop=True)
            
            # Calculate bar positions
            x_positions = np.arange(len(df)) + (idx - n_configs/2 + 0.5) * bar_width
            
            # Create bars with error handling for log scale
            plt.bar(x_positions, 
                   df['average_pseudo_perplexity'],
                   width=bar_width,
                   label=name,
                   color=colors[idx],
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=1)
        
        # Set x-tick labels with asterisks for excluded languages
        tick_labels = [
            f"{code}*" if code in excluded_langs else code
            for code in df['language']
        ]
        
        plt.xticks(ticks=range(len(df)), labels=tick_labels, rotation=45, fontsize=10, ha='right')
        plt.yscale("log")
        
        # Improve grid appearance
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Enhance labels and title
        plt.xlabel("Language Code", fontsize=14, labelpad=10)
        plt.ylabel("Pseudo-perplexity Score (log scale)", fontsize=14, labelpad=10)
        plt.title(f"{model_key} Pseudo-perplexity Comparison by Language\n{dataset_name}",
                 fontsize=16, weight='bold', pad=20)
        
        # Enhance legend
        plt.legend(title="Model Configuration",
                  title_fontsize=12,
                  fontsize=10,
                  # bbox_to_anchor=(1.05, 1),
                  loc='upper right')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f"{model_key}_{dataset_name}_perplexity_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# Generate plots for ConceptNet and Glot
plot_perplexity_comparison(conceptnet_pppl_data, lang_size_data, 'CN_sents', "ConceptNet", './plots/conceptnet')
plot_perplexity_comparison(glot_pppl_data, lang_size_data, 'Glot_docs', "Glot", './plots/glot')
