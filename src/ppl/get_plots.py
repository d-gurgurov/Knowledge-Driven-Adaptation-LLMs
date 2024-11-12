import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Load language size data
lang_size_data = pd.read_csv('/netscratch/dgurgurov/thesis/src/language_data_sizes.csv')

# Load perplexity data for CN and Glot models
glot_files = {
    "mBERT_Glot_Baseline": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/mbert/flores/baseline/average_perplexities.csv',
    "mBERT_Glot_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/mbert/flores/seq_bn/average_perplexities.csv',
    "mBERT_Glot_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/mbert/flores/lora/average_perplexities.csv',
    "mBERT_Glot_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/mbert/flores/seq_bn_inv/average_perplexities.csv',
    "XLM-R_Glot_Baseline": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/xlm-r/flores/baseline/average_perplexities.csv',
    "XLM-R_Glot_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/xlm-r/flores/seq_bn/average_perplexities.csv',
    "XLM-R_Glot_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/xlm-r/flores/lora/average_perplexities.csv',
    "XLM-R_Glot_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/xlm-r/flores/seq_bn_inv/average_perplexities.csv'
}

conceptnet_files = {
    "mBERT_ConceptNet_Baseline": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/mbert/flores/baseline/average_perplexities.csv',
    "mBERT_ConceptNet_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/mbert/cn/seq_bn/average_perplexities.csv',
    "mBERT_ConceptNet_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/mbert/cn/lora/average_perplexities.csv',
    "mBERT_ConceptNet_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/mbert/cn/seq_bn_inv/average_perplexities.csv',
    "XLM-R_ConceptNet_Baseline": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/xlm-r/flores/baseline/average_perplexities.csv',
    "XLM-R_ConceptNet_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/xlm-r/cn/seq_bn/average_perplexities.csv',
    "XLM-R_ConceptNet_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/xlm-r/cn/lora/average_perplexities.csv',
    "XLM-R_ConceptNet_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/ppl/xlm-r/cn/seq_bn_inv/average_perplexities.csv'
}

def load_ppl_data(file_dict):
    """Load perplexity data from multiple CSV files into a dictionary of DataFrames."""
    ppl_data = {}
    for name, file_path in file_dict.items():
        df = pd.read_csv(file_path)
        ppl_data[name] = df[['language_code', 'perplexity']]
    return ppl_data

# Load perplexity data for CN and Glot
cn_ppl_data = load_ppl_data(conceptnet_files)
glot_ppl_data = load_ppl_data(glot_files)

def plot_perplexity_comparison(ppl_data, lang_size_data, size_column, dataset_name, save_dir):
    sns.set_style("whitegrid")
    os.makedirs(save_dir, exist_ok=True)
    exclude_languages = {
        "mBERT": ["sin_Sinh", "mlt_Latn", "uig_Arab", "amh_Ethi", "bod_Tibt"],
        "XLM-R": ["ckb_Arab", "yor_Latn", "mlt_Latn", "bod_Tibt"]
    }
    
    # Set a custom color palette with good contrast
    colors = sns.color_palette("husl", n_colors=4)
    
    sorted_lang_size = lang_size_data.sort_values(by=size_column).reset_index(drop=True)
    
    model_groups = {}
    for name, df in ppl_data.items():
        model_key = name.split('_', 1)[0]
        model_groups.setdefault(model_key, []).append((name, df))
    
    for model_key, configurations in model_groups.items():
        plt.figure(figsize=(15, 10))  # Increased figure size for better readability
        excluded_langs = exclude_languages.get(model_key, [])
        
        # Calculate bar positions
        n_configs = len(configurations)
        bar_width = 0.8 / n_configs
        
        for idx, (name, df) in enumerate(configurations):
            # Sort by language class
            df = df.merge(sorted_lang_size[['language_code', "class"]], on='language_code')
            df = df.sort_values(by="class")
            
            # Calculate bar positions
            x_positions = np.arange(len(df)) + (idx - n_configs/2 + 0.5) * bar_width
            
            # Create bars with error handling for log scale
            plt.bar(x_positions, 
                   df['perplexity'],
                   width=bar_width,
                   label=name,
                   color=colors[idx],
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=1)
            
            # Add error bars if standard deviation is available
            if 'perplexity_std' in df.columns:
                plt.errorbar(x_positions,
                           df['perplexity'],
                           yerr=df['perplexity_std'],
                           fmt='none',
                           color='black',
                           capsize=3,
                           capthick=1,
                           alpha=0.5)
            
            # Annotate excluded languages
            for i, (code, ppl) in enumerate(zip(df['language_code'], df['perplexity'])):
                if code in excluded_langs:
                    plt.text(x_positions[i], ppl * 1.1,
                           "*", color='red', fontsize=12, ha='center')
        
        # Set x-tick labels with asterisks for excluded languages
        tick_labels = [
            f"{code}*" if code in excluded_langs else code
            for code in df['language_code']
        ]
        
        plt.xticks(ticks=range(len(df)), labels=tick_labels, rotation=45, fontsize=10, ha='right')
        plt.yscale("log")
        
        # Improve grid appearance
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Enhance labels and title
        plt.xlabel("Language Code", fontsize=14, labelpad=10)
        plt.ylabel("Perplexity Score (log scale)", fontsize=14, labelpad=10)
        plt.title(f"{model_key} Perplexity Comparison by Language\n{dataset_name}",
                 fontsize=16, weight='bold', pad=20)
        
        # Enhance legend
        plt.legend(title="Model Configuration",
                  title_fontsize=12,
                  fontsize=10,
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left')
        
        # Add a text box explaining the asterisk
        plt.figtext(0.99, 0.02,
                   "* Excluded from main analysis",
                   fontsize=10,
                   ha='right',
                   style='italic')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f"{model_key}_{dataset_name}_perplexity_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# Generate plots for ConceptNet and Glot
plot_perplexity_comparison(cn_ppl_data, lang_size_data, 'CN_sents', "ConceptNet", './plots/conceptnet')
plot_perplexity_comparison(glot_ppl_data, lang_size_data, 'Glot_docs', "Glot", './plots/glot')
