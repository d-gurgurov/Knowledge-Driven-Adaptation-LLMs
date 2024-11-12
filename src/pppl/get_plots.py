import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
        "mBERT": ["si", "mt", "ug", "am", "bo"],
        "XLM-R": ["ku", "yo", "mt", "bo"]
    }

    sorted_lang_size = lang_size_data.sort_values(by=size_column).reset_index(drop=True)
    
    model_groups = {}
    for name, df in pppl_data.items():

        model_key = name.split('_', 1)[0]
        model_groups.setdefault(model_key, []).append((name, df))
    
    for model_key, configurations in model_groups.items():
        plt.figure(figsize=(12, 8))

        excluded_langs = exclude_languages.get(model_key, [])
        
        for name, df in configurations:
            # Sort based on language data counts
            # df = df[~df['language_code'].isin(excluded_langs)]
            # df = df.merge(sorted_lang_size[['language_code', size_column]], on='language_code')
            # df = df.sort_values(by=size_column)

            # Sort based on language clss
            df = df.merge(sorted_lang_size[['language', "class"]], on='language')
            df = df.sort_values(by="class")
            x_values = range(len(df))

            # Plot perplexity for each adapter
            sns.lineplot(x=range(len(df)), y=df['average_pseudo_perplexity'], marker='o', label=name)

            # Add asterisks above the highest point for each excluded language
            for i, code in enumerate(df['language']):
                if code in excluded_langs:
                    # Get the highest perplexity value for this excluded language
                    highest_point = df.loc[df['language'] == code, 'average_pseudo_perplexity'].max()
                    # Annotate with an asterisk above this point
                    plt.text(
                        x=i, 
                        y=highest_point * 1.05,  # Position slightly above the highest point
                        s="*", 
                        color='red', 
                        fontsize=12, 
                        ha='center'
                    )

        # Calculate the x-axis positions where class boundaries change
        class_boundaries = []
        last_class = df['class'].iloc[0]
        for i, current_class in enumerate(df['class']):
            if current_class != last_class:
                class_boundaries.append(i - 0.5)  # Position boundary between classes
                last_class = current_class

        # Add vertical lines at class boundaries
        for boundary in class_boundaries:
            plt.axvline(x=boundary, color='gray', linestyle='--', linewidth=1)

        # Prepare x-tick labels with asterisks for excluded languages
        tick_labels = [
            f"{code}*" if code in excluded_langs else code
            for code in df['language']
        ]
        
        # Set language codes as x-axis labels at respective x positions
        plt.xticks(ticks=x_values, labels=tick_labels, rotation=45, fontsize=8, ha='center')
        plt.yscale("log")
        plt.xlabel("Language Code", fontsize=14)
        plt.ylabel("Perplexity Score", fontsize=14)
        plt.title(f"{model_key} Perplexity vs Language Size ({dataset_name})", fontsize=16, weight='bold')
        plt.legend(title="Model Configuration")
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        save_path = os.path.join(save_dir, f"{model_key}_{dataset_name}_perplexity_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# Generate plots for ConceptNet and Glot
plot_perplexity_comparison(conceptnet_pppl_data, lang_size_data, 'CN_sents', "ConceptNet", './plots/conceptnet')
plot_perplexity_comparison(glot_pppl_data, lang_size_data, 'Glot_docs', "Glot", './plots/glot')
