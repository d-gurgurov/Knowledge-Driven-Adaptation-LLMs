import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the languages mapping
languages_mapping = {
    "tel_Telu": "te", "ben_Beng": "bn", "lvs_Latn": "lv", "mlt_Latn": "mt", 
    "amh_Ethi": "am", "uzn_Latn": "uz", "sun_Latn": "su", "cym_Latn": "cy", 
    "mar_Deva": "mr", "ckb_Arab": "ku", "mkd_Cyrl": "mk", "kat_Geor": "ka", 
    "slk_Latn": "sk", "ell_Grek": "el", "tha_Thai": "th", "azj_Latn": "az", 
    "slv_Latn": "sl", "heb_Hebr": "he", "ron_Latn": "ro", "dan_Latn": "da", 
    "urd_Arab": "ur", "sin_Sinh": "si", "yor_Latn": "yo", "swh_Latn": "sw", 
    "uig_Arab": "ug", "bod_Tibt": "bo", "jav_Latn": "jv",
    'bul_Cyrl': 'bg', 'npi_Deva': 'ne', 'zsm_Latn': 'ms'
}

task = "ner"
model = "mbert"

print(task, model)

# Define file paths for pseudo-perplexity and topic classification
pseudo_perplexity_files = {
    "mBERT_Baseline": f'/netscratch/dgurgurov/thesis/downstream_tasks/pppl/flores/{model}/baseline/average_pseudo_perplexities_summary.csv',
    }

task_files = {
    "mBERT_Baseline": f'/netscratch/dgurgurov/thesis/downstream_tasks/{task}/conceptnet_{model}_baseline_results.csv',
    }

model_dict = {"mbert": "mBERT", "xlm-r": "XLM-R"}

task_dict = {"ner": "Named Entity Recognition", "sa": "Sentiment Analysis", "tc": "Topic Classification"}

task_name = f"{task_dict[task]} - {model_dict[model]}"

import scipy.stats as stats

import pandas as pd

def load_and_prepare_baseline_data(pseudo_perplexity_files, tc_files, languages_mapping, model_type):
    # List of models to load
    models = ["mBERT_Baseline", f"mBERT_{model_type}"]

    # Load pseudo-perplexity data for models
    pseudo_perplexity_data = {}
    for key, file in pseudo_perplexity_files.items():
        if key in models:
            df = pd.read_csv(file)
            df["Model"] = key
            pseudo_perplexity_data[key] = df
    
    # Load topic classification data for models and convert F1 to percentage
    tc_data = {}
    for key, file in tc_files.items():
        if key in models:
            df = pd.read_csv(file)
            df["language"] = df["Language"].map(languages_mapping)
            df["Model"] = key
            # Convert F1 score to percentage
            df["Average F1"] = df["Average F1"] * 100
            tc_data[key] = df
    
    # Merge datasets for models
    merged_data = []
    for key in pseudo_perplexity_data:
        if key in tc_data:
            merged = pd.merge(pseudo_perplexity_data[key], tc_data[key],
                            on="language", how="inner")
            merged_data.append(merged)
    
    # Apply language exclusions based on model
    if model == "mbert":
        excluded_languages = ["si", "mt", "ug", "am", "bo", "ku"]
    if model == "xlm-r":
        excluded_languages = ["yo", "mt", "bo"]
    
    # Concatenate all merged data 
    final_data = pd.concat(merged_data, ignore_index=True)
    
    # Print language summary
    initial_languages = set(final_data['language'].unique())
    # final_data = final_data[~final_data['language'].isin(excluded_languages)]
    excluded_count = len(initial_languages) - len(set(final_data['language'].unique()))
    
    print(f"\nLanguage Exclusion Summary:")
    print(f"Initial number of languages: {len(initial_languages)}")
    print(f"Number of languages excluded: {excluded_count}")
    print(f"Final number of languages: {len(set(final_data['language'].unique()))}")
    print("\nExcluded languages:", sorted(list(initial_languages - set(final_data['language'].unique()))))
    print("\nRemaining languages:", sorted(list(set(final_data['language'].unique()))))
    
    return final_data

def plot_baseline_and_adapted_correlation(final_data):
    # Separate baseline and adapted model data
    baseline_data = final_data[final_data['Model_x'] == 'mBERT_Baseline']
    adapted_data = final_data[final_data['Model_x'] == 'mBERT_seq_bn']
    
    # Baseline calculations
    ppl_baseline = baseline_data['average_pseudo_perplexity']
    f1_baseline = baseline_data['Average F1']
    pearson_corr_base, pearson_p_base = stats.pearsonr(ppl_baseline, f1_baseline)
    spearman_corr_base, spearman_p_base = stats.spearmanr(ppl_baseline, f1_baseline)
    
    # Adapted model calculations
    ppl_adapted = adapted_data['average_pseudo_perplexity']
    f1_adapted = adapted_data['Average F1']
    pearson_corr_adapted, pearson_p_adapted = stats.pearsonr(ppl_adapted, f1_adapted)
    spearman_corr_adapted, spearman_p_adapted = stats.spearmanr(ppl_adapted, f1_adapted)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Baseline scatter and regression
    sns.scatterplot(x=ppl_baseline, y=f1_baseline, color='blue', label='Baseline', alpha=0.7)
    sns.regplot(x=ppl_baseline, y=f1_baseline, scatter=False, color='blue', line_kws={'linewidth': 2}, ci=None)
    
    # Adapted model scatter and regression
    sns.scatterplot(x=ppl_adapted, y=f1_adapted, color='red', label='Adapted', alpha=0.7)
    sns.regplot(x=ppl_adapted, y=f1_adapted, scatter=False, color='red', line_kws={'linewidth': 2}, ci=None)
    
    # Formatting
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('Pseudo Perplexity')
    plt.ylabel('F1 Score (%)')
    plt.title(f"{task_name}")
    plt.grid(True)

    # Find the overall minimum F1 score
    y_min = min(f1_baseline.min(), f1_adapted.min())

    # Set y-axis lower limit
    plt.ylim(bottom=y_min * 0.9)  # Start slightly below the lowest F1 score
    
    
    # Add correlation information to the plot
    info_text = (f"Baseline - Pearson: {pearson_corr_base:.2f} (p={pearson_p_base:.2e})\n"
                 f"Baseline - Spearman: {spearman_corr_base:.2f} (p={spearman_p_base:.2e})\n"
                 f"Adapted - Pearson: {pearson_corr_adapted:.2f} (p={pearson_p_adapted:.2e})\n"
                 f"Adapted - Spearman: {spearman_corr_adapted:.2f} (p={spearman_p_adapted:.2e})")
    print(info_text)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"base_cor_{task}_{model}_log_with_adapted", dpi=300, bbox_inches='tight')
    
    return (pearson_corr_base, pearson_p_base, 
            pearson_corr_adapted, pearson_p_adapted)

# Update file paths for pseudo-perplexity and topic classification
pseudo_perplexity_files = {
    "mBERT_Baseline": f'/netscratch/dgurgurov/thesis/downstream_tasks/pppl/flores/{model}/baseline/average_pseudo_perplexities_summary.csv',
    "mBERT_seq_bn": f'/netscratch/dgurgurov/thesis/downstream_tasks/pppl/flores/{model}/seq_bn/average_pseudo_perplexities_summary.csv',
}

task_files = {
    "mBERT_Baseline": f'/netscratch/dgurgurov/thesis/downstream_tasks/{task}/conceptnet_{model}_baseline_results.csv',
    "mBERT_seq_bn": f'/netscratch/dgurgurov/thesis/downstream_tasks/{task}/glot_{model}_seq_bn_results.csv',
}

# Load and prepare the data
final_data = load_and_prepare_baseline_data(pseudo_perplexity_files, task_files, languages_mapping, "seq_bn")

# Plot the baseline and adapted model correlation
results = plot_baseline_and_adapted_correlation(final_data)
