import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Parameters
task = "sa"
model = "xlm-r"
lang_names = True

# Mapping dictionaries
model_dict = {"mbert": "mBERT", "xlm-r": "XLM-R"}
task_dict = {"ner": "Named Entity Recognition", "sa": "Sentiment Analysis", "tc": "Topic Classification"}
task_name = f"{task_dict[task]} - {model_dict[model]}"

# Load datasets
perplexity_df = pd.read_csv(f'/netscratch/dgurgurov/thesis/downstream_tasks/{task}/conceptnet_{model}_baseline_results.csv')
pseudo_perplexity_df = pd.read_csv(f'/netscratch/dgurgurov/thesis/downstream_tasks/{task}/glot_{model}_seq_bn_inv_results.csv')
wiki_info_df = pd.read_csv(f'/netscratch/dgurgurov/thesis/src/{("wikipedia" if model == "mbert" else "cc100")}_datasets_info.csv')
language_data_df = pd.read_csv('/netscratch/dgurgurov/thesis/src/language_data_sizes.csv')

# Convert Glot_mb to GB
language_data_df['Glot_GB'] = language_data_df['Glot_mb'] / 1024

# Merge datasets with language information
pseudo_perplexity_df = pd.merge(pseudo_perplexity_df, language_data_df[['language', 'Glot_GB']],
                                left_on='Language', right_on='language')
perplexity_df = pd.merge(perplexity_df, language_data_df[['language', 'Glot_GB']],
                         left_on='Language', right_on='language')

# Language mapping
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

# Map language codes
perplexity_df['Language'] = perplexity_df['Language'].replace(languages_mapping)
pseudo_perplexity_df['Language'] = pseudo_perplexity_df['Language'].replace(languages_mapping)

# Merge with wiki info
merged_ppl_df = pd.merge(perplexity_df[['Language', 'Average F1', 'Glot_GB']],
                         wiki_info_df[['Language', 'Size (GB)']], on='Language')
merged_pppl_df = pd.merge(pseudo_perplexity_df[['Language', 'Average F1', 'Glot_GB']],
                          wiki_info_df[['Language', 'Size (GB)']], on='Language')

# NER training data
if task == "ner":
    task_data = {
        "Language": ["bg", "ms", "mt", "ne", "jv", "ug", "bo", "si", "su", "am", 
                     "sw", "ka", "lv", "sk", "sl", "uz", "yo", "ur", "mk", "da", 
                     "mr", "bn", "he", "ro", "te", "cy", "az", "el", "ku", "th"],
        "#train": [20000, 20000, 100, 100, 100, 100, 100, 100, 100, 100, 
                   1000, 10000, 10000, 20000, 15000, 1000, 100, 20000, 10000, 
                   20000, 5000, 10000, 20000, 20000, 1000, 10000, 10000, 20000, 100, 20000]
    }
elif task == "sa":
    task_data = {
        "Language": [
            "su", "am", "sw", "ka", "ne", "ug", "lv", "sk", "si", "sl", "uz", 
            "bg", "yo", "ur", "mk", "da", "mr", "bn", "he", "ro", "te", "cy", 
            "az", "bo", "ku", "el", "jv", "mt", "th", "ms"
        ],
        "#train": [
            381, 709, 738, 1080, 1189, 1962, 2408, 3560, 3502, 3501, 3273, 
            5412, 5414, 7356, 6557, 7000, 8000, 8264, 8932, 10800, 11386, 
            17500, 19600, 7004, 6000, 5936, 17500, 595, 8103, 7926
        ]
    }

task_data_df = pd.DataFrame(task_data)

# Merge task data
merged_ppl_df = pd.merge(merged_ppl_df, task_data_df, on='Language', how='left')

# Plotting F1 scores vs training examples with regression fit
plt.figure(figsize=(10, 6))

sc = plt.scatter(x=merged_ppl_df['#train'], 
                     y=merged_ppl_df['Average F1'], 
                     c=merged_ppl_df['Size (GB)'],
                     cmap='viridis',
                     alpha=0.7)

sns.regplot(
    x=merged_ppl_df['#train'], 
    y=merged_ppl_df['Average F1'], 
    logx=True, 
    scatter=False,
    line_kws={'color': 'blue'},
    ci=None
)

# Add colorbar
colorbar = plt.colorbar(sc)
colorbar.set_label('Pre-Training Size (GB)', rotation=270, labelpad=15)

# Add labels
plt.title(f"{task_name} - F1 vs Training Examples")
plt.xlabel('Number of Training Examples')
plt.ylabel('F1 Score')
plt.grid(True)

# Optional: Add language names
if lang_names:
    for i in range(len(merged_ppl_df)):
        plt.text(merged_ppl_df['#train'].iloc[i], merged_ppl_df['Average F1'].iloc[i], 
                 merged_ppl_df['Language'].iloc[i], fontsize=5, ha='right')

# Save plot
plt.savefig(f'{task}_vs_training_exs_{model}_with_fit.png', dpi=300, bbox_inches='tight')
plt.close()

# Print correlations
pearson_corr_ppl, _ = pearsonr(merged_ppl_df['#train'], merged_ppl_df['Average F1'])
print(f"F1 Before - Pearson Correlation: {pearson_corr_ppl:.2f}")

spearman_corr_ppl, _ = spearmanr(merged_ppl_df['#train'], merged_ppl_df['Average F1'])
print(f"F1 Before - Spearman Correlation: {spearman_corr_ppl:.2f}")
