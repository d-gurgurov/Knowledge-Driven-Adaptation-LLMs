import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

task = "ner"
model = "xlm-r"
lang_names = False

model_dict = {"mbert": "mBERT", "xlm-r": "XLM-R"}

task_dict = {"ner": "Named Entity Recognition", "sa": "Sentiment Analysis", "tc": "Topic Classification"}

task_name = f"{task_dict[task]} - {model_dict[model]} - ConceptNet"

# Load perplexity scores
perplexity_df = pd.read_csv(f'/netscratch/dgurgurov/thesis/downstream_tasks/{task}/conceptnet_{model}_baseline_results.csv')
# Load pseudo-perplexity scores
pseudo_perplexity_df = pd.read_csv(f'/netscratch/dgurgurov/thesis/downstream_tasks/{task}/conceptnet_{model}_seq_bn_inv_results.csv')
# Load Wikipedia dataset info
if model == "mbert":
    wiki_info_df = pd.read_csv('/netscratch/dgurgurov/thesis/src/wikipedia_datasets_info.csv')
if model == "xlm-r":
    wiki_info_df = pd.read_csv('/netscratch/dgurgurov/thesis/src/cc100_datasets_info.csv')

# Load language adaptation data
language_data_df = pd.read_csv('/netscratch/dgurgurov/thesis/src/language_data_sizes.csv')

# Convert Glot_mb to GB
language_data_df['Glot_GB'] = language_data_df['Glot_mb'] / 1024

pseudo_perplexity_df = pd.merge(pseudo_perplexity_df, language_data_df[['language', 'Glot_GB', 'CN_mb']], 
                         left_on='Language', 
                         right_on='language')

perplexity_df = pd.merge(perplexity_df, language_data_df[['language', 'Glot_GB', 'CN_mb']], 
                         left_on='Language', 
                         right_on='language')

# Define the comprehensive mapping of language codes for all 30 languages
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

# Map language codes in both datasets using the mapping dictionary
perplexity_df['Language'] = perplexity_df['Language'].replace(languages_mapping)
pseudo_perplexity_df['Language'] = pseudo_perplexity_df['Language'].replace(languages_mapping)

# Merge datasets on language code
merged_ppl_df = pd.merge(perplexity_df[['Language', 'Average F1', 'CN_mb']], 
                         wiki_info_df[['Language', 'Size (GB)']], 
                         left_on='Language', 
                         right_on='Language')
merged_pppl_df = pd.merge(pseudo_perplexity_df[['Language', 'Average F1', 'CN_mb']], 
                          wiki_info_df[['Language', 'Size (GB)']], 
                          left_on='Language', 
                          right_on='Language')

# Define color map based on the Glot_GB field for coloring
norm = plt.Normalize(vmin=merged_pppl_df['CN_mb'].min(), vmax=merged_pppl_df['CN_mb'].max())
cmap = plt.cm.plasma

# Define languages to exclude
if model == "mbert":
    excluded_languages = ["si", "mt", "ug", "am", "bo", "ku"]

if model == "xlm-r":
    excluded_languages = ["yo", "mt", "bo"]

# Filter out excluded languages
merged_ppl_df = merged_ppl_df[~merged_ppl_df['Language'].isin(excluded_languages)]
merged_pppl_df = merged_pppl_df[~merged_pppl_df['Language'].isin(excluded_languages)]

# Plotting
plt.figure(figsize=(10, 6))
plt.yscale("linear")
plt.xscale("linear")
plt.scatter(merged_ppl_df['Size (GB)'], merged_ppl_df['Average F1'], color='green', alpha=0.6, label='F1 Before')
sc = plt.scatter(merged_pppl_df['Size (GB)'], merged_pppl_df['Average F1'], c=merged_pppl_df['CN_mb'], cmap=cmap, norm=norm, alpha=0.6, label='F1 After')
plt.title(task_name)
plt.xlabel('Dataset Size (GB)')
plt.ylabel('F1 Score')
plt.grid(True)

if model=="xlm-r":
    cbar = plt.colorbar(sc)
    cbar.set_label('Adaptation Data Size (GB)')

# Add language names close to the points, shifted slightly to the left
if lang_names:
    for i in range(len(merged_ppl_df)):
        plt.text(merged_ppl_df['Size (GB)'].iloc[i] - 0.02, merged_ppl_df['Average F1'].iloc[i], 
                merged_ppl_df['Language'].iloc[i], fontsize=5, ha='right')

    for i in range(len(merged_pppl_df)):
        plt.text(merged_pppl_df['Size (GB)'].iloc[i] - 0.02, merged_pppl_df['Average F1'].iloc[i], 
                merged_pppl_df['Language'].iloc[i], fontsize=5, ha='right')

from scipy.stats import pearsonr, spearmanr

# Calculate Pearson and Spearman correlations for "F1 Before"
pearson_corr_ppl, pearson_pval_ppl = pearsonr(merged_ppl_df['Size (GB)'], merged_ppl_df['Average F1'])
spearman_corr_ppl, spearman_pval_ppl = spearmanr(merged_ppl_df['Size (GB)'], merged_ppl_df['Average F1'])

# Calculate Pearson and Spearman correlations for "F1 After"
pearson_corr_pppl, pearson_pval_pppl = pearsonr(merged_pppl_df['Size (GB)'] + merged_pppl_df['CN_mb'], merged_pppl_df['Average F1'])
spearman_corr_pppl, spearman_pval_pppl = spearmanr(merged_pppl_df['Size (GB)'] + merged_pppl_df['CN_mb'], merged_pppl_df['Average F1'])

# Print the correlation results
print(f"F1 Before - Pearson Correlation: {pearson_corr_ppl:.2f}, p-value: {pearson_pval_ppl:.2e}")
print(f"F1 Before - Spearman Correlation: {spearman_corr_ppl:.2f}, p-value: {spearman_pval_ppl:.2e}")
print(f"F1 After - Pearson Correlation: {pearson_corr_pppl:.2f}, p-value: {pearson_pval_pppl:.2e}")
print(f"F1 After - Spearman Correlation: {spearman_corr_pppl:.2f}, p-value: {spearman_pval_pppl:.2e}")

# Set y-axis limits starting at the smallest value
y_min = merged_ppl_df['Average F1'].min()
plt.ylim(bottom=y_min * 0.9, top=1.0)


# Add labels and title
plt.title(task_name)
plt.xlabel('Dataset Size (GB)')
plt.ylabel('F1 Score')
plt.grid(True)

# Perform polynomial fitting in log space for both datasets
degree = 1
log_sizes_ppl = np.log(merged_ppl_df['Size (GB)'])
log_perplexities = np.log(merged_ppl_df['Average F1'])
log_sizes_pppl = np.log(merged_pppl_df['Size (GB)'])
log_pseudo_perplexities = np.log(merged_pppl_df['Average F1'])

# Fit polynomial to log-transformed data
ppl_fit = np.polyfit(log_sizes_ppl, log_perplexities, degree)
pppl_fit = np.polyfit(log_sizes_pppl, log_pseudo_perplexities, degree)
p_ppl = np.poly1d(ppl_fit)
p_pppl = np.poly1d(pppl_fit)

# Generate x values for plotting the polynomial fit line in original space
x_fit = np.linspace(min(merged_ppl_df['Size (GB)'].min(), merged_pppl_df['Size (GB)'].min()), 
                    max(merged_ppl_df['Size (GB)'].max(), merged_pppl_df['Size (GB)'].max()), 100)
y_fit_ppl = np.exp(p_ppl(np.log(x_fit)))  # Transform back to original scale
y_fit_pppl = np.exp(p_pppl(np.log(x_fit)))  # Transform back to original scale

# Plot the polynomial fit lines
plt.plot(x_fit, y_fit_ppl, color='green', linestyle='--', label='Fit (F1 Before)')
plt.plot(x_fit, y_fit_pppl, color='blue', linestyle='--', label='Fit (F1 After)')
plt.legend()

# Save the plot as an image file
plt.savefig(f'{task}_vs_data_cn_{model}.png', dpi=300, bbox_inches='tight')

# Close the plot to free up memory
plt.close()


# Calculate improvement in F1 score
improvement_df = pd.merge(merged_pppl_df[['Language', 'Average F1', 'CN_mb']], 
                          merged_ppl_df[['Language', 'Average F1']], 
                          on='Language', 
                          suffixes=('_after', '_before'))

improvement_df['F1_Improvement'] = improvement_df['Average F1_after'] - improvement_df['Average F1_before']

# Plotting CN_mb vs F1 Improvement
plt.figure(figsize=(10, 6))
plt.yscale("linear")
plt.xscale("linear")
plt.scatter(improvement_df['CN_mb'], improvement_df['F1_Improvement'], color='blue', alpha=0.6)

# Add labels for languages, if desired
if lang_names:
    for i in range(len(improvement_df)):
        plt.text(improvement_df['CN_mb'].iloc[i], improvement_df['F1_Improvement'].iloc[i], 
                 improvement_df['Language'].iloc[i], fontsize=8, ha='right')

# Add regression line (linear fit)
fit = np.polyfit(improvement_df['CN_mb'], improvement_df['F1_Improvement'], 1)
p = np.poly1d(fit)
x_fit = np.linspace(improvement_df['CN_mb'].min(), improvement_df['CN_mb'].max(), 100)
y_fit = p(x_fit)

plt.plot(x_fit, y_fit, color='red', linestyle='--', label='Linear Fit')

# Add labels and title
plt.title(f'Adaptation Data Size vs F1 Improvement ({task_name})')
plt.xlabel('Adaptation Data Size (CN_mb)')
plt.ylabel('F1 Improvement')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig(f'{task}_vs_data_cn_improvement_{model}.png', dpi=300, bbox_inches='tight')

# Close the plot to free up memory
plt.close()

