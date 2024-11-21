import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load perplexity scores
perplexity_df = pd.read_csv('/netscratch/dgurgurov/thesis/downstream_tasks/pppl/flores/mbert/seq_bn/average_pseudo_perplexities_summary.csv')
# Load pseudo-perplexity scores
pseudo_perplexity_df = pd.read_csv('/netscratch/dgurgurov/thesis/downstream_tasks/pppl/flores/mbert/baseline/average_pseudo_perplexities_summary.csv')
# Load Wikipedia dataset info
wiki_info_df = pd.read_csv('/netscratch/dgurgurov/thesis/src/wikipedia_datasets_info.csv')

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
perplexity_df['language'] = perplexity_df['language'].replace(languages_mapping)
pseudo_perplexity_df['language'] = pseudo_perplexity_df['language'].replace(languages_mapping)

# Merge datasets on language code
merged_ppl_df = pd.merge(perplexity_df[['language', 'average_pseudo_perplexity']], 
                         wiki_info_df[['Language', 'Size (GB)']], 
                         left_on='language', 
                         right_on='Language')
merged_pppl_df = pd.merge(pseudo_perplexity_df[['language', 'average_pseudo_perplexity']], 
                          wiki_info_df[['Language', 'Size (GB)']], 
                          left_on='language', 
                          right_on='Language')

# Define languages to exclude
excluded_languages = ["si", "mt", "ug", "am", "bo", "ku"]

# Filter out excluded languages
merged_ppl_df = merged_ppl_df[~merged_ppl_df['language'].isin(excluded_languages)]
merged_pppl_df = merged_pppl_df[~merged_pppl_df['language'].isin(excluded_languages)]

# Load language adaptation data
language_data_df = pd.read_csv('/netscratch/dgurgurov/thesis/src/language_data_sizes.csv')

# Convert Glot_mb to GB
language_data_df['Glot_GB'] = language_data_df['Glot_mb'] / 1024

merged_pppl_df = pd.merge(merged_pppl_df, language_data_df[['language_code', 'Glot_GB']], 
                         left_on='Language', 
                         right_on='language_code')

# Merge the adaptation data with the merged perplexity dataframe for the 'after adaptation' scores
merged_ppl_df = pd.merge(merged_ppl_df, language_data_df[['language_code', 'Glot_GB']], 
                         left_on='language', 
                         right_on='language_code')

from scipy.stats import pearsonr, spearmanr

# Calculate Pearson and Spearman correlations for "F1 Before"
pearson_corr_ppl, pearson_pval_ppl = pearsonr(merged_ppl_df['Size (GB)'], merged_ppl_df['average_pseudo_perplexity'])
spearman_corr_ppl, spearman_pval_ppl = spearmanr(merged_ppl_df['Size (GB)'], merged_ppl_df['average_pseudo_perplexity'])

# Calculate Pearson and Spearman correlations for "F1 After"
pearson_corr_pppl, pearson_pval_pppl = pearsonr(merged_pppl_df['Size (GB)'] + merged_pppl_df['Glot_GB'], merged_pppl_df['average_pseudo_perplexity'])
spearman_corr_pppl, spearman_pval_pppl = spearmanr(merged_pppl_df['Size (GB)'] + merged_pppl_df['Glot_GB'], merged_pppl_df['average_pseudo_perplexity'])

# Print the correlation results
print(f"F1 Before - Pearson Correlation: {pearson_corr_ppl:.2f}, p-value: {pearson_pval_ppl:.2e}")
print(f"F1 Before - Spearman Correlation: {spearman_corr_ppl:.2f}, p-value: {spearman_pval_ppl:.2e}")
print(f"F1 After - Pearson Correlation: {pearson_corr_pppl:.2f}, p-value: {pearson_pval_pppl:.2e}")
print(f"F1 After - Spearman Correlation: {spearman_corr_pppl:.2f}, p-value: {spearman_pval_pppl:.2e}")


# Define color map based on the Glot_GB field for coloring
norm = plt.Normalize(vmin=merged_ppl_df['Glot_GB'].min(), vmax=merged_ppl_df['Glot_GB'].max())
cmap = plt.cm.plasma

# Plotting with color mapping
plt.figure(figsize=(10, 6))
plt.yscale("log")

# Scatter plot with color based on adaptation data size
sc = plt.scatter(merged_ppl_df['Size (GB)'], 
                 merged_ppl_df['average_pseudo_perplexity'], 
                 c=merged_ppl_df['Glot_GB'], 
                 cmap=cmap, norm=norm, alpha=0.6, label='Pseudo-Perplexity (after)')

# Scatter plot for 'before' scores in green
plt.scatter(merged_pppl_df['Size (GB)'], merged_pppl_df['average_pseudo_perplexity'], 
            color='green', alpha=0.6, label='Pseudo-Perplexity (before)')

# # Color bar
# cbar = plt.colorbar(sc)
# cbar.set_label('Adaptation Data Size (GB)')

plt.title('mBERT Base vs. Seq_bn - Pseudo-Perplexity vs. Wikipedia Dataset Size')
plt.xlabel('Wikipedia Dataset Size (GB)')
plt.ylabel('Pseudo-Perplexity before and after adaptation')
plt.grid(True)

# Plot language names centered inside the circles
for i in range(len(merged_ppl_df)):
    plt.text(merged_ppl_df['Size (GB)'].iloc[i], merged_ppl_df['average_pseudo_perplexity'].iloc[i], 
             merged_ppl_df['language'].iloc[i], fontsize=5, ha='center', va='center', color="black")

for i in range(len(merged_pppl_df)):
    plt.text(merged_pppl_df['Size (GB)'].iloc[i], merged_pppl_df['average_pseudo_perplexity'].iloc[i], 
             merged_pppl_df['language'].iloc[i], fontsize=5, ha='center', va='center', color="black")


# Polynomial fitting
degree = 1
log_sizes_ppl = np.log(merged_ppl_df['Size (GB)'])
log_perplexities = np.log(merged_ppl_df['average_pseudo_perplexity'])
log_sizes_pppl = np.log(merged_pppl_df['Size (GB)'])
log_pseudo_perplexities = np.log(merged_pppl_df['average_pseudo_perplexity'])

# Fit and plot polynomial fits
ppl_fit = np.polyfit(log_sizes_ppl, log_perplexities, degree)
pppl_fit = np.polyfit(log_sizes_pppl, log_pseudo_perplexities, degree)
p_ppl = np.poly1d(ppl_fit)
p_pppl = np.poly1d(pppl_fit)
x_fit = np.linspace(min(merged_ppl_df['Size (GB)'].min(), merged_pppl_df['Size (GB)'].min()), 
                    max(merged_ppl_df['Size (GB)'].max(), merged_pppl_df['Size (GB)'].max()), 100)
y_fit_ppl = np.exp(p_ppl(np.log(x_fit)))
y_fit_pppl = np.exp(p_pppl(np.log(x_fit)))

plt.plot(x_fit, y_fit_ppl, color='blue', linestyle='--', label='Polynomial Fit (After Adaptation)')
plt.plot(x_fit, y_fit_pppl, color='green', linestyle='--', label='Polynomial Fit (Before Adaptation)')
plt.legend()

# Save and close the plot
plt.savefig('ppl_vs_wiki_colored.png', dpi=300, bbox_inches='tight')
plt.close()
