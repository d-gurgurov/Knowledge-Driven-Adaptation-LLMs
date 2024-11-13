import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load perplexity scores
perplexity_df = pd.read_csv('/netscratch/dgurgurov/thesis/downstream_tasks/ppl/mbert/flores/baseline/average_perplexities.csv')
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
perplexity_df['language_code'] = perplexity_df['language_code'].replace(languages_mapping)
pseudo_perplexity_df['language'] = pseudo_perplexity_df['language'].replace(languages_mapping)

# Merge datasets on language code
merged_ppl_df = pd.merge(perplexity_df[['language_code', 'perplexity']], 
                         wiki_info_df[['Language', 'Size (GB)']], 
                         left_on='language_code', 
                         right_on='Language')
merged_pppl_df = pd.merge(pseudo_perplexity_df[['language', 'average_pseudo_perplexity']], 
                          wiki_info_df[['Language', 'Size (GB)']], 
                          left_on='language', 
                          right_on='Language')

# Define languages to exclude
excluded_languages = ["si", "mt", "ug", "am", "bo", "ku"]

# Filter out excluded languages
merged_ppl_df = merged_ppl_df[~merged_ppl_df['language_code'].isin(excluded_languages)]
merged_pppl_df = merged_pppl_df[~merged_pppl_df['language'].isin(excluded_languages)]

print(merged_ppl_df)
print(merged_pppl_df)

# Plotting
plt.figure(figsize=(10, 6))
plt.yscale("log")
plt.scatter(merged_ppl_df['Size (GB)'], merged_ppl_df['perplexity'], color='blue', alpha=0.6, label='Perplexity')
plt.scatter(merged_pppl_df['Size (GB)'], merged_pppl_df['average_pseudo_perplexity'], color='green', alpha=0.6, label='Pseudo-Perplexity')
plt.title('Perplexity vs. Wikipedia Dataset Size')
plt.xlabel('Wikipedia Dataset Size (GB)')
plt.ylabel('Perplexity / Pseudo-Perplexity')
plt.grid(True)

# Add language names close to the points, shifted slightly to the left
for i in range(len(merged_ppl_df)):
    plt.text(merged_ppl_df['Size (GB)'].iloc[i] - 0.02, merged_ppl_df['perplexity'].iloc[i], 
             merged_ppl_df['language_code'].iloc[i], fontsize=8, ha='right')

for i in range(len(merged_pppl_df)):
    plt.text(merged_pppl_df['Size (GB)'].iloc[i] - 0.02, merged_pppl_df['average_pseudo_perplexity'].iloc[i], 
             merged_pppl_df['language'].iloc[i], fontsize=8, ha='right')

# Perform polynomial fitting in log space for both datasets
degree = 2
log_sizes_ppl = np.log(merged_ppl_df['Size (GB)'])
log_perplexities = np.log(merged_ppl_df['perplexity'])
log_sizes_pppl = np.log(merged_pppl_df['Size (GB)'])
log_pseudo_perplexities = np.log(merged_pppl_df['average_pseudo_perplexity'])

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
plt.plot(x_fit, y_fit_ppl, color='blue', linestyle='--', label='Polynomial Fit (Perplexity)')
plt.plot(x_fit, y_fit_pppl, color='green', linestyle='--', label='Polynomial Fit (Pseudo-Perplexity)')
plt.legend()

# Save the plot as an image file
plt.savefig('perplexity_vs_pseudo_perplexity_vs_wiki_size_log_fit.png', dpi=300, bbox_inches='tight')

# Close the plot to free up memory
plt.close()

print("Plot saved as 'perplexity_vs_pseudo_perplexity_vs_wiki_size_log_fit.png'")