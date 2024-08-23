import pandas as pd
import os

# Define the list of languages and their corresponding file patterns
language_list = [
    'amh_Ethi', 'uzb_Latn', 'sun_Latn', 'cym_Latn', 'mar_Deva', 
    'kur_Arab', 'mkd_Cyrl', 'kat_Geor', 'slk_Latn', 'ell_Grek', 
    'tha_Thai', 'aze_Latn', 'lvs_Latn', 'slv_Latn', 'heb_Hebr', 
    'ron_Latn', 'dan_Latn', 'urd_Arab', 'sin_Sinh', 'yor_Latn', 
    'swa_Latn', 'uig_Arab', 'bod_Tibt', 'mlt_Latn', 'jav_Latn', 
    'nep_Deva', 'msa_Latn', 'bul_Cyrl', 'tel_Telu', 'ben_Beng', 'tel-Telu', 'ben-Beng'
]

# Initialize a list to store the statistics
stats = []

# Define the path to your data directory
data_dir = "/netscratch/dgurgurov/thesis/data/glot"

# Loop over each language in your list
for lang_code in language_list:
    lang_stats = {
        "Language Code": lang_code,
    }

    # Initialize stats for each split
    for split in ["train", "val", "test"]:
        file_path = os.path.join(data_dir, f"{split}_glot_{lang_code}.csv")
        
        if os.path.exists(file_path):
            # Number of lines (sentences)
            num_lines = sum(1 for _ in open(file_path)) - 1  # subtract 1 for header
            
            # File size in MB
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # Add to stats
            lang_stats[f"{split.capitalize()} Sentences"] = num_lines  # type: ignore
            lang_stats[f"{split.capitalize()} Size (MB)"] = round(file_size_mb, 2)  # type: ignore
        else:
            lang_stats[f"{split.capitalize()} Sentences"] = "N/A"
            lang_stats[f"{split.capitalize()} Size (MB)"] = "N/A"
    
    stats.append(lang_stats)

# Create a DataFrame from the statistics
stats_df = pd.DataFrame(stats)

# Save the DataFrame as a Markdown table
markdown_table = stats_df.to_markdown(index=False)

# Save to a Markdown file
with open("glotdata.md", "w") as f:
    f.write(markdown_table)

print("Statistics table saved to glot_data.md")
