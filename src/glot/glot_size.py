import pandas as pd
import os

# Define language metadata (this is an example, adjust it according to your data)
language_metadata = {
    "amh-Ethi": {"full_name": "Amharic", "family": "Afro-Asiatic"},
    "uzn-Latn": {"full_name": "Uzbek", "family": "Turkic"},
    "sun-Latn": {"full_name": "Sundanese", "family": "Austronesian"},
    "cym-Latn": {"full_name": "Welsh", "family": "Indo-European"},
    "mar-Deva": {"full_name": "Marathi", "family": "Indo-European"},
    "ckb-Arab": {"full_name": "Central Kurdish", "family": "Indo-European"},
    "mkd-Cyrl": {"full_name": "Macedonian", "family": "Indo-European"},
    "kat-Geor": {"full_name": "Georgian", "family": "Kartvelian"},
    "slk-Latn": {"full_name": "Slovak", "family": "Indo-European"},
    "ell-Grek": {"full_name": "Greek", "family": "Indo-European"},
    "tha-Thai": {"full_name": "Thai", "family": "Kra-Dai"},
    "azj-Latn": {"full_name": "Azerbaijani", "family": "Turkic"},
    "lvs-Latn": {"full_name": "Latvian", "family": "Indo-European"},
    "slv-Latn": {"full_name": "Slovenian", "family": "Indo-European"},
    "heb-Hebr": {"full_name": "Hebrew", "family": "Afro-Asiatic"},
    "ron-Latn": {"full_name": "Romanian", "family": "Indo-European"},
    "dan-Latn": {"full_name": "Danish", "family": "Indo-European"},
    "urd-Arab": {"full_name": "Urdu", "family": "Indo-European"},
    "sin-Sinh": {"full_name": "Sinhala", "family": "Indo-European"},
    "yor-Latn": {"full_name": "Yoruba", "family": "Niger-Congo"},
    "swh-Latn": {"full_name": "Swahili", "family": "Niger-Congo"},
    "uig-Arab": {"full_name": "Uyghur", "family": "Turkic"},
    "bod-Tibt": {"full_name": "Tibetan", "family": "Sino-Tibetan"},
    "mlt-Latn": {"full_name": "Maltese", "family": "Afro-Asiatic"},
    "jav-Latn": {"full_name": "Javanese", "family": "Austronesian"},
    "npi-Deva": {"full_name": "Nepali", "family": "Indo-European"},
    "zsm-Latn": {"full_name": "Malay", "family": "Austronesian"},
    "bul-Cyrl": {"full_name": "Bulgarian", "family": "Indo-European"},
    "tel-Telu": {"full_name": "Telugu", "family": "Dravidian"},
    "ben-Beng": {"full_name": "Bengali", "family": "Indo-European"}
}

# Initialize a list to store the statistics
stats = []

# Define the path to your data directory
data_dir = "/netscratch/dgurgurov/thesis/data/glot"

# Loop over each language in your metadata
for lang_code, meta in language_metadata.items():
    lang_stats = {
        "Language Code": lang_code,
        "Language Name": meta["full_name"],
        "Language Family": meta["family"]
    }
    
    # Initialize stats for each split
    for split in ["train", "val"]:
        file_path = os.path.join(data_dir, f"{split}_glot_{lang_code}.csv")
        
        if os.path.exists(file_path):
            # Number of lines (sentences)
            num_lines = sum(1 for _ in open(file_path)) - 1  # subtract 1 for header
            
            # File size in MB
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # Add to stats
            lang_stats[f"{split.capitalize()} Sentences"] = num_lines
            lang_stats[f"{split.capitalize()} Size (MB)"] = round(file_size_mb, 2)
        else:
            lang_stats[f"{split.capitalize()} Sentences"] = "N/A"
            lang_stats[f"{split.capitalize()} Size (MB)"] = "N/A"
    
    stats.append(lang_stats)

# Create a DataFrame from the statistics
stats_df = pd.DataFrame(stats)

# Save the DataFrame as a Markdown table
markdown_table = stats_df.to_markdown(index=False)

# Save to a Markdown file
with open("glotcc_data.md", "w") as f:
    f.write(markdown_table)

print("Statistics table saved to glotcc_data.md")