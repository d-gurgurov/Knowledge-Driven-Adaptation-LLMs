import pandas as pd
import os

# Define language metadata (this is an example, adjust it according to your data)
language_metadata = {
    "bg": {"full_name": "Bulgarian", "family": "Indo-European"},
    "ms": {"full_name": "Indonesian", "family": "Austronesian"},
    "ne": {"full_name": "Nepali", "family": "Indo-European"},
    "jv": {"full_name": "Javanese", "family": "Austronesian"},
    "mt": {"full_name": "Maltese", "family": "Afro-Asiatic"},
    "bo": {"full_name": "Tibetan", "family": "Sino-Tibetan"},
    "ug": {"full_name": "Uyghur", "family": "Turkic"},
    "sw": {"full_name": "Swahili", "family": "Niger-Congo"},
    "yo": {"full_name": "Yoruba", "family": "Niger-Congo"},
    "si": {"full_name": "Sinhala", "family": "Indo-European"},
    "ur": {"full_name": "Urdu", "family": "Indo-European"},
    "da": {"full_name": "Danish", "family": "Indo-European"},
    "ro": {"full_name": "Romanian", "family": "Indo-European"},
    "he": {"full_name": "Hebrew", "family": "Afro-Asiatic"},
    "sl": {"full_name": "Slovenian", "family": "Indo-European"},
    "lv": {"full_name": "Latvian", "family": "Indo-European"},
    "az": {"full_name": "Azerbaijani", "family": "Turkic"},
    "th": {"full_name": "Thai", "family": "Kra-Dai"},
    "el": {"full_name": "Greek", "family": "Indo-European"},
    "sk": {"full_name": "Slovak", "family": "Indo-European"},
    "ka": {"full_name": "Georgian", "family": "Kartvelian"},
    "bn": {"full_name": "Bengali", "family": "Indo-European"},
    "mk": {"full_name": "Macedonian", "family": "Indo-European"},
    "ku": {"full_name": "Kurdish", "family": "Indo-European"},
    "te": {"full_name": "Telugu", "family": "Dravidian"},
    "mr": {"full_name": "Marathi", "family": "Indo-European"},
    "cy": {"full_name": "Welsh", "family": "Indo-European"},
    "su": {"full_name": "Sundanese", "family": "Austronesian"},
    "uz": {"full_name": "Uzbek", "family": "Turkic"},
    "am": {"full_name": "Amharic", "family": "Afro-Asiatic"}
}


# Initialize a list to store the statistics
stats = []

# Define the path to your data directory
data_dir = "/netscratch/dgurgurov/thesis/data/conceptnet"

# Loop over each language in your metadata
for lang_code, meta in language_metadata.items():
    lang_stats = {
        "Language Code": lang_code,
        "Language Name": meta["full_name"],
        "Language Family": meta["family"]
    }
    
    # Initialize stats for each split
    for split in ["train", "val", "test"]:
        file_path = os.path.join(data_dir, f"{split}_cn_{lang_code}.csv")
        
        if os.path.exists(file_path):
            # Number of lines (sentences)
            num_lines = sum(1 for _ in open(file_path)) - 1  # subtract 1 for header
            
            # File size in MB
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # Add to stats
            lang_stats[f"{split.capitalize()} Sentences"] = num_lines #type: ignore
            lang_stats[f"{split.capitalize()} Size (MB)"] = round(file_size_mb, 2) #type: ignore
        else:
            lang_stats[f"{split.capitalize()} Sentences"] = "N/A"
            lang_stats[f"{split.capitalize()} Size (MB)"] = "N/A"
    
    stats.append(lang_stats)

# Create a DataFrame from the statistics
stats_df = pd.DataFrame(stats)

# Save the DataFrame as a Markdown table
markdown_table = stats_df.to_markdown(index=False)

# Save to a Markdown file
with open("conceptnet_data.md", "w") as f:
    f.write(markdown_table)

print("Statistics table saved to conceptnet_stats.md")
