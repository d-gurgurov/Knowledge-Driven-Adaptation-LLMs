import pandas as pd
from datasets import load_dataset

# List of languages to download
languages = [
    'am', 'uz', 'su', 'cy', 'mr', 'te', 'ku', 'mk', 
    'bn', 'ka', 'sk', 'el', 'th', 'az', 'lv', 
    'sl', 'he', 'ro', 'da', 'ur', 'si', 
    'yo', 'sw', 'ug', 'bo', 'mt', 
    'jv', 'ne', 'ms', 'bg'
]

# Initialize a list to store dataset info
dataset_info = []

# Loop through each language and load the dataset
for lang in languages:
    try:
        # Load dataset for the specified language and date
        dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}")  # Adjust date as necessary
        
        # Get number of documents and size in GB
        num_documents = len(dataset['train'])  # Assuming we want the training set
        size_in_gb = dataset['train'].info.size_in_bytes / (1024**3)  # Convert bytes to GB
        
        # Append info to list
        dataset_info.append({
            "Language": lang,
            "Number of Documents": num_documents,
            "Size (GB)": size_in_gb
        })
        print(size_in_gb)
        
    except Exception as e:
        print(f"Error loading dataset for {lang}: {e}")

# Create a DataFrame from the collected info
df = pd.DataFrame(dataset_info)

# Save to CSV file
df.to_csv("wikipedia_datasets_info.csv", index=False)

print("Dataset information saved to wikipedia_datasets_info.csv")