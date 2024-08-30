import pandas as pd
import os

# Define the languages
languages = [
    'mar-Deva', 'mkd-Cyrl', 'kat-Geor', 'slk-Latn', 'ell-Grek', 'tha-Thai',
    'azj-Latn', 'lvs-Latn', 'slv-Latn', 'heb-Hebr', 'ron-Latn', 'dan-Latn',
    'urd-Arab', 'uig-Arab', 'bod-Tibt', 'mlt-Latn', 'jav-Latn', 'npi-Deva',
    'zsm-Latn', 'bul-Cyrl', 'tel-Telu', 'ben-Beng'
]

# Directory where the CSV files are located
data_dir = '/netscratch/dgurgurov/thesis/data/glot'  # Change this to your actual data directory

# Function to clean and save the datasets
def clean_and_save_datasets(language):
    # Define file paths
    train_file = os.path.join(data_dir, f'train_glot_{language}.csv')
    val_file = os.path.join(data_dir, f'val_glot_{language}.csv')
    
    # Clean and save the train dataset
    try:
        train_data = pd.read_csv(train_file, on_bad_lines='skip')  # Skip bad lines
        train_data.to_csv(train_file, index=False)  # Save cleaned data
        print(f"Cleaned and saved train dataset for {language}.")
    except Exception as e:
        print(f"Error processing train dataset for {language}: {e}")

    # Clean and save the validation dataset
    try:
        val_data = pd.read_csv(val_file, on_bad_lines='skip')  # Skip bad lines
        val_data.to_csv(val_file, index=False)  # Save cleaned data
        print(f"Cleaned and saved validation dataset for {language}.")
    except Exception as e:
        print(f"Error processing validation dataset for {language}: {e}")

# Loop through each language and clean the datasets
for lang in languages:
    clean_and_save_datasets(lang)