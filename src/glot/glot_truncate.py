import os

# Define the directory where the CSV files are located
directory = '/netscratch/dgurgurov/thesis/data/glot'

# Define the maximum file sizes
max_size_train = 1 * 1024 * 1024 * 1024  # 1 GB for training files
max_size_val = 200 * 1024 * 1024  # 200 MB for validation files

# List of language codes as provided
language_codes_glotcc = [ 'amh-Ethi', 'uzn-Latn', 'sun-Latn', 'cym-Latn', 'mar-Deva', 
                          'ckb-Arab', 'mkd-Cyrl', 'kat-Geor', 'slk-Latn',  'ell-Grek', 
                          'tha-Thai', 'azj-Latn', 'lvs-Latn', 'slv-Latn', 'heb-Hebr',
                          'ron-Latn', 'dan-Latn', 'urd-Arab', 'sin-Sinh', 'yor-Latn', 
                          'swh-Latn', 'uig-Arab', 'bod-Tibt', 'mlt-Latn', 'jav-Latn', 
                          'npi-Deva', 'zsm-Latn', 'bul-Cyrl', 'tel-Telu', 'ben-Beng']

# Iterate over each language code
for code in language_codes_glotcc:
    # Construct file paths for train and val files
    train_file = os.path.join(directory, f'train_glot_{code}.csv')
    val_file = os.path.join(directory, f'val_glot_{code}.csv')

    # Check and truncate the train file if it exists
    if os.path.exists(train_file):
        file_size = os.path.getsize(train_file)
        if file_size > max_size_train:
            print(f"Truncating {train_file} (size: {file_size / (1024**3):.2f} GB)")
            with open(train_file, 'r') as f:
                lines = f.readlines()

            # Approximate number of lines to keep to stay within 1 GB
            keep_lines = int(len(lines) * (max_size_train / file_size))

            with open(train_file, 'w') as f:
                f.writelines(lines[:keep_lines])

    # Check and truncate the val file if it exists
    if os.path.exists(val_file):
        file_size = os.path.getsize(val_file)
        if file_size > max_size_val:
            print(f"Truncating {val_file} (size: {file_size / (1024**2):.2f} MB)")
            with open(val_file, 'r') as f:
                lines = f.readlines()

            # Approximate number of lines to keep to stay within 200 MB
            keep_lines = int(len(lines) * (max_size_val / file_size))

            with open(val_file, 'w') as f:
                f.writelines(lines[:keep_lines])
