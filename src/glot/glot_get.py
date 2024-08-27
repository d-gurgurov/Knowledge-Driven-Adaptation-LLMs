from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Define the language codes in the format expected by the dataset
# cis-lmu/Glot500 - missing - tel_Telu, ben_Beng

language_codes_glot500 = ['amh_Ethi', 'uzb_Latn', 'sun_Latn', 'cym_Latn', 'mar_Deva', 
                  'kur_Arab', 'mkd_Cyrl', 'kat_Geor', 'slk_Latn', 'ell_Grek', 
                  'tha_Thai', 'aze_Latn', 'lvs_Latn', 'slv_Latn', 'heb_Hebr', 
                  'ron_Latn', 'dan_Latn', 'urd_Arab', 'sin_Sinh', 'yor_Latn', 
                  'swa_Latn', 'uig_Arab', 'bod_Tibt', 'mlt_Latn', 'jav_Latn', 
                  'nep_Deva', 'msa_Latn', 'bul_Cyrl', 'tel_Telu', 'ben_Beng']

language_codes_glotcc = ['amh-Ethi', 'uzb-Latn', 'sun-Latn', 'cym-Latn', 'mar-Deva', 
                         'ckb-Arab', 'mkd-Cyrl', 'kat-Geor', 'slk-Latn', 'ell-Grek', 
                         'tha-Thai', 'azj-Latn', 'lvs-Latn', 'slv-Latn', 'heb-Hebr',
                         'ron-Latn', 'dan-Latn', 'urd-Arab', 'sin-Sinh', 'yor-Latn', 
                         'swh-Latn', 'uig-Arab', 'bod-Tibt', 'mlt-Latn', 'jav-Latn', 
                         'npi-Deva', 'zsm-Latn', 'bul_Cyrl', 'tel-Telu', 'ben-Beng']

for lang_code in language_codes_glotcc:
    # Load the dataset
    ds = load_dataset("cis-lmu/GlotCC-v1", lang_code)
    
    # Access the training data
    train_data = ds['train'] # type: ignore
    
    # Convert to a Pandas DataFrame
    df = pd.DataFrame(train_data) # type: ignore
    
    # Split the data into 80% train, 10% validation, 10% test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Save the train, validation, and test data to CSV files
    train_df.to_csv(f"/netscratch/dgurgurov/thesis/data/glot/train_glot_{lang_code}.csv", index=False, escapechar='\\')
    val_df.to_csv(f"/netscratch/dgurgurov/thesis/data/glot/val_glot_{lang_code}.csv", index=False, escapechar='\\')
    test_df.to_csv(f"/netscratch/dgurgurov/thesis/data/glot/test_glot_{lang_code}.csv", index=False, escapechar='\\')
    
    print(f"Saved {lang_code} train data to train_glot_{lang_code}.csv")
    print(f"Saved {lang_code} validation data to val_glot_{lang_code}.csv")
    print(f"Saved {lang_code} test data to test_glot_{lang_code}.csv")