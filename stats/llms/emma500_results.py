import os
import json
import pandas as pd
from sklearn.metrics import f1_score
from datasets import load_dataset

# Define the languages to process
languages = [
    'amh_Ethi', 'azj_Latn', 'ben_Beng', 'bod_Tibt', 'bul_Cyrl',
    'ckb_Arab', 'cym_Latn', 'dan_Latn', 'ell_Grek', 'heb_Hebr',
    'jav_Latn', 'kat_Geor', 'lvs_Latn', 'mar_Deva', 'mlt_Latn',
    'mkd_Cyrl', 'npi_Deva', 'ron_Latn', 'sin_Sinh', 'slk_Latn',
    'slv_Latn', 'sun_Latn', 'swh_Latn', 'tel_Telu', 'tha_Thai',
    'uig_Arab', 'urd_Arab', 'uzn_Latn', 'yor_Latn', 'zsm_Latn'
]

# Directory containing folders for different models, each containing JSONL files
# Download the results from here - https://github.com/MaLA-LM/emma-500/tree/main/evaluation_results/SIB-200
input_root_dir = "/Users/daniilgurgurov/Downloads/results"
output_file = "f1_scores_combined.csv"

# Initialize a list to store the results
all_results = []

# Process each model folder
for model_folder in os.listdir(input_root_dir):
    model_path = os.path.join(input_root_dir, model_folder)
    if not os.path.isdir(model_path):
        continue

    # Initialize a list to store the results for the current model
    model_results = []

    # Process each language file in the model folder
    for lang in languages:
        file_path = os.path.join(model_path, f"{lang}.jsonl")

        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Load the test dataset for the language
        test_dataset = load_dataset('Davlan/sib200', lang, split='test')
        true_labels = [example['category'] for example in test_dataset]

        # Load predictions from the JSONL file
        predicted_labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                predicted_labels.append(data['predicted_category'])

        # Calculate the F1 score
        f1 = f1_score(true_labels, predicted_labels, average='macro')
        print(f"{model_folder} - {lang}: F1-score = {f1:.4f}")

        # Save the result
        model_results.append({"model": model_folder, "language": lang, "f1_score": f1})

    # Calculate the average F1 score for the model
    if model_results:
        average_f1 = sum(result['f1_score'] for result in model_results) / len(model_results)
        print(f"{model_folder}: Average F1-score = {average_f1:.4f}")
        model_results.append({"model": model_folder, "language": "Average", "f1_score": average_f1})

    # Add the model results to the combined results
    all_results.extend(model_results)

# Save all results to a CSV file
results_df = pd.DataFrame(all_results)
results_df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
