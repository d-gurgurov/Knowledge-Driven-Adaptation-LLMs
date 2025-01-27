import pandas as pd
from datasets import load_dataset
from sklearn.metrics import f1_score
import os

# List of languages
languages = [
    'amh_Ethi', 'azj_Latn', 'ben_Beng', 'bod_Tibt', 'bul_Cyrl',
    'ckb_Arab', 'cym_Latn', 'dan_Latn', 'ell_Grek', 'heb_Hebr',
    'jav_Latn', 'kat_Geor', 'lvs_Latn', 'mar_Deva', 'mlt_Latn',
    'mkd_Cyrl', 'npi_Deva', 'ron_Latn', 'sin_Sinh', 'slk_Latn',
    'slv_Latn', 'sun_Latn', 'swh_Latn', 'tel_Telu', 'tha_Thai',
    'uig_Arab', 'urd_Arab', 'uzn_Latn', 'yor_Latn', 'zsm_Latn'
]

# Directory containing the TSV files
# Download the results from here https://github.com/dadelani/SIB-200
tsv_dir = "/Users/daniilgurgurov/Desktop/sib-200_gpt-results/output_gpt3.5"

# To store F1-scores
language_f1_scores = {}

def compute_f1(language):
    # Load true labels from the dataset
    test_dataset = load_dataset('Davlan/sib200', language, split='test')
    true_labels = [example['category'] for example in test_dataset]

    # Load GPT-3.5 results from the TSV file
    tsv_path = os.path.join(tsv_dir, f"{language}.tsv")
    if not os.path.exists(tsv_path):
        print(f"TSV file for {language} not found. Skipping.")
        return None

    gpt_results = pd.read_csv(tsv_path, sep='\t')

    # Ensure alignment of lengths
    if len(gpt_results) != len(true_labels):
        print(f"Mismatch in dataset length for {language}. Skipping.")
        return None

    # Check if the topic name is inside the GPT-3.5 response
    predictions = []
    for gpt_response, true_label in zip(gpt_results['gpt-3'], true_labels):
        gpt_response_lower = str(gpt_response).lower()
        true_label_lower = str(true_label).lower()

        # Check if the response suggests more than one category
        if "," in gpt_response_lower or "/" in gpt_response_lower:
            predictions.append(False)  # Mark as incorrect if multiple categories are suggested
        else:
            predictions.append(true_label_lower in gpt_response_lower)

    # Convert boolean predictions to labels (1 for correct, 0 for incorrect)
    predicted_labels = [1 if pred else 0 for pred in predictions]
    true_binary_labels = [1] * len(true_labels)  # All true labels are "correct"

    # Compute F1-score
    f1 = f1_score(true_binary_labels, predicted_labels)
    return f1

# Iterate over all languages and compute F1-scores
for lang in languages:
    print(f"Processing {lang}...")
    f1 = compute_f1(lang)
    if f1 is not None:
        language_f1_scores[lang] = f1

# Compute the average F1-score
average_f1 = sum(language_f1_scores.values()) / len(language_f1_scores) if language_f1_scores else 0

# Save results to a file
output_file = "language_f1_scores.tsv"
with open(output_file, "w") as f:
    f.write("Language\tF1-Score\n")
    for lang, f1 in language_f1_scores.items():
        f.write(f"{lang}\t{f1:.4f}\n")
    f.write(f"\nAverage\t{average_f1:.4f}\n")

print(f"Results saved to {output_file}")