import os
import json
import pandas as pd

# Define base directory path and other configurations
base_dir = "/netscratch/dgurgurov/thesis/donwstream_tasks/tc"
sources = ["conceptnet", "glot"]
models = ["mbert", "xlm-r"]
configurations = ["baseline", "seq_bn", "seq_bn_inv", "lora"]
languages = ['amh_Ethi', 'uzn_Latn', 'sun_Latn', 'cym_Latn', 'mar_Deva', 
             'ckb_Arab', 'mkd_Cyrl', 'kat_Geor', 'slk_Latn', 'ell_Grek', 
             'tha_Thai', 'azj_Latn', 'lvs_Latn', 'slv_Latn', 'heb_Hebr',
             'ron_Latn', 'dan_Latn', 'urd_Arab', 'sin_Sinh', 'yor_Latn', 
             'swh_Latn', 'uig_Arab', 'bod_Tibt', 'mlt_Latn', 'jav_Latn', 
             'npi_Deva', 'zsm_Latn', 'bul_Cyrl', 'tel_Telu', 'ben_Beng']
seeds = [1, 2, 3]

# Traverse through each configuration and collect scores
for source in sources:
    for model in models:
        for config in configurations:
            rows = []  # To store rows for this specific source-model-configuration
            
            for language in languages:
                seed_scores_f1 = []

                # Gather F1 scores from each seed
                for seed in seeds:
                    result_file = os.path.join(base_dir, source, model, config, language, str(seed), "test_metrics.json")
                    if os.path.isfile(result_file):
                        with open(result_file, "r") as f:
                            scores = json.load(f)
                            seed_scores_f1.append(scores["f1"])  # Collect F1 score for each seed

                # Calculate average F1 across seeds and format row
                if seed_scores_f1:
                    avg_f1 = sum(seed_scores_f1) / len(seed_scores_f1)
                    rows.append({
                        "Language": language,
                        "Scores": str(seed_scores_f1),
                        "Average F1": avg_f1
                    })

            # Convert rows to a DataFrame and save to CSV for this configuration
            df = pd.DataFrame(rows)
            output_csv = os.path.join(base_dir, f"{source}_{model}_{config}_results.csv")
            df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")
