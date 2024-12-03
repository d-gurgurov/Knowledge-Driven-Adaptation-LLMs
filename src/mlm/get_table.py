import os
import json

# Base directory for all info
base_dir = '/ds/text/ConceptNet/lang_adapters'
output_file = 'evaluation_acc_table.tex'  # Specify the output .tex file name

languages_mapping = {
        "tel_Telu": "te", "ben_Beng": "bn", "lvs_Latn": "lv", "mlt_Latn": "mt", "amh_Ethi": "am", 
        "uzn_Latn": "uz", "sun_Latn": "su", "cym_Latn": "cy", "mar_Deva": "mr", "ckb_Arab": "ku", 
        "mkd_Cyrl": "mk", "kat_Geor": "ka", "slk_Latn": "sk", "ell_Grek": "el", "tha_Thai": "th", 
        "azj_Latn": "az", "slv_Latn": "sl", "heb_Hebr": "he", "ron_Latn": "ro", "dan_Latn": "da", 
        "urd_Arab": "ur", "sin_Sinh": "si", "yor_Latn": "yo", "swh_Latn": "sw", "uig_Arab": "ug",
        "bod_Tibt": "bo", "jav_Latn": "jv", "npi_Deva": "ne", "bul_Cyrl": "bg", "quy_Latn": "qu", 
        "lim_Latn": "li", "wol_Latn": "wo", "gla_Latn": "gd", "mya_Mymr": "my", "ydd_Hebr": "yi",
        "hau_Latn": "ha", "snd_Arab": "sd", "som_Latn": "so", "ckb_Arab": "ku", "pbt_Arab": "ps", "khm_Khmr": "km",
        "guj_Gujr": "gu", "afr_Latn": "af", "glg_Latn": "gl", "isl_Latn": "is", "kaz_Cyrl": "kk", "azj_Latn": "az", 
        "tam_Taml": "ta", "lij_Latn": "lv", "ell_Grek": "el", "ukr_Cyrl": "uk", "srd_Latn": "sc", "grn_Latn": "gn",
        "lin_Latn": "li", "zul_Latn": "zu", "hat_Latn": "ht", "xho_Latn": "xh", "jav_Latn": "jv", "san_Deva": "sa",
        "lao_Laoo": "la", "pan_Guru": "pa", "gle_Latn": "ga", "kir_Cyrl": "ky", "epo_Latn": "eo", "kan_Knda": "kn",
        "bel_Cyrl": "be", "hye_Armn": "hy", "mal_Mlym": "ml", "est_Latn": "et", "zsm_Latn": "ms", "lit_Latn": "lt",
        "tha_Thai": "th"
    }

# Initialize a dictionary to hold evaluation losses
eval_losses = {}

# Walk through the directory structure
for model in os.listdir(base_dir):
    model_path = os.path.join(base_dir, model)
    if os.path.isdir(model_path):
        for architecture in os.listdir(model_path):
            arch_path = os.path.join(model_path, architecture)
            if os.path.isdir(arch_path):
                for adapter_config in os.listdir(arch_path):
                    config_path = os.path.join(arch_path, adapter_config)
                    if os.path.isdir(config_path):
                        for language in os.listdir(config_path):
                            lang_path = os.path.join(config_path, language)
                            if os.path.isdir(lang_path):
                                # Check if language name needs to be mapped
                                iso_code = languages_mapping.get(language, language)  # Use mapped value or original name
                                
                                # Read the JSON file
                                json_file = os.path.join(lang_path, 'all_results.json')
                                try:
                                    with open(json_file, 'r') as f:
                                        results = json.load(f)
                                        eval_loss = results.get('eval_accuracy', None)

                                        # Round eval_loss to 2 decimal places if it's a float
                                        if isinstance(eval_loss, float):
                                            eval_loss = round(eval_loss, 2)

                                        # Initialize nested dictionaries if not present
                                        if iso_code not in eval_losses:
                                            eval_losses[iso_code] = {}
                                        if model not in eval_losses[iso_code]:
                                            eval_losses[iso_code][model] = {}
                                        if architecture not in eval_losses[iso_code][model]:
                                            eval_losses[iso_code][model][architecture] = {}
                                        if adapter_config not in eval_losses[iso_code][model][architecture]:
                                            eval_losses[iso_code][model][architecture][adapter_config] = {}

                                        # Store evaluation loss
                                        eval_losses[iso_code][model][architecture][adapter_config] = eval_loss
                                except FileNotFoundError:
                                    print(f"File not found: {json_file}")
                                except json.JSONDecodeError:
                                    print(f"Error decoding JSON from: {json_file}")


# Initialize the LaTeX table content
latex_table = r"""
\begin{table}[h!]
\centering
\small
\def\arraystretch{1.05}
\resizebox{\columnwidth}{!}{
\begin{tabular}{l|cccccc|cccccc}
\toprule
\multirow{2}{*}{\textbf{ISO}} & \multicolumn{6}{c}{\textbf{ConceptNet}} & \multicolumn{6}{c}{\textbf{Glot}} \\
\cmidrule(lr){2-7} \cmidrule(lr){8-14}
 &  \multicolumn{3}{c}{\textbf{mBERT}} & \multicolumn{3}{c}{\textbf{XLM-R}} & \multicolumn{3}{c}{\textbf{mBERT}} & \multicolumn{3}{c}{\textbf{XLM-R}} \\
 & \textbf{S\_bn} & \textbf{LoRA} & \textbf{S\_bn\_inv} & \textbf{S\_bn} & \textbf{LoRA} & \textbf{S\_bn\_inv} & \textbf{S\_bn} & \textbf{LoRA} & \textbf{S\_bn\_inv} & \textbf{S\_bn} & \textbf{LoRA} & \textbf{S\_bn\_inv} \\
 \midrule
"""

# Construct the LaTeX table rows
for iso_code, sources in eval_losses.items():
    # Start a row with the language ISO code
    row = f"{iso_code} "

    # Process each data source, starting with ConceptNet and then Glot
    for source in ['conceptnet', 'glot']:
        if source in sources:
            models = sources[source]
            # For each model within the source
            for model in ['mbert', 'xlm-r']:
                if model in models:
                    adapter_configs = models[model]

                    # Retrieve evaluation losses for each configuration, using "-" if missing
                    s_bn_loss = adapter_configs.get('seq_bn', "-")
                    lora_loss = adapter_configs.get('lora', "-")
                    s_bn_inv_loss = adapter_configs.get('seq_bn_inv', "-")

                    # Add losses to the row in the specified order
                    row += f"& {s_bn_loss} & {lora_loss} & {s_bn_inv_loss} "
                else:
                    # Add placeholders if the model is missing
                    row += "& - & - & - "
        else:
            # Add placeholders if the source is missing
            row += "& - & - & - & - & - & - "

    # Finish the row
    latex_table += row + r"\\" + "\n"

# Finalize the LaTeX table
latex_table += r"""
\bottomrule
\end{tabular}
}
\caption{Evaluation Losses for Language Adapters by Model, Architecture, and Language}
\label{tab:evaluation_losses}
\end{table}
"""

# Write to the output file
output_path = os.path.join(base_dir, output_file)
with open(output_file, 'w') as f:
    f.write(latex_table)

print("LaTeX table generated and saved to", output_file)
