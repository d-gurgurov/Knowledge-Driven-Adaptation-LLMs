import pandas as pd

def read_data(file_path):
    # Read the CSV file into a pandas DataFrame
    return pd.read_csv(file_path)

def calculate_group_averages(df, target_languages):
    """
    Calculate averages for languages in target_languages and those not in it
    """
    # Create masks for the two groups
    in_group = df['Language'].isin(target_languages)
    
    # Calculate averages for both groups
    target_avg = df[in_group].iloc[:, 1:].mean(numeric_only=True).round(2)
    other_avg = df[~in_group].iloc[:, 1:].mean(numeric_only=True).round(2)
    
    return target_avg, other_avg

def generate_latex_table(output_file='model_comparison_ppl.tex', **data_files):
    # Read and merge all the data files provided as arguments
    merged_data = None
    suffix_counter = 0  # Initialize a counter for unique suffixes
    
    for model_name, file_path in data_files.items():
        data = read_data(file_path)
        data = data.rename(columns={"Average F1": f"{model_name}"})
        
        # Round values to two decimal places
        data[f'{model_name}'] = (data[f'{model_name}'] * 100)
        
        if merged_data is None:
            merged_data = data
        else:
            suffix = f"_{suffix_counter}"  # Generate a unique suffix for each merge
            merged_data = pd.merge(merged_data, data, on='Language', suffixes=('', suffix))
            suffix_counter += 1

    # Calculate averages for each model's column (excluding the 'Language' column)
    averages = merged_data.iloc[:, 1:].mean(numeric_only=True).round(2) 
    print(averages)

    # Define target languages
    target_languages = [
        "sin_Sinh",  # Sinhala
        "mlt_Latn",  # Maltese
        "uig_Arab",  # Uyghur
        "amh_Ethi",  # Amharic
        "bod_Tibt",  # Tibetan
        "ckb_Arab",  # Kurdish
        "yor_Latn"   # Yoruba
    ]

    # Calculate group averages
    target_avg, other_avg = calculate_group_averages(merged_data, target_languages)
    
    print(f"\nAverages for target languages {target_languages}:")
    print(target_avg)
    print("\nAverages for other languages:")
    print(other_avg)

    # Start building the LaTeX table
    latex = r"""\begin{table}[h!]
\centering
\small
\def\arraystretch{1.05}
\resizebox{\columnwidth}{!}{
\begin{tabular}{l|c:ccc|ccc||c:ccc|ccc}
\toprule
\multirow{2}{*}{\textbf{ISO}} & \multicolumn{7}{c}{\textbf{mBERT}} & \multicolumn{7}{c}{\textbf{XLM-R}} \\
\cmidrule(lr){2-8} \cmidrule(lr){9-15}
 &  \multicolumn{1}{c}{} & \multicolumn{3}{c}{\textbf{ConceptNet}} & \multicolumn{3}{c}{\textbf{Glot}} & \multicolumn{1}{c}{} & \multicolumn{3}{c}{\textbf{ConceptNet}} & \multicolumn{3}{c}{\textbf{Glot}} \\
 \cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){10-12} \cmidrule(lr){13-15}
 & \textbf{Base} & \textbf{S\_bn} & \textbf{LoRA} & \textbf{S\_bn\_inv} & \textbf{S\_bn} & \textbf{LoRA} & \textbf{S\_bn\_inv} & \textbf{Base} & \textbf{S\_bn} & \textbf{LoRA} & \textbf{S\_bn\_inv} & \textbf{S\_bn} & \textbf{LoRA} & \textbf{S\_bn\_inv} \\
 \midrule
"""

    # Add rows for each language with its values for each model and adapter
    for _, row in merged_data.iterrows():
        language = row['Language']
        mbert_conceptnet_baseline = row.get('mBERT_ConceptNet_Baseline', '-')
        mbert_conceptnet_seq_bn = row.get('mBERT_ConceptNet_Seq_bn', '-')
        mbert_conceptnet_lora = row.get('mBERT_ConceptNet_LoRA', '-')
        mbert_conceptnet_seq_bn_inv = row.get('mBERT_ConceptNet_Seq_bn_inv', '-')
        mbert_glot_seq_bn = row.get('mBERT_Glot_Seq_bn', '-')
        mbert_glot_lora = row.get('mBERT_Glot_LoRA', '-')
        mbert_glot_seq_bn_inv = row.get('mBERT_Glot_Seq_bn_inv', '-')
        
        xlm_r_conceptnet_baseline = row.get('XLM-R_ConceptNet_Baseline', '-')
        xlm_r_conceptnet_seq_bn = row.get('XLM-R_ConceptNet_Seq_bn', '-')
        xlm_r_conceptnet_lora = row.get('XLM-R_ConceptNet_LoRA', '-')
        xlm_r_conceptnet_seq_bn_inv = row.get('XLM-R_ConceptNet_Seq_bn_inv', '-')
        xlm_r_glot_seq_bn = row.get('XLM-R_Glot_Seq_bn', '-')
        xlm_r_glot_lora = row.get('XLM-R_Glot_LoRA', '-')
        xlm_r_glot_seq_bn_inv = row.get('XLM-R_Glot_Seq_bn_inv', '-')
        
        latex += f"{language} & {mbert_conceptnet_baseline} & {mbert_conceptnet_seq_bn} & {mbert_conceptnet_lora} & {mbert_conceptnet_seq_bn_inv} & "
        latex += f"{mbert_glot_seq_bn} & {mbert_glot_lora} & {mbert_glot_seq_bn_inv} & "
        latex += f"{xlm_r_conceptnet_baseline} & {xlm_r_conceptnet_seq_bn} & {xlm_r_conceptnet_lora} & {xlm_r_conceptnet_seq_bn_inv} & "
        latex += f"{xlm_r_glot_seq_bn} & {xlm_r_glot_lora} & {xlm_r_glot_seq_bn_inv} \\\\\n"

    # Close the LaTeX table
    latex += r"""\bottomrule
\end{tabular}
}
\caption{F1 scores comparison across different adapters for mBERT and XLM-R in ConceptNet and Glot.}
\label{tab:model_comparison_ner}
\end{table}
"""
    
    # Save the LaTeX to a file
    with open(output_file, 'w') as f:
        f.write(latex)
    
    print(f"LaTeX table saved to {output_file}")
    return latex

if __name__ == "__main__":
    # Define the paths to the files for each model, data source, and adapters
    conceptnet_files = {
        "mBERT_ConceptNet_Baseline": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/conceptnet_mbert_baseline_results.csv',
        "mBERT_ConceptNet_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/conceptnet_mbert_seq_bn_results.csv',
        "mBERT_ConceptNet_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/conceptnet_mbert_lora_results.csv',
        "mBERT_ConceptNet_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/conceptnet_mbert_seq_bn_inv_results.csv',
        
        "XLM-R_ConceptNet_Baseline": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/glot_xlm-r_baseline_results.csv',
        "XLM-R_ConceptNet_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/conceptnet_xlm-r_seq_bn_results.csv',
        "XLM-R_ConceptNet_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/conceptnet_xlm-r_lora_results.csv',
        "XLM-R_ConceptNet_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/conceptnet_xlm-r_seq_bn_inv_results.csv'
    }
    
    glot_files = {
        "mBERT_Glot_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/glot_mbert_seq_bn_results.csv',
        "mBERT_Glot_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/glot_mbert_lora_results.csv',
        "mBERT_Glot_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/glot_mbert_seq_bn_inv_results.csv',
        
        "XLM-R_Glot_Seq_bn": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/glot_xlm-r_seq_bn_results.csv',
        "XLM-R_Glot_LoRA": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/glot_xlm-r_lora_results.csv',
        "XLM-R_Glot_Seq_bn_inv": '/netscratch/dgurgurov/thesis/downstream_tasks/ner/glot_xlm-r_seq_bn_inv_results.csv'
    }
    
    # Combine ConceptNet and Glot files for table generation
    model_files = {**conceptnet_files, **glot_files}
    
    # Generate and print the LaTeX table
    latex_table = generate_latex_table(**model_files)
