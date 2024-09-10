import pandas as pd

def read_data(file_path):
    # Read the CSV file into a pandas DataFrame
    return pd.read_csv(file_path)

def generate_markdown_table(output_file='model_comparison.md', **data_files):
    # Read and merge all the data files provided as arguments
    merged_data = None
    for model_name, file_path in data_files.items():
        data = read_data(file_path)
        data = data.rename(columns={"average_pseudo_perplexity": f"{model_name}"})
        
        # Round values to two decimal places
        data[f'{model_name}'] = data[f'{model_name}'].round(2)
        
        if merged_data is None:
            merged_data = data
        else:
            merged_data = pd.merge(merged_data, data, on='language')
    
    # Create the markdown table
    markdown = "| Model/Language | " + " | ".join(merged_data['language'].tolist()) + " |\n"
    markdown += "| --- " * (len(merged_data.columns)) + "|\n"
    
    # Add rows for each model and its adapters
    for model_name in data_files.keys():
        row = [model_name] + merged_data[f'{model_name}'].tolist()
        # Format the row values to have 2 decimal places
        row = [model_name] + [f"{item:.2f}" for item in merged_data[f'{model_name}'].tolist()]
        markdown += "| " + " | ".join(row) + " |\n"
    
    # Save the markdown to a file
    with open(output_file, 'w') as f:
        f.write(markdown)
    
    print(f"Markdown table saved to {output_file}")
    return markdown

if __name__ == "__main__":
    # Define the paths to the files for each model and adapters
    model_files = {
        "mBERT": '/netscratch/dgurgurov/thesis/results/conceptnet/mbert/baseline/average_pseudo_perplexities_summary.csv',
        "mBERT_LoRA": '/netscratch/dgurgurov/thesis/results/conceptnet/mbert/lora/average_pseudo_perplexities_summary.csv',
        "mBERT_Seq_bn": '/netscratch/dgurgurov/thesis/results/conceptnet/mbert/seq_bn/average_pseudo_perplexities_summary.csv',
        "mBERT_Seq_bn_inv": '/netscratch/dgurgurov/thesis/results/conceptnet/mbert/seq_bn_inv/average_pseudo_perplexities_summary.csv',
        "XLM-R": '/netscratch/dgurgurov/thesis/results/conceptnet/xlm-r/baseline/average_pseudo_perplexities_summary.csv',
        "XLM-R_LoRA": '/netscratch/dgurgurov/thesis/results/conceptnet/xlm-r/lora/average_pseudo_perplexities_summary.csv',
        "XLM-R_Seq_bn": '/netscratch/dgurgurov/thesis/results/conceptnet/xlm-r/seq_bn/average_pseudo_perplexities_summary.csv',
        "XLM-R_Seq_bn_inv": '/netscratch/dgurgurov/thesis/results/conceptnet/xlm-r/seq_bn_inv/average_pseudo_perplexities_summary.csv'
    }
    
    # Generate and print the markdown table
    markdown_table = generate_markdown_table(**model_files)
