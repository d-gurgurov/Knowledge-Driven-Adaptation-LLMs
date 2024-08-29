import pandas as pd
from abc import ABC
from math import log, exp
from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedModel
import torch
from tqdm import tqdm
import random
from adapters import AdapterConfig, AutoAdapterModel  
import os

class Metric(ABC):
    def __init__(self, model: PreTrainedModel, device: torch.device) -> None:
        self.device = device  # Store the device (CPU/GPU)

    def __call__(self, sentences: list[str]) -> dict[str, list[float] | float]:
        raise NotImplementedError
    
def find_latest_checkpoint(adapter_base_path: str, language_code: str) -> str:
    """
    Finds the latest checkpoint directory for the given language.

    :param adapter_base_path: Base path where adapters are stored.
    :param language_code: Language code for which to find the adapter.
    :returns: The path to the latest checkpoint directory.
    """
    lang_adapter_path = os.path.join(adapter_base_path, language_code)
    checkpoints = [d for d in os.listdir(lang_adapter_path) if d.startswith("checkpoint")]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  # Sort by checkpoint number
    latest_checkpoint = checkpoints[-1]  # Get the latest checkpoint
    return os.path.join(lang_adapter_path, latest_checkpoint)

class PseudoPerplexity(Metric):
    """
    Returns the pseudo-perplexity for a list of sentences. Only designed for masked language models.
    """

    def __init__(self, model: PreTrainedModel, device: torch.device):
        super().__init__(model, device)
        self.model: PreTrainedModel = model.to(self.device)  # Move model to GPU if available
        self.tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)

    def __call__(self, sentences: list[str]) -> dict[str, list[float] | float]:
        """
        Returns the perplexities of each sentence as well as the mean perplexity across all sentences using the model outputs.
        """
        assert len(sentences) > 0

        pseudo_perplexities: list[float] = []
        for sentence in tqdm(sentences, desc="Computing pseudo-perplexity"):
            tokenized_sentence: torch.Tensor = self.tokenizer.encode(sentence, return_tensors="pt").to(self.device)  # Move input to GPU
            num_tokens = tokenized_sentence.shape[-1]

            pseudo_log_likelihood = self.pseudo_log_likelihood(tokenized_sentence)
            pseudo_perplexity = exp(-1 / num_tokens * pseudo_log_likelihood)
            pseudo_perplexities.append(pseudo_perplexity)

        average_pseudo_perplexity: float = sum(pseudo_perplexities) / len(pseudo_perplexities)
        return {"values": pseudo_perplexities, "average": average_pseudo_perplexity}

    def pseudo_log_likelihood(self, tokenized_sentence: torch.Tensor) -> float:
        """
        Calculates the pseudo-log-likelihood (PLL) for a sentence under a model and tokenizer by masking every token in the sentence,
        one by one, and adding up all log probabilities of the masked token appearing at its position.
        """
        pseudo_log_likelihood = 0
        for token_position, original_token_id in enumerate(tokenized_sentence.squeeze()):
            masked_sentence = tokenized_sentence.clone().to(self.device)  # Ensure masked sentence is on the GPU
            masked_sentence[:, token_position] = self.tokenizer.mask_token_id
            with torch.no_grad():
                output = self.model(input_ids=masked_sentence)
                logits: torch.Tensor = output.logits.squeeze()
            probabilities = logits[token_position].softmax(dim=0)
            probability = probabilities[original_token_id]
            pseudo_log_likelihood += log(probability)

        return pseudo_log_likelihood

def compute_pseudo_perplexity_from_csv(model_name: str, adapter_dir: str, csv_file: str, column_name: str, output_file: str, num_sentences: int = 1000, seed: int = 42) -> float:
    """
    Computes the pseudo-perplexity for a subset of sentences in the specified column of a CSV file and saves the results.

    :param model_name: The name of the pretrained model to use.
    :param adapter_dir: The directory containing the language-specific adapter.
    :param csv_file: The path to the input CSV file.
    :param column_name: The name of the column containing the sentences.
    :param output_file: The path to save the output CSV file with pseudo-perplexities.
    :param num_sentences: Number of sentences to sample for pseudo-perplexity calculation.
    :param seed: Random seed for reproducibility.
    :returns: The average pseudo-perplexity for the sampled sentences.
    """
    # Set the random seed for reproducibility
    random.seed(seed)

    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Extract the sentences from the specified column
    sentences = df[column_name].tolist()

    # Sample a subset of sentences
    if len(sentences) > num_sentences:
        sentences = random.sample(sentences, num_sentences)

    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model with adapters
    model = AutoAdapterModel.from_pretrained(model_name)  # Load the model with adapter support
    adapter_config_path = f"{adapter_dir}/mlm/adapter_config.json"
    lang_adapter_config = AdapterConfig.load(adapter_config_path)  # Load the adapter configuration
    adapter_dir = f"{adapter_dir}/mlm"
    model.load_adapter(adapter_dir, config=lang_adapter_config, load_as="lang_adapter", with_head=True)  # Load the adapter
    model.set_active_adapters("lang_adapter")  # Set the adapter as active
    print(model.adapter_summary())
    model.to(device)  # Move the model to GPU if available

    # Initialize the pseudo-perplexity metric
    pseudo_perplexity_metric = PseudoPerplexity(model, device)

    # Compute pseudo-perplexity
    results = pseudo_perplexity_metric(sentences)

    # Add the results back to the DataFrame for the sampled sentences
    sampled_df = pd.DataFrame({column_name: sentences})
    sampled_df["pseudo_perplexity"] = results["values"]

    # Save the DataFrame with pseudo-perplexities to a new CSV file
    sampled_df.to_csv(output_file, index=False)

    print(f"Pseudo-perplexity values saved to {output_file}.")
    print(f"Average pseudo-perplexity: {results['average']}")

    return results['average']

if __name__ == "__main__":
    model_name = "google-bert/bert-base-multilingual-cased"  # Example model: google-bert/bert-base-multilingual-cased FacebookAI/xlm-roberta-base
    
    model = "mbert"
    adapter = "seq_bn_inv"

    base_data_path = "/netscratch/dgurgurov/thesis/data/conceptnet/test_cn_"  # Base path for input files
    base_results_path = f"/netscratch/dgurgurov/thesis/results/conceptnet/{model}/{adapter}/pseudo_perplexity_"  # Base path for result files
    adapter_base_path = f"/netscratch/dgurgurov/thesis/lang_adapters/conceptnet/{model}/{adapter}/"  # Base path for adapters
    summary_output_file = f"/netscratch/dgurgurov/thesis/results/conceptnet/{model}/{adapter}/average_pseudo_perplexities_summary.csv"

    languages_to_process = ['am', 'uz', 'su', 'cy', 'mr', 'te', 'ku', 'mk', 'bn', 'ka', 'sk', 'el', 'th', 'az', 'lv', 'sl', 
                            'he', 'ro', 'da', 'ur', 'si', 'yo', 'sw', 'ug', 'bo', 'mt', 'jv', 'ne', 'ms', 'bg']

    summary_results = []

    for language_code in languages_to_process:
        csv_file = f"{base_data_path}{language_code}.csv" 
        output_file = f"{base_results_path}{language_code}.csv"
        adapter_dir = find_latest_checkpoint(adapter_base_path, language_code)  # Find the latest checkpoint directory

        # Compute and save pseudo-perplexities for a sample of sentences
        average_pseudo_perplexity = compute_pseudo_perplexity_from_csv(model_name, adapter_dir, csv_file, "text", output_file)
        
        # Store the average pseudo-perplexity along with the language code
        summary_results.append({"language": language_code, "average_pseudo_perplexity": average_pseudo_perplexity})

    # Save the summary of average pseudo-perplexities for all languages
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(summary_output_file, index=False)

    print(f"Summary of average pseudo-perplexity for all languages saved to {summary_output_file}.")
