import pandas as pd
import torch
from abc import ABC
from math import log, exp
from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedModel
from tqdm import tqdm
import random

class Metric(ABC):
    def __init__(self, model: PreTrainedModel, device: torch.device) -> None:
        self.device = device  # Store the device (CPU/GPU)

    def __call__(self, sentences: list[str]) -> dict[str, list[float] | float]:
        raise NotImplementedError

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
            tokenized_sentence: torch.Tensor = self.tokenizer.encode(sentence, return_tensors="pt", truncation=True, max_length=512).to(self.device)  # Move input to GPU
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

def compute_pseudo_perplexity_from_csv(model_name: str, csv_file: str, column_name: str, output_file: str, num_sentences: int = 1000, seed: int = 42) -> float:
    """
    Computes the pseudo-perplexity for a subset of sentences in the specified column of a CSV file and saves the results.

    :param model_name: The name of the pretrained model to use.
    :param csv_file: The path to the input CSV file.
    :param column_name: The name of the column containing the sentences.
    :param output_file: The path to save the output CSV file with pseudo-perplexities.
    :param num_sentences: Number of sentences to sample for pseudo-perplexity calculation.
    :param seed: Random seed for reproducibility.
    :returns: The average pseudo-perplexity for the sampled sentences.
    """
    # Set the random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Load the CSV file
    df = pd.read_csv(csv_file)
    df = df.dropna()

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Filter sentences and ensure they are within the model's maximum sequence length
    filtered_sentences = df[column_name].tolist()

    # Sort the filtered sentences to ensure consistency before sampling
    filtered_sentences.sort()

    # Sample a subset of sentences
    if len(filtered_sentences) > num_sentences:
        random.seed(seed)  # Reset the seed before sampling
        sentences = random.sample(filtered_sentences, num_sentences)
    else:
        sentences = filtered_sentences

    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the pseudo-perplexity metric
    model = AutoModelForMaskedLM.from_pretrained(model_name)
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
    language_codes_glot500 = ['mlt_Latn', 'amh_Ethi', 'uzb_Latn', 'sun_Latn', 'cym_Latn', 'mar_Deva', 
                            'kur_Arab', 'mkd_Cyrl', 'kat_Geor', 'slk_Latn', 'ell_Grek', 
                            'tha_Thai', 'aze_Latn', 'lvs_Latn', 'slv_Latn', 'heb_Hebr', 
                            'ron_Latn', 'dan_Latn', 'urd_Arab', 'sin_Sinh', 'yor_Latn', 
                            'swa_Latn', 'uig_Arab', 'bod_Tibt', 'jav_Latn', 
                            'nep_Deva', 'msa_Latn', 'bul_Cyrl', 'tel-Telu', 'ben-Beng']

    model_name = "google-bert/bert-base-multilingual-cased"  # Example model google-bert/bert-base-multilingual-cased FacebookAI/xlm-roberta-base
    results_summary = []

    for language_code in language_codes_glot500:
        csv_file = f"/netscratch/dgurgurov/thesis/data/glot/test_glot_{language_code}.csv" 
        output_file = f"/netscratch/dgurgurov/thesis/results/glot/mbert/{language_code}_pseudo_perplexity_results.csv" 
        average_pseudo_perplexity = compute_pseudo_perplexity_from_csv(model_name, csv_file, "text", output_file)
        results_summary.append({"language_code": language_code, "average_pseudo_perplexity": average_pseudo_perplexity})

    # Save the summary of average pseudo-perplexities for all languages
    results_df = pd.DataFrame(results_summary)
    summary_output_file = "/netscratch/dgurgurov/thesis/results/glot/mbert/average_pseudo_perplexities_summary.csv"
    results_df.to_csv(summary_output_file, index=False)

    print(f"Summary of average pseudo-perplexities saved to {summary_output_file}.")
