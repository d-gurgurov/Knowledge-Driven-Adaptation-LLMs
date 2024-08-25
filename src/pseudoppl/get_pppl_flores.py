import pandas as pd
import torch
from abc import ABC
from math import log, exp
from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedModel
from tqdm import tqdm
import random
from datasets import load_dataset

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
        self.model: PreTrainedModel = model.to(self.device)  # type: ignore
        self.tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)

    def __call__(self, sentences: list[str]) -> dict[str, list[float] | float]:
        assert len(sentences) > 0

        pseudo_perplexities: list[float] = []
        for sentence in tqdm(sentences, desc="Computing pseudo-perplexity"):
            tokenized_sentence: torch.Tensor = self.tokenizer.encode(sentence, return_tensors="pt", truncation=True, max_length=512).to(self.device)  # type: ignore
            num_tokens = tokenized_sentence.shape[-1]

            pseudo_log_likelihood = self.pseudo_log_likelihood(tokenized_sentence)
            pseudo_perplexity = exp(-1 / num_tokens * pseudo_log_likelihood)
            pseudo_perplexities.append(pseudo_perplexity)

        average_pseudo_perplexity: float = sum(pseudo_perplexities) / len(pseudo_perplexities)
        return {"values": pseudo_perplexities, "average": average_pseudo_perplexity}

    def pseudo_log_likelihood(self, tokenized_sentence: torch.Tensor) -> float:
        pseudo_log_likelihood = 0
        for token_position, original_token_id in enumerate(tokenized_sentence.squeeze()):
            masked_sentence = tokenized_sentence.clone().to(self.device)
            masked_sentence[:, token_position] = self.tokenizer.mask_token_id  # type: ignore
            with torch.no_grad():
                output = self.model(input_ids=masked_sentence)
                logits: torch.Tensor = output.logits.squeeze()
            probabilities = logits[token_position].softmax(dim=0)
            probability = probabilities[original_token_id]
            pseudo_log_likelihood += log(probability)

        return pseudo_log_likelihood

def compute_pseudo_perplexity_from_flores(model_name: str, language_code: str, output_file: str, seed: int = 42) -> float:
    """
    Computes the pseudo-perplexity for a subset of sentences from the FLORES-200 dataset and saves the results.

    :param model_name: The name of the pretrained model to use.
    :param language_code: The language code for the FLORES-200 dataset.
    :param output_file: The path to save the output CSV file with pseudo-perplexities.
    :param num_sentences: Number of sentences to sample for pseudo-perplexity calculation.
    :param seed: Random seed for reproducibility.
    :returns: The average pseudo-perplexity for the sampled sentences.
    """
    # Set the random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Load the FLORES-200 dataset
    dataset = load_dataset("facebook/flores", name=language_code)
    print(dataset)
    sentences = dataset['devtest']['sentence']  # type: ignore

    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the pseudo-perplexity metric
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    pseudo_perplexity_metric = PseudoPerplexity(model, device)

    # Compute pseudo-perplexity
    results = pseudo_perplexity_metric(sentences)

    # Add the results back to the DataFrame for the sampled sentences
    df = pd.DataFrame({"text": sentences})
    df["pseudo_perplexity"] = results["values"]

    # Save the DataFrame with pseudo-perplexities to a new CSV file
    df.to_csv(output_file, index=False)

    print(f"Pseudo-perplexity values saved to {output_file}.")
    print(f"Average pseudo-perplexity: {results['average']}")

    return results['average']  # type: ignore


if __name__ == "__main__":
    language_codes_glot500 = ['tel_Telu', 'ben_Beng', 'lvs_Latn', 'mlt_Latn', 'amh_Ethi', 'uzb_Latn', 'sun_Latn', 'cym_Latn', 'mar_Deva', 
                            'ckb_Arab', 'mkd_Cyrl', 'kat_Geor', 'slk_Latn', 'ell_Grek', 
                            'tha_Thai', 'azj_Latn', 'slv_Latn', 'heb_Hebr', 
                            'ron_Latn', 'dan_Latn', 'urd_Arab', 'sin_Sinh', 'yor_Latn', 
                            'swh_Latn', 'uig_Arab', 'bod_Tibt', 'jav_Latn', 
                            'npi_Deva', 'zsm_Latn', 'bul_Cyrl']

    model_name = "FacebookAI/xlm-roberta-base"  # FacebookAI/xlm-roberta-base google-bert/bert-base-multilingual-cased
    results_summary = []

    for language_code in language_codes_glot500:
        output_file = f"/netscratch/dgurgurov/thesis/results/flores/xlm-r/{language_code}_pseudo_perplexity_results.csv" 
        average_pseudo_perplexity = compute_pseudo_perplexity_from_flores(model_name, language_code, output_file)
        results_summary.append({"language_code": language_code, "average_pseudo_perplexity": average_pseudo_perplexity})

    # Save the summary of average pseudo-perplexities for all languages
    results_df = pd.DataFrame(results_summary)
    summary_output_file = "/netscratch/dgurgurov/thesis/results/flores/xlm-r/average_pseudo_perplexities_summary.csv"
    results_df.to_csv(summary_output_file, index=False)

    print(f"Summary of average pseudo-perplexities saved to {summary_output_file}.")