import pandas as pd
import torch
from abc import ABC
from math import log, exp
from transformers import AutoTokenizer, PreTrainedModel
from adapters import AdapterConfig, AutoAdapterModel # type: ignore
from tqdm import tqdm
import random
from datasets import load_dataset
import os

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

def find_latest_checkpoint(adapter_base_path: str, language_code: str) -> str:
    """
    Finds the latest checkpoint directory for the given language.
    """
    lang_adapter_path = os.path.join(adapter_base_path, language_code)
    checkpoints = [d for d in os.listdir(lang_adapter_path) if d.startswith("checkpoint")]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  # Sort by checkpoint number
    latest_checkpoint = checkpoints[-1]  # Get the latest checkpoint
    return os.path.join(lang_adapter_path, latest_checkpoint)

def compute_pseudo_perplexity_with_adapter(model_name: str, adapter_dir: str, language_code: str, output_file: str, seed: int = 42) -> float:
    """
    Computes the pseudo-perplexity for a subset of sentences from the FLORES-200 dataset and saves the results, using a language adapter.

    :param model_name: The name of the pretrained model to use.
    :param adapter_dir: Directory where the adapters are stored.
    :param language_code: The language code for the FLORES-200 dataset.
    :param output_file: Path to save the output CSV file with pseudo-perplexities.
    :param seed: Random seed for reproducibility.
    :returns: The average pseudo-perplexity for the sampled sentences.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Load the FLORES-200 dataset
    dataset = load_dataset("facebook/flores", name=language_code)
    sentences = dataset['devtest']['sentence']  # type: ignore

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model with adapter support
    model = AutoAdapterModel.from_pretrained(model_name)

    # Find the latest checkpoint for the adapter
    latest_checkpoint = find_latest_checkpoint(adapter_dir, language_code)
    adapter_config_path = os.path.join(latest_checkpoint, "mlm", "adapter_config.json")
    lang_adapter_config = AdapterConfig.load(adapter_config_path)

    # Load the adapter and set it as active
    adapter_path = os.path.join(latest_checkpoint, "mlm")
    model.load_adapter(adapter_path, config=lang_adapter_config, load_as="lang_adapter", with_head=True)
    model.set_active_adapters("lang_adapter")
    print(model.adapter_summary())

    model.to(device)

    # Initialize the pseudo-perplexity metric
    pseudo_perplexity_metric = PseudoPerplexity(model, device)

    # Compute pseudo-perplexity
    results = pseudo_perplexity_metric(sentences)

    # Save results
    df = pd.DataFrame({"text": sentences})
    df["pseudo_perplexity"] = results["values"]
    df.to_csv(output_file, index=False)

    print(f"Pseudo-perplexity values saved to {output_file}.")
    print(f"Average pseudo-perplexity: {results['average']}")

    return results['average']

if __name__ == "__main__":
    model_name = "google-bert/bert-base-multilingual-cased"
    adapter_type = "seq_bn_inv"

    adapter_dir_base = f"/netscratch/dgurgurov/thesis/lang_adapters/glot/mbert/{adapter_type}"
    results_summary = []

    language_codes_glotcc = ['tel_Telu', 'ben_Beng', 'lvs_Latn', 'mlt_Latn', 'amh_Ethi', 
                            'uzn_Latn', 'sun_Latn', 'cym_Latn', 'mar_Deva', 
                            'ckb_Arab', 'mkd_Cyrl', 'kat_Geor', 'slk_Latn', 'ell_Grek', 
                            'tha_Thai', 'azj_Latn', 'slv_Latn', 'heb_Hebr', 
                            'ron_Latn', 'dan_Latn', 'urd_Arab', 'sin_Sinh', 'yor_Latn', 
                            'swh_Latn', 'uig_Arab', 'bod_Tibt', 'jav_Latn', 
                            'npi_Deva', 'zsm_Latn', 'bul_Cyrl']

    for language_code in language_codes_glotcc:
        output_file = f"/netscratch/dgurgurov/thesis/results/flores/mbert/{adapter_type}/{language_code}_pseudo_perplexity_results.csv"
        avg_pseudo_perplexity = compute_pseudo_perplexity_with_adapter(model_name, adapter_dir_base, language_code, output_file)
        results_summary.append({"language_code": language_code, "average_pseudo_perplexity": avg_pseudo_perplexity})

    # Save the summary
    results_df = pd.DataFrame(results_summary)
    summary_output_file = f"/netscratch/dgurgurov/thesis/results/flores/mbert/{adapter_type}/average_pseudo_perplexities_summary.csv"
    results_df.to_csv(summary_output_file, index=False)

    print(f"Summary saved to {summary_output_file}.")
