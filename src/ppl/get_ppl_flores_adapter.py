import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from evaluate import load
from adapters import AdapterConfig, AutoAdapterModel, AdapterTrainer 
import pandas as pd 

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Path to the model."})

@dataclass
class DataTrainingArguments:
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Max evaluation samples."})
    mlm_probability: float = field(default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss."})
    line_by_line: bool = field(default=False, metadata={"help": "Whether to treat each line as a separate sequence."})
    pad_to_max_length: bool = field(default=False, metadata={"help": "Whether to pad all samples to max length."})

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    
    # Load accuracy metric
    metric = load("accuracy")
    accuracy = metric.compute(predictions=preds, references=labels)
    
    return {"accuracy": accuracy["accuracy"], "predictions": preds}

def find_latest_checkpoint(adapter_base_path: str, language_code: str) -> str:
    """
    Finds the latest checkpoint directory for the given language.
    """
    lang_adapter_path = os.path.join(adapter_base_path, language_code)
    checkpoints = [d for d in os.listdir(lang_adapter_path) if d.startswith("checkpoint")]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  # Sort by checkpoint number
    latest_checkpoint = checkpoints[-1]  # Get the latest checkpoint
    return os.path.join(lang_adapter_path, latest_checkpoint)

def main():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Define model and language codes
    model_name = 'FacebookAI/xlm-roberta-base'
    language_codes_glotcc = ['tel_Telu', 'ben_Beng', 'lvs_Latn', 'mlt_Latn', 'amh_Ethi', 'uzn_Latn', 
                             'sun_Latn', 'cym_Latn', 'mar_Deva', 'ckb_Arab', 'mkd_Cyrl', 
                             'kat_Geor', 'slk_Latn', 'ell_Grek', 'tha_Thai', 'azj_Latn', 
                             'slv_Latn', 'heb_Hebr', 'ron_Latn', 'dan_Latn', 'urd_Arab', 
                             'sin_Sinh', 'yor_Latn', 'swh_Latn', 'uig_Arab', 'bod_Tibt', 
                             'jav_Latn', 'npi_Deva', 'zsm_Latn', 'bul_Cyrl']

    results_summary = []

    # Load model with adapter support
    model = AutoAdapterModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Adapter setup
    adapter_type = "seq_bn"
    adapter_dir = f"/ds/text/LangAdapters/lang_adapters/glot/xlm-r/{adapter_type}"  # Base path for adapters

    # Loop through each language and evaluate
    for language_code in language_codes_glotcc:
        try:
            latest_checkpoint = find_latest_checkpoint(adapter_dir, language_code)
            adapter_config_path = os.path.join(latest_checkpoint, "mlm", "adapter_config.json")
            lang_adapter_config = AdapterConfig.load(adapter_config_path)

            # Load the adapter and set it as active
            adapter_path = os.path.join(latest_checkpoint, "mlm")
            model.load_adapter(adapter_path, config=lang_adapter_config, load_as="lang_adapter", with_head=True)
            model.set_active_adapters("lang_adapter")
            logger.info(f"Loaded adapter for {language_code}.")

            # Load the FLORES-200 dataset
            dataset = load_dataset("facebook/flores", name=language_code)
            eval_dataset = dataset['devtest']  # Use devtest set for evaluation

            # Tokenize the dataset
            def tokenize_function(examples):
                return tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=512)

            tokenized_datasets = eval_dataset.map(tokenize_function, batched=True)

            # Limit to max_eval_samples if specified
            data_args = DataTrainingArguments()
            if data_args.max_eval_samples is not None:
                tokenized_datasets = tokenized_datasets.select(range(data_args.max_eval_samples))

            # Define data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm_probability=data_args.mlm_probability,
                pad_to_multiple_of=8 if data_args.line_by_line else None,
            )

            # Define training arguments
            training_args = TrainingArguments(
                output_dir='./results',
                per_device_eval_batch_size=64,
                do_eval=True,
            )

            # Create Trainer
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                eval_dataset=tokenized_datasets,
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                data_collator=data_collator,
            )

            # Evaluate the model
            metrics = trainer.evaluate()

            # Calculate perplexity
            perplexity = math.exp(metrics["eval_loss"]) if "eval_loss" in metrics else float("inf")
            logger.info(f"Perplexity for {language_code}: {perplexity:.2f}")
            logger.info(f"Accuracy for {language_code}: {metrics.get('eval_accuracy'):.2f}")

            # Append results for the current language
            results_summary.append({
                "language_code": language_code,
                "accuracy": metrics.get("eval_accuracy"),
                "perplexity": perplexity
            })
        
        except Exception as e:
            logger.error(f"Error processing language {language_code}: {str(e)}")

    # Save the summary to a CSV file
    results_df = pd.DataFrame(results_summary)
    summary_output_file = "./average_pseudo_perplexities_summary.csv"
    results_df.to_csv(summary_output_file, index=False)
    logger.info(f"Results saved to {summary_output_file}")

if __name__ == "__main__":
    main()
