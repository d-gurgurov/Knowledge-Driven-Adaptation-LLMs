import pandas as pd
import logging
import math
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from evaluate import load

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

def main():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Define model and language codes
    model_name = "google-bert/bert-base-multilingual-cased"
    language_codes_glotcc = ['tel_Telu', 'ben_Beng', 'lvs_Latn', 'mlt_Latn', 'amh_Ethi', 'uzn_Latn', 'sun_Latn', 'cym_Latn', 
                             'mar_Deva', 'ckb_Arab', 'mkd_Cyrl', 'kat_Geor', 'slk_Latn', 'ell_Grek', 'tha_Thai', 'azj_Latn', 
                             'slv_Latn', 'heb_Hebr', 'ron_Latn', 'dan_Latn', 'urd_Arab', 'sin_Sinh', 'yor_Latn', 'swh_Latn', 
                             'uig_Arab', 'bod_Tibt', 'jav_Latn', 'npi_Deva', 'zsm_Latn', 'bul_Cyrl']

    results_summary = []

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Loop through each language and evaluate
    for language_code in language_codes_glotcc:
        # Load dataset for the specific language
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
        trainer = Trainer(
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
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except KeyError:
            logger.error(f"Evaluation metrics do not contain 'eval_loss' for {language_code}.")
            perplexity = None
        except OverflowError:
            perplexity = float("inf")

        # Append results for the current language
        results_summary.append({
            "language_code": language_code,
            "accuracy": metrics.get("eval_accuracy"),
            "perplexity": perplexity
        })
        logger.info(f"Results for {language_code}: {results_summary[-1]}")

    # Save the summary to a CSV file
    results_df = pd.DataFrame(results_summary)
    summary_output_file = "./average_pseudo_perplexities_summary.csv"
    results_df.to_csv(summary_output_file, index=False)
    logger.info(f"Results saved to {summary_output_file}")

if __name__ == "__main__":
    main()
