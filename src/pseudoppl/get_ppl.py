import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
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
    validation_file: Optional[str] = field(default=None, metadata={"help": "Validation data file."})
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
    metric = load("accuracy")
    return metric.compute(predictions=preds, references=labels)

def main():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load model and tokenizer
    model_args = ModelArguments(model_name_or_path='bert-base-multilingual-cased')
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path)

    # Load the dataset
    data_args = DataTrainingArguments(validation_file='/netscratch/dgurgurov/thesis/data/glot/test_glot_mlt_Latn.csv')
    raw_datasets = load_dataset('csv', data_files={'validation': data_args.validation_file})

    # Drop rows with missing values
    def remove_missing_values(example):
        return example['text'] is not None and example['text'].strip() != ""

    raw_datasets = raw_datasets.filter(remove_missing_values)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # Prepare evaluation dataset
    eval_dataset = tokenized_datasets['validation']
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if data_args.line_by_line else None,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=8,
        do_eval=True,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator,  # Include data collator
    )

    # Evaluate the model
    metrics = trainer.evaluate()

    print(metrics)

    # Calculate perplexity
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except KeyError:
        logger.error("Evaluation metrics do not contain 'eval_loss'.")
        return
    except OverflowError:
        perplexity = float("inf")

    logger.info(f"Perplexity: {perplexity:.2f}")
    logger.info(f"Accuracy: {metrics['eval_accuracy']:.2f}")

if __name__ == "__main__":
    main()