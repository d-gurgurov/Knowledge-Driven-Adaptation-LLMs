# import dependencies
import argparse
import evaluate
import numpy as np
import os
import json
import random
import torch
from transformers import set_seed

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig, AutoModel,
    TrainingArguments, Trainer,
)
from adapters import AutoAdapterModel, AdapterConfig, AdapterTrainer
from adapters.composition import Stack

# useful functions
def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a model for a sentiment analysis task.")
    parser.add_argument("--language", type=str, default="", help="Language at hand")
    parser.add_argument("--output_dir", type=str, default="./training_output", help="Output directory for training results")
    parser.add_argument("--adapter_dir", type=str, default="", help="Directory containing the pre-trained adapter checkpoint")
    parser.add_argument("--adapter_config", type=str, default="", help="Directory containing the pre-trained adapter config")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased", help="Name of the pre-trained model")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Batch size per device during evaluation")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="Evaluation strategy during training")
    parser.add_argument("--save_strategy", type=str, default="no", help="Saving strategy during training")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimization")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--language_adapter", type=str, default="yes", help="Whether to use language adapter")
    parser.add_argument("--adapter_source", type=str, default="conceptnet", help="Adapter source")
    return parser.parse_args()

args = parse_arguments()

def encode_batch(examples):
    """Encodes a batch of input data using the model tokenizer."""
    all_encoded = {"input_ids": [], "attention_mask": [], "labels": []}
    if "bert" in str(args.model_name):
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
    elif "xlm-r" in str(args.model_name):
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

    if "category" in examples:
        examples["label"] = examples.pop("category")

    for text, label in zip(examples["text"], examples["label"]):
        encoded = tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        all_encoded["input_ids"].append(encoded["input_ids"])
        all_encoded["attention_mask"].append(encoded["attention_mask"])
        all_encoded["labels"].append(label)
    
    return all_encoded

def preprocess_dataset(dataset):
    dataset = dataset.map(encode_batch, batched=True)
    dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    return dataset

def calculate_f1_on_test_set(trainer, test_dataset, tokenizer):
    print("Calculating F1 score on the test set...")
    test_predictions = trainer.predict(test_dataset)

    f1_metric = evaluate.load("f1")
    test_metrics = {
        "f1": f1_metric.compute(
            predictions=np.argmax(test_predictions.predictions, axis=-1),
            references=test_predictions.label_ids,
            average="macro",
        )["f1"],
    }

    print("Test F1 score:", test_metrics["f1"])
    return test_metrics


def main():

    # se the seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    # Define language mapping
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

    languages_mapping_hf = {
        'amh_Ethi': 'Amharic',
        'azj_Latn': 'Azerbaijani',
        'ben_Beng': 'Bengali',
        'bod_Tibt': 'Tibetan',
        'bul_Cyrl': 'Bulgarian',
        'ckb_Arab': 'Kurdish',
        'cym_Latn': 'Welsh',
        'dan_Latn': 'Danish',
        'ell_Grek': 'Greek',
        'heb_Hebr': 'Hebrew',
        'jav_Latn': 'Javanese',
        'kat_Geor': 'Georgian',
        'lvs_Latn': 'Latvian',
        'mar_Deva': 'Marathi',
        'mlt_Latn': 'Maltese',
        'mkd_Cyrl': 'Macedonian',
        'npi_Deva': 'Nepali',
        'ron_Latn': 'Romanian',
        'sin_Sinh': 'Sinhala',
        'slk_Latn': 'Slovak',
        'slv_Latn': 'Slovenian',
        'sun_Latn': 'Sundanese',
        'swh_Latn': 'Swahili',
        'tel_Telu': 'Telugu',
        'tha_Thai': 'Thai',
        'uig_Arab': 'Uyghur',
        'urd_Arab': 'Urdu',
        'uzn_Latn': 'Uzbek',
        'yor_Latn': 'Yoruba',
        'zsm_Latn': 'Indonesian'
    }

    def find_latest_checkpoint(adapter_base_path: str, language_code: str) -> str:
        """
        Finds the latest checkpoint directory for the given language.
        """
        language_code=language_code.replace("_", "-")
        lang_adapter_path = os.path.join(adapter_base_path, language_code)
        checkpoints = [d for d in os.listdir(lang_adapter_path) if d.startswith("checkpoint")]
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  # Sort by checkpoint number
        latest_checkpoint = checkpoints[-1]  # Get the latest checkpoint
        return os.path.join(lang_adapter_path, latest_checkpoint)

    # prepare data
    dataset = load_dataset(f"dgurgurov/{languages_mapping_hf[args.language].lower()}_sa")
    print("Dataset loaded for", languages_mapping_hf[args.language].lower())

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    
    if "bert" in str(args.model_name):
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
    elif "xlm-r" in str(args.model_name):
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

    train_dataset = preprocess_dataset(train_dataset)
    val_dataset = preprocess_dataset(val_dataset)
    test_dataset = preprocess_dataset(test_dataset)

    # prepare model
    config = AutoConfig.from_pretrained(args.model_name)
    print("Model used: ", args.model_name)

    if args.model_name!="google-bert/bert-base-multilingual-cased" or "FacebookAI/xlm-roberta-base":
        local_name=find_latest_checkpoint(args.model_name, args.language)
        print(local_name)
        config = AutoConfig.from_pretrained(local_name)
        model = AutoAdapterModel.from_pretrained(local_name, config=config)
    else:
        model = AutoAdapterModel.from_pretrained(args.model_name, config=config)

    if args.language_adapter=="yes":
        # load language adapter and add classification head for topic classification
        if args.adapter_source=="conceptnet":
            adapter_dir = find_latest_checkpoint(args.adapter_dir, languages_mapping[args.language])
        if args.adapter_source=="glot":
            adapter_dir = find_latest_checkpoint(args.adapter_dir, args.language)
        lang_adapter_config = AdapterConfig.load(adapter_dir + "/mlm/adapter_config.json")
        model.load_adapter(adapter_dir + "/mlm", config=lang_adapter_config, load_as="lang_adapter", with_head=False)

        model.add_adapter("sentiment_classification")
        model.add_classification_head("sentiment_classification", num_labels=2)
        model.train_adapter(["sentiment_classification"])
        
        # set the active adapter and enable training only for the classification head
        model.active_adapters = Stack("lang_adapter", "sentiment_classification")
    else:
        # no language adapter used
        model.add_adapter("sentiment_classification")
        model.add_classification_head("sentiment_classification", num_labels=2)
        model.train_adapter(["sentiment_classification"])
    
    print(model.adapter_summary())

    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.evaluation_strategy,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        warmup_ratio=0.05,
        logging_steps=20,
    )

    f1_metric = evaluate.load("f1")
    
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: {
                "f1": f1_metric.compute(
                    predictions=np.argmax(pred.predictions, axis=-1),
                    references=pred.label_ids,
                    average="macro",
                )["f1"],
            },
    )

    trainer.train()

    calculate_f1_on_test_set(trainer, test_dataset, tokenizer)

    output_file_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(output_file_path, "w") as json_file:
        json.dump(calculate_f1_on_test_set(trainer, test_dataset, tokenizer), json_file, indent=2)


if __name__ == "__main__":
    main()





