# import dependencies
import argparse
import numpy as np
import os
import json
import random
import torch
from transformers import set_seed

from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    DataCollatorForTokenClassification,
    AutoModel,
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
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs")
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


def compute_metrics(p, label_names):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metric = load_metric("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }
    for k in results.keys():
        if k not in flattened_results.keys():
            flattened_results[k+"_f1"] = results[k]["f1"]
    return flattened_results


def main():

    # set the seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    def tokenize_adjust_labels(all_samples_per_split):
        tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], max_length=512, is_split_into_words=True)
        #tokenized_samples is not a datasets object so this alone won't work with Trainer API, hence map is used 
        #so the new keys [input_ids, labels (after adjustment)]
        #can be added to the datasets dict for each train test validation split
        total_adjusted_labels = []
        print(len(tokenized_samples["input_ids"]))
        for k in range(0, len(tokenized_samples["input_ids"])):
            prev_wid = -1
            word_ids_list = tokenized_samples.word_ids(batch_index=k)
            existing_label_ids = all_samples_per_split["ner_tags"][k]
            i = -1
            adjusted_label_ids = []
        
            for wid in word_ids_list:
                if(wid is None):
                    adjusted_label_ids.append(-100)
                elif(wid!=prev_wid):
                    i = i + 1
                    adjusted_label_ids.append(existing_label_ids[i])
                    prev_wid = wid
                else:
                    label_name = label_names[existing_label_ids[i]]
                    adjusted_label_ids.append(existing_label_ids[i])
                
            total_adjusted_labels.append(adjusted_label_ids)
        tokenized_samples["labels"] = total_adjusted_labels
        tokenized_samples["labels"] = [list(map(int, x)) for x in tokenized_samples["labels"]]

        return tokenized_samples

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

    # prepare data
    dataset = load_dataset("wikiann", languages_mapping[args.language])
    label_names = dataset["train"].features["ner_tags"].feature.names
    id2label = {id_: label for id_, label in enumerate(label_names)}
    label2id = {label: id_ for id_, label in enumerate(label_names)}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_dataset = dataset.map(tokenize_adjust_labels, batched=True)

    # prepare model
    config = AutoConfig.from_pretrained(args.model_name, id2label=id2label, label2id=label2id)
    model = AutoAdapterModel.from_pretrained(args.model_name, config=config)

    def find_latest_checkpoint(adapter_base_path: str, language_code: str) -> str:
        """
        Finds the latest checkpoint directory for the given language.
        """
        lang_adapter_path = os.path.join(adapter_base_path, language_code)
        checkpoints = [d for d in os.listdir(lang_adapter_path) if d.startswith("checkpoint")]
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))  # Sort by checkpoint number
        latest_checkpoint = checkpoints[-1]  # Get the latest checkpoint
        return os.path.join(lang_adapter_path, latest_checkpoint)

    if args.language_adapter=="yes":
        # load language adapter and add classification head for topic classification
        if args.adapter_source=="conceptnet":
            adapter_dir = find_latest_checkpoint(args.adapter_dir, languages_mapping[args.language])
        if args.adapter_source=="glot":
            adapter_dir = find_latest_checkpoint(args.adapter_dir, args.language)
        lang_adapter_config = AdapterConfig.load(adapter_dir + "/mlm/adapter_config.json")
        model.load_adapter(adapter_dir + "/mlm", config=lang_adapter_config, load_as="lang_adapter", with_head=False)

        model.add_adapter("ner")
        model.add_tagging_head("ner", num_labels=len(label_names), id2label=id2label)
        model.train_adapter(["ner"])
        
        # set the active adapter and enable training only for the classification head
        model.active_adapters = Stack("lang_adapter", "ner")
    else:
        # no language adapter used
        model.add_adapter("ner")
        model.add_tagging_head("ner", num_labels=len(label_names), id2label=id2label)
        model.train_adapter(["ner"])
    
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
    
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, label_names)
    )

    # train model
    trainer.train()

    # test model
    test_results = trainer.evaluate(tokenized_dataset["test"])
    output_file_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(output_file_path, "w") as f:
        json.dump(test_results, f)

if __name__ == "__main__":
    main()