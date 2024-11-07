#!/bin/bash

# Define variables
languages=('amh_Ethi' 'uzn_Latn' 'sun_Latn' 'cym_Latn' 'mar_Deva' 
           'ckb_Arab' 'mkd_Cyrl' 'kat_Geor' 'slk_Latn' 'ell_Grek' 
           'tha_Thai' 'azj_Latn' 'lvs_Latn' 'slv_Latn' 'heb_Hebr'
           'ron_Latn' 'dan_Latn' 'urd_Arab' 'sin_Sinh' 'yor_Latn' 
           'swh_Latn' 'uig_Arab' 'bod_Tibt' 'mlt_Latn' 'jav_Latn' 
           'npi_Deva' 'zsm_Latn' 'bul_Cyrl' 'tel_Telu' 'ben_Beng')

# Configurable parameters
source="conceptnet"   # Can be set to "glot"
model="mbert"         # FacebookAI/xlm-roberta-base google-bert/bert-base-multilingual-cased
configuration="baseline"  # Can be set to "seq_bn", "seq_bn_inv", or "lora"

# Directory base path
base_dir="/ds/text/LangAdapters/downstream_tasks/ner"

# Loop over each language and seed
for lang in "${languages[@]}"; do
  for seed in 1 2 3; do
    # Define output directory
    output_dir="${base_dir}/${source}/${model}/${configuration}/${lang}/${seed}"

    # Create the output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Run the script with the specified arguments
    python train_ner.py \
      --language "$lang" \
      --output_dir "$output_dir" \
      --adapter_dir "/ds/text/LangAdapters/lang_adapters/$source/$model/$configuration/${lang}" \
      --model_name "google-bert/bert-base-multilingual-cased" \
      --learning_rate 1e-4 \
      --num_train_epochs 100 \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 32 \
      --evaluation_strategy "epoch" \
      --save_strategy "no" \
      --weight_decay 0.01 \
      --seed "$seed" \
      --language_adapter "yes" \
      --adapter_source "$source"
  done
done
