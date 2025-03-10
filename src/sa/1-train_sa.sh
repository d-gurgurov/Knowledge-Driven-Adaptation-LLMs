#!/bin/bash

pip install evaluate adapters
pip install transformers[torch] torch
pip install -U scikit-learn==1.5.2

huggingface-cli login --token blabla # --add-to-git-credential

# Define variables
languages=('cym_Latn') # 'mlt_Latn' 'sin_Sinh' 'uig_Arab' 'swh_Latn' 

# 'amh_Ethi' 'uzn_Latn' 'sun_Latn' 'cym_Latn' 'mar_Deva' 
#            'ckb_Arab' 'mkd_Cyrl' 'kat_Geor' 'slk_Latn' 'ell_Grek' 
#            'tha_Thai' 'azj_Latn' 'lvs_Latn' 'slv_Latn' 'heb_Hebr'
#            'ron_Latn' 'dan_Latn' 'urd_Arab' 'sin_Sinh' 'yor_Latn' 
#            'swh_Latn' 'uig_Arab' 'bod_Tibt' 'jav_Latn' 
#            'npi_Deva' 'zsm_Latn' 'bul_Cyrl' 'tel_Telu' 'ben_Beng'

# Configurable parameters
source="glot"   # Can be set to "glot"
model="llama3"         # FacebookAI/xlm-roberta-base google-bert/bert-base-multilingual-cased
configuration="baseline"  # Can be set to "seq_bn", "seq_bn_inv", or "lora"

# Directory base path
base_dir="/netscratch/dgurgurov/thesis/downstream_tasks/sa"
model_dir="meta-llama/Meta-Llama-3-8B" # google-bert/bert-base-multilingual-cased FacebookAI/xlm-roberta-base


# Loop over each language and seed
for lang in "${languages[@]}"; do
  for seed in 1 2 3; do # 1
    # Define output directory
    output_dir="${base_dir}/${source}/${model}/${configuration}/${lang}/${seed}"

    # Create the output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Run the script with the specified arguments
    python train_sa.py \
      --language "$lang" \
      --output_dir "$output_dir" \
      --adapter_dir "/netscratch/dgurgurov/thesis/src/mlm/lang_adapters/$source/$model/$configuration" \
      --model_name "$model_dir" \
      --learning_rate 2e-5 \
      --num_train_epochs 25 \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 8 \
      --evaluation_strategy "epoch" \
      --save_strategy "epoch" \
      --weight_decay 0.01 \
      --seed "$seed" \
      --language_adapter "no" \
      --adapter_source "$source" \
      --configuration "no_finetune"
  done
done
