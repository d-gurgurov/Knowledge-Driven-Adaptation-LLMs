#!/bin/bash

pip install evaluate adapters seqeval
pip install -U scikit-learn==1.5.2

huggingface-cli login --token blabla --add-to-git-credential


# Define variables
all_languages=(
    'amh_Ethi' 'azj_Latn' 'ben_Beng' 'bod_Tibt' 'bul_Cyrl' 
    'ckb_Arab' 'cym_Latn' 'dan_Latn' 'ell_Grek' 'heb_Hebr' 
    'jav_Latn' 'kat_Geor' 'lvs_Latn' 'mar_Deva' 'mlt_Latn' 
    'mkd_Cyrl' 'npi_Deva' 'ron_Latn' 'sin_Sinh' 'slk_Latn' 
    'slv_Latn' 'sun_Latn' 'swh_Latn' 'tel_Telu' 'tha_Thai' 
    'uig_Arab' 'urd_Arab' 'uzn_Latn' 'yor_Latn' 'zsm_Latn'
)

all_languages=('mlt_Latn' 'sin_Sinh' 'uig_Arab' 'swh_Latn' 'cym_Latn')


# Configurable parameters
source="glot"   # Can be set to "glot"
model="llama3"         # FacebookAI/xlm-roberta-base google-bert/bert-base-multilingual-cased
configuration="seq_bn_inv"  # Can be set to "seq_bn", "seq_bn_inv", or "lora"

# Directory base path
base_dir="/netscratch/dgurgurov/thesis/downstream_tasks/ner"

# Loop over each language and seed
for lang in "${all_languages[@]}"; do
  for seed in 1 2 3; do
    # Define output directory
    output_dir="${base_dir}/${source}/${model}/${configuration}/${lang}/${seed}"

    # Create the output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Run the script with the specified arguments
    python train_ner.py \
      --language "$lang" \
      --output_dir "$output_dir" \
      --adapter_dir "/netscratch/dgurgurov/thesis/src/mlm/lang_adapters/$source/$model/$configuration" \
      --adapter_cn_dir "/ds/text/ConceptNet/lang_adapters/conceptnet/$model/$configuration" \
      --adapter_glot_dir "/netscratch/dgurgurov/thesis/src/mlm/lang_adapters/glot/$model/$configuration" \
      --model_name "meta-llama/Meta-Llama-3-8B" \
      --learning_rate 2e-5 \
      --num_train_epochs 25 \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 8 \
      --evaluation_strategy "epoch" \
      --save_strategy "epoch" \
      --weight_decay 0.01 \
      --seed "$seed" \
      --language_adapter "yes" \
      --adapter_source "$source" \
      --configuration "no_finetune"
  done
done
