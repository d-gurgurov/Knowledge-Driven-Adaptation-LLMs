#!/bin/bash

pip install adapters
pip install evaluate
pip install -U scikit-learn==1.5.2
pip install transformers

huggingface-cli login --token blabla --add-to-git-credential

# Define variables
languages=('mlt_Latn' 'sin_Sinh' 'uig_Arab' 'swh_Latn' 'cym_Latn') # 
# languages=('mlt_Latn' 'amh_Ethi' 'uzn_Latn' 'sun_Latn' 'cym_Latn' 
#            'ckb_Arab' 'mkd_Cyrl' 'kat_Geor' 'slk_Latn' 'ell_Grek' 
#            'tha_Thai' 'azj_Latn' 'lvs_Latn' 'slv_Latn' 'heb_Hebr'
#            'ron_Latn' 'dan_Latn' 'urd_Arab' 'sin_Sinh' 'yor_Latn' 
#            'swh_Latn' 'uig_Arab' 'bod_Tibt' 'mar_Deva' 'jav_Latn' 
#            'npi_Deva' 'zsm_Latn' 'bul_Cyrl' 'tel_Telu' 'ben_Beng')

# Configurable parameters
source="glot"   # Can be set to "glot"
model="llama3"         
configuration="seq_bn_inv"  # Can be set to "seq_bn", "seq_bn_inv", or "lora"

# Directory base path
base_dir="/netscratch/dgurgurov/thesis/downstream_tasks/tc"
# fine-tuned mbert and xlm-r = "/ds/text/LangAdapters/model_fine-tune/$model"
# or base SLMs = FacebookAI/xlm-roberta-base google-bert/bert-base-multilingual-cased
# or base LLaMa2 = /netscratch/dgurgurov/thesis/src/mlm/llama_hf
model_dir="meta-llama/Meta-Llama-3-8B"

# Loop over each language and seed
for lang in "${languages[@]}"; do
  for seed in 1 2 3; do
    # Define output directory
    output_dir="${base_dir}/${source}/${model}/${configuration}/${lang}/${seed}"

    # Create the output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Run the script with the specified arguments
    python train_tc.py \
      --language "$lang" \
      --output_dir "$output_dir" \
      --adapter_dir "/netscratch/dgurgurov/thesis/src/mlm/lang_adapters/$source/$model/$configuration" \
      --model_name "$model_dir" \
      --learning_rate 2e-5 \
      --num_train_epochs 20 \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 16 \
      --evaluation_strategy "epoch" \
      --save_strategy "epoch" \
      --weight_decay 0.01 \
      --seed "$seed" \
      --language_adapter "yes" \
      --adapter_source "$source" \
      --configuration "no_finetune"
  done
done
