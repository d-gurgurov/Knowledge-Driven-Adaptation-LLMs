#!/bin/bash

pip install adapters evaluate scikit-learn
pip install accelerate -U

languages=('mlt-Latn' 'uzn-Latn' 'sun-Latn' 'cym-Latn' 'mar-Deva' 
                          'ckb-Arab' 'mkd-Cyrl' 'kat-Geor' 'slk-Latn' 'ell-Grek' 
                          'tha-Thai' 'azj-Latn' 'lvs-Latn' 'slv-Latn' 'heb-Hebr'
                          'ron-Latn' 'dan-Latn' 'urd-Arab' 'sin-Sinh' 'yor-Latn' 
                          'swh-Latn' 'uig-Arab' 'bod-Tibt' 'amh-Ethi' 'jav-Latn' 
                          'npi-Deva' 'zsm-Latn' 'bul-Cyrl' 'tel-Telu' 'ben-Beng')
adapter_sources=("glot")

export NCCL_DEBUG=WARN

# model choices: FacebookAI/xlm-roberta-base google-bert/bert-base-multilingual-cased

for source in "${adapter_sources[@]}"; do
    echo "Using adapter source: $source"
    
    for lang in "${languages[@]}"; do
        echo "Training for language: $lang"
        
        torchrun --nproc_per_node=2 run_mlm.py \
            --model_name_or_path google-bert/bert-base-multilingual-cased \
            --train_file "/ds/text/LangAdapters/data/$source/train_glot_${lang}.csv" \
            --validation_file "/ds/text/LangAdapters/data/$source/val_glot_${lang}.csv" \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --do_train \
            --do_eval \
            --logging_dir "./model_fine-tune/glot/mbert/${lang}/logs" \
            --output_dir "./model_fine-tune/glot/mbert/${lang}" \
            --learning_rate 1e-4 \
            --overwrite_output_dir \
            --load_best_model_at_end=True \
            --save_total_limit=1 \
            --evaluation_strategy='steps' \
            --save_strategy='steps' \
            --max_steps=100000 \
            --line_by_line
        
        echo "Training for $lang completed."
    done
done
