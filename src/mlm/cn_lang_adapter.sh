#!/bin/bash

pip install adapters evaluate
pip install accelerate -U

languages=('am' 'uz' 'su' 'cy' 'mr' 'te' 'ku' 'mk' 'bn' 'ka' 'sk' 'el' 'th' 'az' 'lv' 'sl' 'he' 'ro' 'da' 'ur' 'si' 'yo' 'sw' 'ug' 'bo' 'mt' 'jv' 'ne' 'ms' 'bg')
adapter_sources=("conceptnet")

export NCCL_DEBUG=WARN

# model choices: FacebookAI/xlm-roberta-base google-bert/bert-base-multilingual-cased

for source in "${adapter_sources[@]}"; do
    echo "Using adapter source: $source"
    
    for lang in "${languages[@]}"; do
        echo "Training for language: $lang"
        
        python run_mlm.py \
            --model_name_or_path google-bert/bert-base-multilingual-cased \
            --train_file "/netscratch/dgurgurov/thesis/data/$source/train_cn_${lang}.csv" \
            --validation_file "/netscratch/dgurgurov/thesis/data/$source/val_cn_${lang}.csv" \
            --per_device_train_batch_size 64 \
            --per_device_eval_batch_size 64 \
            --do_train \
            --do_eval \
            --logging_dir "/netscratch/dgurgurov/thesis/lang_adapters/conceptnet/mbert/seq_bn_inv/${lang}/logs" \
            --output_dir "/netscratch/dgurgurov/thesis/lang_adapters/conceptnet/mbert/seq_bn_inv/${lang}" \
            --train_adapter \
            --learning_rate 1e-5 \
            --adapter_config seq_bn_inv \
            --overwrite_output_dir \
            --load_best_model_at_end=True \
            --save_total_limit=1 \
            --evaluation_strategy='steps' \
            --save_strategy='steps' \
            --max_steps=25000 \
            --line_by_line
        
        echo "Training for $lang completed."
    done
done
