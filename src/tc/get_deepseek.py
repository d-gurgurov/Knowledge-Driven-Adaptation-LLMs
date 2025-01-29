from datasets import load_dataset
from sklearn.metrics import f1_score
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

def evaluate_language(language, model, tokenizer):
    dataset = load_dataset('Davlan/sib200', language)
    categories = dataset["train"].unique("category")
    
    # Ensure all categories are lowercased for consistency
    categories = [cat.lower() for cat in categories]
    
    prompt_template = (
        "Classify the following text into one of these categories: "
        "{categories}. Text: {input_text} Answer:"
    )

    results = []
    true_labels = []
    predicted_labels = []
    
    for sample in tqdm(dataset['test'], desc=f"Evaluating {language}", leave=False):
        input_text = sample['text']
        true_label = sample['category'].lower()  # Ensure lowercase for matching
        
        # Create the prompt by replacing placeholders
        prompt = prompt_template.format(
            categories=", ".join(categories), 
            input_text=input_text
        )
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate prediction with memory-efficient settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                use_cache=False  # Prevent memory spikes
            )  
        
        # Decode prediction and clean up text
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Extract the part of the response after "Answer:"
        match = re.search(r"Answer:\s*(.*)", predicted_text, re.IGNORECASE)
        if match:
            extracted_text = match.group(1).strip()  # Extract the part after "Answer:"
        else:
            extracted_text = "invalid"  # Default to invalid if no "Answer:" is found
        
        # Match the extracted text to the closest category
        matched_categories = [
            cat for cat in categories if cat in extracted_text.lower()
        ]
        
        if len(matched_categories) == 1:  # Accept only single-label predictions
            predicted_label = matched_categories[0]
        else:
            predicted_label = "invalid"  # Label as invalid if multiple categories are matched or no match
        
        # Store results
        results.append({
            "input": input_text,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "raw_response": predicted_text
        })
        
        # Add only valid predictions to the lists
        if predicted_label != "invalid":
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
        
        # Free memory
        del inputs, outputs
        torch.cuda.empty_cache()
    
    # Save results to JSONL file
    with open(f'llama_70b/{language}_results.jsonl', 'w') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')
    
    # Calculate F1 score only for valid predictions
    if true_labels and predicted_labels:
        f1 = f1_score(true_labels, predicted_labels, average='macro', labels=categories)
    else:
        f1 = 0.0  # Set F1 score to 0 if there are no valid predictions
    
    return f1, results


def main():
    languages = [
        'mlt_Latn', 'amh_Ethi', 'uzn_Latn', 'sun_Latn', 'cym_Latn', 
        'ckb_Arab', 'mkd_Cyrl', 'kat_Geor', 'slk_Latn', 'ell_Grek', 
        'tha_Thai', 'azj_Latn', 'lvs_Latn', 'slv_Latn', 'heb_Hebr',
        'ron_Latn', 'dan_Latn', 'urd_Arab', 'sin_Sinh', 'yor_Latn', 
        'swh_Latn', 'uig_Arab', 'bod_Tibt', 'mar_Deva', 'jav_Latn', 
        'npi_Deva', 'zsm_Latn', 'bul_Cyrl', 'tel_Telu', 'ben_Beng'
    ]

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    # deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
    # deepseek-ai/DeepSeek-R1-Distill-Llama-70B
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatically spread across available GPUs
    )

    results = {}
    for lang in tqdm(languages, desc="Overall Progress"):
        f1, _ = evaluate_language(lang, model, tokenizer)
        results[lang] = f1
        print(f"F1-score for {lang}: {f1}")

    with open('llama_70b/sib200_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
