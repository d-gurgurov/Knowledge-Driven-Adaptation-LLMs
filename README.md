# External Knowledge-Driven Adaptation of Multilingual Language Models for Low-Resource Languages

This repository contains the code and data associated with my master's thesis, which focuses on enhancing multilingual Language Models (LMs) for low-resource languages (LRLs). LRLs often face significant challenges in natural language processing (NLP) due to the scarcity of data and the linguistic diversity that characterizes these languages. This work aims to address these challenges by integrating external knowledge into models like mBERT and XLM-R using adapter-based techniques.

## Overview

The core of this research lies in developing and experimenting with language adapters trained on both structured and unstructured data sources:
- **Structured Knowledge:** ConceptNet, a semantic network that provides valuable relational knowledge.
- **Unstructured Data:** GlotCC, a large-scale corpus of multilingual web text.

### Key Components

1. **Language Adapters:** Specialized adapters are trained to capture language-specific knowledge from ConceptNet and GlotCC.
2. **Adapter Fusion:** During inference, these adapters are dynamically combined using Adapter Fusion to leverage the strengths of both knowledge sources.
3. **Task-Specific Fine-Tuning:** Task-specific adapters are used to fine-tune models for Sentiment Analysis (SA) and Named Entity Recognition (NER) tasks.
4. **Modular Design:** The approach allows for flexible experimentation with different adapter architectures, including Sequential Bottleneck, Invertible Layers, and LoRA, to enhance model performance across 30 diverse LRLs.

## Results and Evaluation

The evaluation of these methods is conducted on datasets representing a wide array of LRLs. Various objective functions and adapter configurations are explored to determine the most effective strategies for improving NLP tools for underrepresented languages.

## Repository Structure

- TO BE ADDED
- `code/`: Contains scripts and Jupyter notebooks for training and evaluating language adapters and task-specific models.
- `data/`: Includes the datasets used in this study, along with preprocessing scripts.
- `results/`: Evaluation results, including performance metrics and visualizations.
- `docs/`: Documentation and additional resources.

## Citation

```bibtex
@misc{gurgurov2025smallmodelsbigimpact,
      title={Small Models, Big Impact: Efficient Corpus and Graph-Based Adaptation of Small Multilingual Language Models for Low-Resource Languages}, 
      author={Daniil Gurgurov and Ivan Vykopal and Josef van Genabith and Simon Ostermann},
      year={2025},
      eprint={2502.10140},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.10140}, 
}
```
