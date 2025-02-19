# Small Models, Big Impact

[![arXiv](https://img.shields.io/badge/arXiv-2502.10140v1-b31b1b.svg)](https://arxiv.org/abs/2502.10140)

This repository contains the code, data, and models (soon!) associated with the paper "Small Models, Big Impact: Efficient Corpus and Graph-Based Adaptation of Small Multilingual Language Models for Low-Resource Languages," focusing on enhancing multilingual Language Models (mLMs) for low-resource languages (LRLs). LRLs often face significant challenges in natural language processing (NLP) due to the scarcity of data. 

We explore parameter-efficient adaptation of small mLMs for LRLs, comparing adapters, continued pre-training, and large-scale LM prompting. Our findings show that: 

- (1) limited adaptation data (≤1 GB text or a few MB of KG data) provides significant gains, with Sequential Bottleneck excelling in MLM and Invertible Bottleneck in downstream tasks;
- (2) smaller mLMs like XLM-R outperform massive LLMs (e.g., GPT-3.5, LLaMA-3) for LRLs;
- (3) pre-training data size strongly influences performance, with adaptation yielding diminishing returns for the languages with the extensive inclusion in a mode's pre-training data.

## [Overview](pplx://action/followup)

This research develops and experiments with language adapters trained on both structured and unstructured data sources:

-   **[Structured Knowledge](pplx://action/followup):** ConceptNet, a multilingual knowledge graph providing valuable relational knowledge across 304 languages. We convert ConceptNet triples into natural language sentences using predefined predicates.
-   **[Unstructured Data](pplx://action/followup):** GlotCC-V1, a large-scale multilingual corpus derived from CommonCrawl, emphasizing LRLs and providing high-quality text in 1,000 languages.

We systematically investigate parameter-efficient adapter-based methods for adapting mLMs to LRLs, evaluating three architectures: Sequential Bottleneck, Invertible Bottleneck, and Low-Rank Adaptation.


## [Experimental Setup](pplx://action/followup)

*   **[Languages](pplx://action/followup):** 30 diverse LRLs were selected, including Thai, Romanian, Bulgarian, and others (see Table 5 in the paper's Appendix B for the full list and details).
*   **[Data Preprocessing](pplx://action/followup):** ConceptNet triples were converted into natural language sentences. GlotCC data was limited to 1GB per language (if exceeding this limit) and cleaned.
*   **[Training Details](pplx://action/followup):**
    *   Language adapters were trained on mBERT and XLM-R using MLM with GlotCC and ConceptNet data.
    *   For LLaMA-3-8B, GlotCC data was used with the Seq\_bn\_inv architecture and CLM objective for a subset of 5 languages.
    *   Training consisted of up to 100,000 steps for GlotCC and 25,000 steps for ConceptNet, with a batch size of 16 and a learning rate of 1e-4.
*   **[Evaluation Tasks](pplx://action/followup):**
    *   Masked Language Modeling (MLM): Evaluated using the FLORES-200 devtest set.
    *   Topic Classification (TC): Evaluated using the 7-class SIB-200 dataset.
    *   Sentiment Analysis (SA): Evaluated using binary-class datasets from multiple sources.
    *   Named Entity Recognition (NER): Evaluated using the WikiANN dataset.

## [Citation](pplx://action/followup)

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

## [Acknowledgements](pplx://action/followup)

*   This work was supported by [DisAI](https://disai.eu/).
*   We thank the open-source community for providing valuable resources and tools.

## [Contact](pplx://action/followup)

For questions or issues, please contact daniil.gurgurov@dfki.de.
