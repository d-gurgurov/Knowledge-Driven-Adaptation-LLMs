# External Knowledge-Driven Adaptation of Multilingual Language Models for Low-Resource Languages

[![arXiv](https://img.shields.io/badge/arXiv-2502.10140v1-b31b1b.svg)](https://arxiv.org/abs/2502.10140)

This repository contains the code, data, and models (soon!) associated with the paper "Small Models, Big Impact: Efficient Corpus and Graph-Based Adaptation of Small Multilingual Language Models for Low-Resource Languages," focusing on enhancing multilingual Language Models (mLMs) for low-resource languages (LRLs). LRLs often face significant challenges in natural language processing (NLP) due to the scarcity of data. We explore parameter-efficient adaptation of small mLMs for LRLs, comparing adapters, continued pre-training, and large-scale LM prompting. Our findings show that: (1) limited adaptation data (≤1 GB text or a few MB of KG data) provides significant gains, with Sequential Bottleneck excelling in MLM and Invertible Bottleneck in downstream tasks; (2) smaller mLMs like XLM-R outperform massive LLMs (e.g., GPT-3.5, LLaMA-3) for LRLs; (3) pre-training data size strongly influences performance, with adaptation yielding diminishing returns for the languages with the extensive inclusion in a mode's pre-training data.

## [Overview](pplx://action/followup)

This research develops and experiments with language adapters trained on both structured and unstructured data sources:

-   **[Structured Knowledge](pplx://action/followup):** ConceptNet, a multilingual knowledge graph providing valuable relational knowledge across 304 languages. We convert ConceptNet triples into natural language sentences using predefined predicates.
-   **[Unstructured Data](pplx://action/followup):** GlotCC-V1, a large-scale multilingual corpus derived from CommonCrawl, emphasizing LRLs and providing high-quality text in 1,000 languages.

We systematically investigate parameter-efficient adapter-based methods for adapting mLMs to LRLs, evaluating three architectures: Sequential Bottleneck, Invertible Bottleneck, and Low-Rank Adaptation.

## [Key Contributions](pplx://action/followup)

*   Demonstrated that limited adaptation data yields significant gains for LRLs (up to 1 GB of free text or a few MB of KG data).
*   Showed the effectiveness of smaller mLMs, such as XLM-R, for LRLs, outperforming both few-shot prompting and adaptation of massive SoTA LLMs.
*   Analyzed 30 LRLs, revealing a direct relationship between pre-training and adaptation data size and performance, with adaptation data providing diminishing returns for languages with larger pre-training data coverage.

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

## [Results](pplx://action/followup)

Key results demonstrating the effectiveness of adapter-based adaptation:

*   **[MLM](pplx://action/followup):** Glot-based adapters substantially improved pseudo-perplexity, particularly for mBERT, with the Seq\_bn adapter achieving the largest reduction.
*   **[Downstream Tasks (TC, SA, NER)](pplx://action/followup):** Adapter-based methods often matched or outperformed full fine-tuning while using far fewer parameters. Invertible Bottleneck adapters slightly outperformed other methods on downstream tasks due to better embedding alignment and larger parameter counts.
*   Smaller mLMs (XLM-R) proved more effective for LRLs than massive LLMs like LLaMA-3.

Refer to Tables 1 and 3 in the paper for detailed results across all tasks and models.

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

*   This work was supported by [DisAI](https://disai.eu/), improving scientific excellence and creativity in combating disinformation with artificial intelligence and language technologies, a Horizon Europe-funded project under GA No. 101079164.
*   We thank the open-source community for providing valuable resources and tools.

## [Contact](pplx://action/followup)

For questions or issues, please contact daniil.gurgurov@dfki.de.
