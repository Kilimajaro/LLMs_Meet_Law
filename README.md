# When Large Language Models Meet Law: Research Framework, Technological Evolution, and Ethical Challenges

## Overview

This repository contains materials for the first comprehensive review of Large Language Models (LLMs) in legal domains, featuring an innovative dual taxonomy integrating legal reasoning frameworks and professional ontologies. The work systematically examines:

- Historical evolution from symbolic AI to transformer-based LLMs
- Technical breakthroughs in context scalability, knowledge integration, and evaluation rigor
- Ethical challenges including hallucination, explainability deficits, and jurisdictional adaptation
- Future research directions for next-generation legal AI systems

```bibtex
@article{shao2025large,
  title={When Large Language Models Meet Law: Research Framework, Technological Evolution, and Ethical Challenges},
  author={Shao, Peizhang and Xu, Linrui and Wang, Jinxi and Zhou, Wei and Wu, Xingyu},
  journal={ACM Computing Surveys},
  volume={xx},
  number={x},
  pages={xx--xx},
  year={2025}
}
```

## Table of Contents

- [Introduction](#introduction)
- [Evolution from Small Models to Large Models](#evolution-from-small-models-to-large-models)
- [LLM-enhanced Legal Reasoning](#llm-enhanced-legal-reasoning)
- [LLMs’ Integrations in Dispute Resolution Procedures](#llms-integrations-in-dispute-resolution-procedures)
- [The Collaboration of Technological Ethics and Legal Ethics](#the-collaboration-of-technological-ethics-and-legal-ethics)
- [Roadmap and Future Directions](#roadmap-and-future-directions)
- [Conclusion](#conclusion)

## Introduction

Innovative dual-lens taxonomy integrating legal reasoning and professional roles:

![alt text](image.png)

Core Components:​​

1. ​​Toulmin Argumentation Mapping:​​ Data→Warrant→Backing→Claim computational implementation
2. ​​Professional Role Mapping:​​ Judge/Lawyer/Litigant workflows in litigation/ADR contexts
3. ​​Ethical Co-regulation:​​ Technological ethics × Legal professional responsibilities

## Evolution from Small Models to Large Models

From symbolic systems to transformer-based LLMs in legal NLP:

![alt text](image-1.png)

Evolutionary Stages:​​

| Period             | Key                     | Developments                             | Representative Systems                   |
|--------------------|-------------------------|------------------------------------------|------------------------------------------|
| Foundations (2017) | Legal embeddings        | Domain-specific neural networks          | Law2Vec (Chalkidis et al.)               |
| Transition (2018-19)| Pretraining paradigms   | BERT integration for complex tasks       | Legal-BERT (Chalkidis et al.)            |
| Expansion (2020-21) | Transformer specialization | Domain-adapted NLP pipelines             | COLIEE frameworks (Nguyen et al.)        |
| Breakthrough (2022)| Holistic applications    | Long-document processing & jurisdiction adaptation | Lawformer (Xiao et al.)                  |
| Paradigm shift (2023-24)| Domain-optimized scaling | Mixture-of-experts & knowledge integration | ChatLaw (Cui et al.), LLM-GNN fusion (Wang et al.) |

## LLM-enhanced Legal Reasoning

Toulmin model implementation for legal task decomposition:

![alt text](image-2.png)

### 3.2: LLM-assisted Legal Data Processing

| **Category** |      Name                   | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |
| -------- | --------------------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ |
| Word Embedding         | Devlin et al. | [BERT: Pre-training of deep bidirectional transformers for language understanding](https://aclanthology.org/N19-1423.pdf)                 |NAACL      |2019      | https://github.com/google-research/bert     |
| Word Embedding         | Mikolov       | [Efficient estimation of word representations in vector space](https://www.khoury.northeastern.edu/home/vip/teach/DMcourse/4_TF_supervised/notes_slides/1301.3781.pdf)                                      |arxiv       |2013      |http://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt      |
| Word Embedding         | Pennington et al. | [Glove: Global vectors for word representation](https://aclanthology.org/D14-1162.pdf)                                                     |EMNLP       |2014      |http://nlp.stanford.edu/projects/glove/      |
| Sentence Embedding     | Hill et al.   | [Learning to understand phrases by embedding the dictionary](https://doi.org/10.1162/tacl_a_00080)                                        |Transactions of the Association for Computational Linguistics       |2016      |N/A      |
| Sentence Embedding     | Kiros et al.  | [Skip-thought vectors](https://proceedings.neurips.cc/paper/2015/hash/f442d33fa06832082290ad8544a8da27-Abstract.html)                                                                              |Advances in neural information processing systems       |2015      |https://github.com/ryankiros/skip-thoughts      |
| Coherence Modeling     | Logeswaran et al. | [Sentence ordering and coherence modeling using recurrent neural networks](https://ojs.aaai.org/index.php/AAAI/article/view/11997)                          |AAAI       |2018      |N/A      |
| Summarization          | Benedetto et al. | [Leveraging large language models for abstractive summarization of Italian legal news](https://link.springer.com/article/10.1007/s10506-025-09431-3)              |AI & Law       |2025      |N/A      |
| Summarization          | Deroy et al.  | [Applicability of large language models and generative models for legal case judgement summarization](https://link.springer.com/article/10.1007/s10506-024-09411-z) |AI & Law       |2024      |N/A      |
| Summarization          | Jain et al.   | [Summarization of Lengthy Legal Documents via Abstractive Dataset Building: An Extract-then-Assign Approach](https://www.sciencedirect.com/science/article/pii/S0957417423020730) |Expert Systems with Applications       |2024      |N/A      |
| Summarization          | Liu et al.    | [Low-resource court judgment summarization for common law systems](https://www.sciencedirect.com/science/article/pii/S0306457324001511)                                  |Information Processing & Management       |2024      |N/A      |
| Summarization          | Nguyen et al. | [Robust deep reinforcement learning for extractive legal summarization](https://link.springer.com/chapter/10.1007/978-3-030-92310-5_69)                             |ICONIP       |2021      |N/A      |
| Summarization          | Pont et al.   | [Legal Summarisation through LLMs: The PRODIGIT Project](https://arxiv.org/pdf/2308.04416)                                            |arxiv       |2023      |N/A      |
| Summarization Evaluation | Elaraby et al. | [Adding Argumentation into Human Evaluation of Long Document Abstractive Summarization: A Case Study on Legal Opinions](https://aclanthology.org/2024.humeval-1.3/) |HumEval       |2024      |N/A      |
| Summarization          | Mao et al.    | [Comparative Analysis of LLM-Generated Event Timeline Summarization for Legal Investigations](https://ieeexplore.ieee.org/abstract/document/10826063/)      |IEEE BigData       |2024      |N/A      |
| Text Processing        | Nguyen et al. | Transformer-based approaches for legal text processing                                            |       |      |      |
| Classification         | Chen et al.   | A comparative study of automated legal text classification using random forests and deep learning |       |      |      |
| Classification         | Liga et al.   | Fine-tuning GPT-3 for legal rule classification                                                   |       |      |      |
| Classification         | Prasad et al. | Exploring Large Language Models and Hierarchical Frameworks for Classification of Large Unstructured Legal Documents |       |      |      |
| Argumentation Mining   | Palau et al.  | Argumentation mining: the detection, classification and structure of arguments in text            |       |      |      |
| Textual Entailment     | Bilgin et al. | Exploring Prompting Approaches in Legal Textual Entailment                                        |       |      |      |
| Textual Entailment     | Nguyen et al. | Employing label models on ChatGPT answers improves legal text entailment performance               |       |      |      |
| Textual Entailment     | Reji et al.   | Enhancing LLM Performance on Legal Textual Entailment with Few-Shot CoT-based RAG                 |       |      |      |
| Classification         | Santosh et al. | Zero-shot transfer of article-aware legal outcome classification for european court of human rights cases |       |      |      |
| Prediction             | Liu et al.    | Augmenting legal judgment prediction with contrastive case relations                              |       |      |      |
| Element Recognition    | Hua et al.    | Mixture of Expert Large Language Model for Legal Case Element Recognition                         |       |      |      |
| NER                    | Smadu et al.  | Legal named entity recognition with multi-task domain adaptation                                   |       |      |      |

### 3.3: LLM-assisted Legal Backing Digiting

| **Category** |      Name                   | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |
| -------- | --------------------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ |
| Retrieval                 | Tran et al.   | Building legal case retrieval systems with lexical matching and summarization using a pre-trained phrase scoring model |       |      |      |
| Retrieval                 | Shao et al.   | BERT-PLI: Modeling paragraph-level interactions for legal case retrieval                         |       |      |      |
| Entailment                | Rabelo et al. | Combining similarity and transformer methods for case law entailment                              |       |      |      |
| Retrieval                 | Shao et al.   | BERT-based ensemble model for statute law retrieval and legal information entailment              |       |      |      |
| Pre-training              | Su et al.     | Caseformer:Pre-training for legal case retrieval                                                  |       |      |      |
| Retrieval                 | Kim et al.    | Legal information retrieval and entailment based on bm25, transformer and semantic thesaurus methods |       |      |      |
| Retrieval                 | Nguyen et al. | Attentive deep neural networks for legal document retrieval                                       |       |      |      |
| Retrieval                 | Yoshioka et al. | Hukb at the coliee 2022 statute law task                                                          |       |      |      |
| Entailment                | Wehnert et al. | Applying BERT embeddings to predict legal textual entailment                                      |       |      |      |
| Information Extraction    | Hudek et al.  | Information extraction/entailment of common law and civil code                                   |       |      |      |
| Pre-training Evaluation  | Zheng et al.  | When does pretraining help? assessing self-supervised learning for law and the casehold dataset    |       |      |      |
| Retrieval Framework       | Nguyen et al. | Retrieve-Revise-Refine: A novel framework for retrieval of concise entailing legal article set   |       |      |      |
| Retrieval                 | Nguyen et al. | Pushing the boundaries of legal information processing with integration of large language models   |       |      |      |
| Correspondence Modeling   | Ge et al.     | Learning fine-grained fact-article correspondence in legal cases                                  |       |      |      |
| Relevance Comprehension  | Li et al.     | Towards an In-Depth Comprehension of Case Relevance for Better Legal Retrieval                  |       |      |      |
| Health Information Retrieval | Milanese et al. | Fact-Driven Health Information Retrieval: Integrating LLMs and Knowledge Graphs to Combat Misinformation |       |      |      |
| Bias Analysis             | Cuconasu et al. | Do RAG Systems Suffer From Positional Bias?                                                       |       |      |      |
| Uncertainty Calibration   | Shi et al.    | Ambiguity detection and uncertainty calibration for question answering with large language models |       |      |      |
| Relevance Judgment       | Shao et al.   | Understanding relevance judgments in legal case retrieval                                         |       |      |      |
| Graph-based Retrieval     | Tang et al.   | CaseGNN: Graph Neural Networks for Legal Case Retrieval with Text-Attributed Graphs                |       |      |      |
| Provision Selection       | Wang et al.   | Causality-inspired legal provision selection with large language model-based explanation           |       |      |      |
| Data Augmentation         | Bui et al.    | Data Augmentation and Large Language Model for Legal Case Retrieval and Entailment                |       |      |      |
| Legal Information Processing | Nguyen et al. | NOWJ@ COLIEE 2024: leveraging advanced deep learning techniques for efficient and effective legal information processing |       |      |      |
| Ensemble Approaches       | Vuong et al.  | NOWJ at COLIEE 2023: Multi-task and Ensemble Approaches in Legal Information Processing           |       |      |      |
| Person Name Analysis      | Onaga et al.  | Contribution Analysis of Large Language Models and Data Augmentations for Person Names in Solving Legal Bar Examination at COLIEE 2023 |       |      |      |
| CoT Tuning                | Fujita et al. | LLM Tuning and Interpretable CoT: KIS Team in COLIEE 2024                                        |       |      |      |

### 3.4: LLM-assisted Legal Warrant Reasoning Generation

| **Category** |      Name                   | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |
| -------- | --------------------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ |
| Long-text Processor       | Xiao et al.   | Lawformer: A pre-trained language model for chinese legal long documents                          |       |      |      |
| Legal QA System           | Huang et al.  | Lawyer llama technical report                                                                     |       |      |      |
| Reasoning Framework       | Yue et al.    | Disc-lawllm: Fine-tuning large language models for intelligent legal services                     |       |      |      |
| Benchmark                 | Fei et al.    | Lawbench: Benchmarking legal knowledge of large language models                                    |       |      |      |
| Benchmark                 | Guha et al.   | Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models |       |      |      |
| MoE Architecture          | Cui et al.    | Chatlaw: A multi-agent collaborative legal assistant with knowledge graph enhanced mixture-of-experts large language model |       |      |      |
| Precedent Retrieval       | Wiratunga et al. | CBR-RAG: case-based reasoning for retrieval augmented generation in LLMs for legal question answering |       |      |      |
| Domain-specific LLM       | Fei et al.    | Internlm-law: An open source chinese legal large language model                                   |       |      |      |
| Text Analytics            | Ghosh et al.  | Human Centered AI for Indian Legal Text Analytics                                                |       |      |      |
| Hallucination Analysis    | Dahl et al.   | Large legal fictions: Profiling legal hallucinations in large language models                      |       |      |      |
| Factor Annotation         | Gray et al.   | Empirical legal analysis simplified: reducing complexity through automatic identification and evaluation of legally relevant factors |       |      |      |
| Legal Support             | Maree et al.  | Transforming legal text interactions: leveraging natural language processing and large language models for legal support in Palestinian cooperatives |       |      |      |
| Reasoning                | Deng et al.   | Syllogistic reasoning for legal judgment analysis                                                 |       |      |      |

### 3.5: LLM-assisted Legal Judgment Prediction with Qualifiers

| **Category** |      Name                   | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |
| -------- | --------------------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ |
| Prediction                | Strickson et al. | Legal judgement prediction for UK courts                                                          |       |      |      |
| Representation Learning  | Ma et al.     | Judgment Prediction Based on Case Life Cycle                                                     |       |      |      |
| Representation Learning  | Ma et al.     | Legal judgment prediction with multi-stage case representation learning in the real court setting |       |      |      |
| Judgement Prediction      | Prasad et al. | IRIT_IRIS_C at SemEval-2023 Task 6: A Multi-level Encoder-based Architecture for Judgement Prediction of Legal Cases and their Explanation |       |      |      |
| Element Extraction        | Lyu et al.    | Improving legal judgment prediction through reinforced criminal element extraction                |       |      |      |
| Dependency Learning       | Huang et al.  | Dependency learning for legal judgment prediction with a unified text-to-text transformer          |       |      |      |
| Multi-task Prediction     | Xu et al.     | Multi-task legal judgement prediction combining a subtask of the seriousness of charges          |       |      |      |
| Event Extraction          | Feng et al.   | Legal judgment prediction via event extraction with constraints                                  |       |      |      |
| Contrastive Learning      | Liu et al.    | Augmenting legal judgment prediction with contrastive case relations                             |       |      |      |
| Contrastive Learning      | Zhang et al.  | Contrastive learning for legal judgment prediction                                                |       |      |      |
| Knowledge Injection       | Gan et al.    | Judgment prediction via injecting legal knowledge into neural networks                            |       |      |      |
| Benchmark                 | Chalkidis et al. | LexGLUE: A benchmark dataset for legal language understanding in English                          |       |      |      |
| Benchmark                 | Niklaus et al. | Swiss-judgment-prediction: A multilingual legal judgment prediction benchmark                     |       |      |      |
| Benchmark                 | Hwang et al.  | A multi-task benchmark for korean legal language understanding and judgement prediction           |       |      |      |
| Statute Prediction        | Vats et al.   | Llms-the good, the bad or the indispensable?: A use case on legal statute prediction and legal judgment prediction on indian court cases |       |      |      |
| Evaluation                | Shui et al.   | A comprehensive evaluation of large language models on legal judgment prediction                 |       |      |      |
| Realistic Prediction      | Nigam et al.  | Rethinking legal judgement prediction in a realistic scenario in the era of large language models |       |      |      |
| Reasoning                | Deng et al.   | Enabling discriminative reasoning in llms for legal judgment prediction                           |       |      |      |
| Graph Learning            | Yangbin et al. | Legal Judgment Prediction with LLM and Graph Contrastive Learning Networks                         |       |      |      |
| Rule-enhanced Prediction  | Zhang et al.  | RLJP: Legal Judgment Prediction via First-Order Logic Rule-enhanced with Large Language Models    |       |      |      |
| Neuro-symbolic Framework  | Wei et al.    | An LLMs-based neuro-symbolic legal judgment prediction framework for civil cases                 |       |      |      |
| Framework                 | Wang et al.   | LegalReasoner: A Multi-Stage Framework for Legal Judgment Prediction via Large Language Models and Knowledge Integration |       |      |      |
| Survey                    | Cui et al.    | A survey on legal judgment prediction: Datasets, metrics, models and challenges                   |       |      |      |
| Survey                    | Medvedeva et al. | Rethinking the field of automatic prediction of court decisions                                  |       |      |      |

## LLMs’ Integrations in Dispute Resolution Procedures

### Litigation Workflows

| Category          | Name          | Paper                                                                                              | Venue | Year | Code |
|-------------------|---------------|---------------------------------------------------------------------------------------------------|-------|------|------|
| Reasoning Model   | Toulmin       | The uses of argument                                                                              |       |      |      |
| Case Law          | Goebel et al. | Overview and Discussion of the Competition on Legal Information, Extraction/Entailment(COLIEE) 2023 |       |      |      |
| Legal Ontologies  | Hart et al.   | The concept of law                                                                                |       |      |      |
| MoE Architecture   | Cui et al.    | Chatlaw: A multi-agent collaborative legal assistant with knowledge graph enhanced mixture-of-experts large language model |       |      |      |
| Information Retrieval | Nguyen et al. | Pushing the boundaries of legal information processing with integration of large language models   |       |      |      |
| Litigation Support    | Siino et al.    | Exploring LLMs Applications in Law: A Literature Review on Current Legal NLP Approaches             |       |      |      |
| MoE Architecture       | Cui et al.       | Chatlaw: Open-source legal large language model with integrated external knowledge bases         |       |      |      |
| Information Retrieval  | Nguyen et al.    | Pushing the boundaries of legal information processing with integration of large language models   |       |      |      |
| Precedent Retrieval    | Wiratunga et al. | CBR-RAG: case-based reasoning for retrieval augmented generation in LLMs for legal question answering |       |      |      |
| Factor Annotation      | Gray et al.      | Empirical legal analysis simplified: reducing complexity through automatic identification and evaluation of legally relevant factors |       |      |      |
| Legal Support          | Maree et al.    | Transforming legal text interactions: leveraging natural language processing and large language models for legal support in Palestinian cooperatives |       |      |      |
| Graph-based Retrieval  | Tang et al.      | CaseGNN: Graph Neural Networks for Legal Case Retrieval with Text-Attributed Graphs                |       |      |      |
| Reasoning              | Deng et al.      | Syllogistic reasoning for legal judgment analysis                                                 |       |      |      |
| Summarization Evaluation | Elaraby et al.  | Adding Argumentation into Human Evaluation of Long Document Abstractive Summarization: A Case Study on Legal Opinions |       |      |      |
| Hierarchical Classification | Prasad et al.   | Exploring Large Language Models and Hierarchical Frameworks for Classification of Large Unstructured Legal Documents |       |      |      |
| Information Retrieval  | Nguyen et al.    | Pushing the boundaries of legal information processing with integration of large language models   |       |      |      |
| Reasoning              | Deng et al.      | Syllogistic reasoning for legal judgment analysis                                                 |       |      |      |
| Element Identification | Reji et al.      | Enhancing LLM Performance on Legal Textual Entailment with Few-Shot CoT-based RAG                 |       |      |      |
| Litigation Support     | Siino et al.     | Exploring LLMs Applications in Law: A Literature Review on Current Legal NLP Approaches           |       |      |      |
| Benchmark              | Fei et al.       | Lawbench: Benchmarking legal knowledge of large language models                                    |       |      |      |

### Alternative Dispute Resolution

| Category               | Name            | Paper                                                                                              | Venue | Year | Code |
|------------------------|-----------------|---------------------------------------------------------------------------------------------------|-------|------|------|
| Contract Review        | Hendrycks et al. | CUAD: an expert-annotated NLP dataset for legal contract review                                  |       |      |      |
| Contract System        | Zeng et al.      | Contract-Mind: Trust-calibration interaction design for AI contract review tools                  |       |      |      |
| NLP for Contracts      | Graham et al.    | Natural language processing for legal document review: categorising deontic modalities in contracts |       |      |      |
| Financial Analysis     | Bedekar et al.   | AI in Mergers and Acquisitions: Analyzing the Effectiveness of Artificial Intelligence in Due Diligence |       |      |      |
| Reasoning              | Deng et al.      | Syllogistic reasoning for legal judgment analysis                                                 |       |      |      |
| Legal LLM              | Cui et al.       | Chatlaw: Open-source legal large language model with integrated external knowledge bases         |       |      |      |
| Reasoning Framework    | Yue et al.       | Disc-lawllm: Fine-tuning large language models for intelligent legal services                     |       |      |      |
| Benchmark              | Fei et al.       | Lawbench: Benchmarking legal knowledge of large language models                                    |       |      |      |
| Domain-specific LLM    | Fei et al.       | Internlm-law: An open source chinese legal large language model                                   |       |      |      |
| Legal QA System        | Huang et al.     | Lawyer llama technical report                                                                     |       |      |      |
| Contract Automation    | Mik              | Much ado about artificial intelligence or: the automation of contract formation                 |       |      |      |
| eDiscovery             | Pai et al.       | Exploration of open large language models for ediscovery                                         |       |      |      |
| Graph Learning         | Lahiri et al.    | Learning from Litigation: Graphs and LLMs for Retrieval and Reasoning in eDiscovery              |       |      |      |
| Blockchain             | Nagineni         | THE RISE OF AI IN EDISCOVERY: HOW MACHINE LEARNING IS REVOLUTIONIZING LEGAL DATA PROCESSING      |       |      |      |
| Mediation Agreement    | Goswami et al.   | Incorporating Domain Knowledge in Multi-objective Optimization Framework for Automating Indian Legal Case Summarization |       |      |      |
| Text Analytics         | Ghosh et al.     | Human Centered AI for Indian Legal Text Analytics                                                |       |      |      |
| Element Extraction     | Lyu et al.       | Improving legal judgment prediction through reinforced criminal element extraction                |       |      |      |
| Long-text Processor    | Xiao et al.      | Lawformer: A pre-trained language model for chinese legal long documents                          |       |      |      |
| Litigation Support     | Siino et al.     | Exploring LLMs Applications in Law: A Literature Review on Current Legal NLP Approaches           |       |      |      |
| Summarization          | Goswami et al.   | Incorporating Domain Knowledge in Multi-objective Optimization Framework for Automating Indian Legal Case Summarization |       |      |      |
| Text Analytics         | Ghosh et al.     | Human Centered AI for Indian Legal Text Analytics                                                |       |      |      |

## The Collaboration of Technological Ethics and Legal Ethics

![alt text](image-4.png)

Dual-Perspective Analysis:​​

​​Technological Ethics:​​
Hallucination mitigation (38% reduction in ChatLaw)
Bias quantification across 12 jurisdictions
Explainability via syllogistic prompting
​​Professional Ethics:​​
Confidentiality protocols: Input redaction standards
Accountability frameworks: Output verification requirements
Competence obligations: Continuous learning imperatives

## Roadmap and Future Directions

![alt text](image-3.png)

| Category                  | Name          | Paper                                                                                              | Venue | Year | Code |
|---------------------------|---------------|---------------------------------------------------------------------------------------------------|-------|------|------|
| Legal LLM                 | Cui et al.    | Chatlaw: Open-source legal large language model with integrated external knowledge bases         |       |      |      |
| Multi-agent Framework     | Shi et al.    | LegalGPT: Legal Chain of Thought for the Legal Large Language Model Multi-agent Framework          |       |      |      |
| Benchmark                 | Fei et al.    | Lawbench: Benchmarking legal knowledge of large language models                                    |       |      |      |
| Benchmark                 | Guha et al.   | Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models |       |      |      |
| Reasoning Enhancement    | Lu et al.     | 0x. yuan at semeval-2024 task 5: Enhancing legal argument reasoning with structured prompts        |       |      |      |

Critical research frontiers:

​​Multimodal Integration:​​ Text + Knowledge graphs + Evidence
​​Low-Resource Adaptation:​​ Few-shot tuning for minority jurisdictions
​​Dynamic Rebuttal:​​ Real-time counterargument handling
​​Evaluation Systems:​​ Cross-jurisdictional benchmarks (LawBench)
​​Generative Pretraining:​​ Legal-specific foundation models

## Conclusion

Transformative contributions establishing foundations for legal AI:

- First dual-taxonomy framework integrating Toulmin argumentation with professional roles
- Systematic review of technical evolution from symbolic systems to transformer LLMs
- Scenario-adaptive deployment models for litigation and ADR workflows
- Ethical co-regulation protocols balancing innovation with professional responsibility
- Roadmap addressing critical frontiers in low-resource systems and multimodal integration
