# ARDI-Project Scholarly QALD Challenge 2024

This project is a student project developed for the Scholarly QALD Challenge 2024: [https://kgqa.github.io/scholarly-QALD-challenge/2024/](https://kgqa.github.io/scholarly-QALD-challenge/2024/).

## Our Approach

Our goal is to build a Question Answering system capable of answering questions related to the scholarly domain. To achieve this, we utilize an approach that involves:

1. **Information Extraction:** We extract relevant information (triples) from 3 data sources, including:
    - DBLP KG
    - OpenAlex KG
    - Wikipedia
2. **Evidence Matching and Verbalization:** We employ evidence matching techniques to identify relevant information within the context and then verbalize it into a format suitable for question answering.
3. **Zero-Shot Prompting and Fine-tuning:** We leverage pre-trained language models and fine-tune them for question answering using the created context. We experiment with both zero-shot prompting and fine-tuning approaches.

## Language Models

We experiment with several state-of-the-art language models:

- Llama 3
- Llama 2
- Mistral
- T5

### Methodology - Answer Generation
<img width="1131" alt="image" src="https://i.imgur.com/ZrOr4YG.png">

## Folder Structure
```bash

.
.
├── README.md
├── config.py
├── data
│   ├── README.md
│   ├── external
│   │   └── README.md
│   ├── processed
│   │   ├── alex
│   │   │   └── README.md
│   │   ├── cleand_dataset
│   │   └── dblp
│   │       └── README.md
│   └── raw
│       ├── README.md
│       └── dataset
├── docs
│   └── README.md
├── main.py
├── reports
│   └── README.md
├── results
│   ├── README.md
└── src
    ├── __init__.py
    ├── data
    │   ├── README.md
    │   ├── add_wiki_data.py
    │   ├── analyse_dataset_alex.py
    │   ├── analyse_dataset_dblp.py
    │   ├── create_dataset_alex.py
    │   ├── create_dataset_dblp.py
    │   ├── data
    │   │   └── processed
    │   ├── evaluation
    │   │   └── qa_eval.py
    │   ├── make_prompt.py
    │   ├── merge_triples.py
    │   ├── postprocessesing_wikidata.py
    │   ├── postprocessing_alex.py
    │   ├── postprocessing_dblp.py
    │   ├── process_wikidata.py
    │   ├── run_queries.py
    │   └── run_query.py
    ├── evaluation
    │   ├── README.md
    │   ├── hm_auto.py
    │   ├── qa_eval.py
    │   ├── results_prepocess.py
    │   ├── rouge.py
    │   └── scripts_for_human_eval
    │       ├── README.md
    │       ├── human_eval.py
    │       ├── input
    │       ├── output
    │       ├── plot_human_eval.py
    │       └── results
    ├── features
    │   ├── evidence_selection.py
    │   └── noise_reduction
    │       ├── README.md
    │       ├── filter_for_vaild_awnsers.py
    │       └── generate_spaql
    │           ├── crate-top-n.py
    │           ├── create_and_run_sparql.py
    │           ├── datasets
    │           ├── eval_results.py
    │           ├── get_example_dataset.py
    │           ├── merge_anwsers.py
    │           ├── simple_noise_reduction.py
    │           └── utils.py
    ├── fine-tuning
    │   └── finetuning.py
    ├── models
    │   ├── README.md
    │   ├── prepare_prompts_context.py
    │   ├── verbalizer
    │   │   ├── README.md
    │   │   ├── __init__.py
    │   │   ├── generatePrompt.py
    │   │   ├── prompt_verbalizer.py
    │   │   └── verbalisation_module.py
    │   └── zero_shot_prompting_pipeline.py
    └── visualization
        └── visualize.py

55 directories, 55 files


```
