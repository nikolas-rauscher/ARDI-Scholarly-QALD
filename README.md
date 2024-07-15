> [!NOTE]
> The latest version of the code is currently on the `develop` branch. You can find it [here](https://github.com/nikolas-rauscher/ARDI-Scholarly-QALD/tree/develop).


# ARDI-Project Scholarly QALD Challenge 2024

This project is a student project developed for the Scholarly QALD Challenge 2024: [https://kgqa.github.io/scholarly-QALD-challenge/2024/](https://kgqa.github.io/scholarly-QALD-challenge/2024/).

## Our Approach

Our goal is to build a Question Answering system capable of answering questions related to the scholarly domain. To achieve this, we utilize an approach that involves:

1. **Context Creation:** We extract relevant information from various data sources, including:
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
<img width="1131" alt="image" src="https://github.com/user-attachments/assets/5119d6d9-f203-4a3d-810e-b57280d8d3a0">

## Folder Structure
```bash

.
.
├── README.md
├── data
│   ├── README.md
│   ├── external
│   │   └── README.md
│   ├── processed
│   │   ├── README.md
│   │   └── cleand_dataset
│   └── raw
│       ├── README.md
│       └── dataset
├── docs
│   └── README.md
├── reports
│   └── README.md
├── results
│   ├── README.md
│   ├── hm_results
│   │   ├── hm_evaluations
│   │   ├── processed
│   │   ├── scores
│   │   ├── scores_rouge
│   │   └── todo
│   └── prompt_context
└── src
    ├── __init__.py
    ├── data
    │   ├── data
    │   │   └── processed
    │   ├── evaluation
    │   │   └── qa_eval.py
    │   ├── make_dataset.py
    │   └── make_prompt.py
    ├── evaluation
    │   ├── hm_auto.py
    │   ├── qa.py
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
    │   ├── build_features.py
    │   ├── evidence_selection.py
    │   └── noise_reduction
    │       ├── README.md
    │       ├── filter_for_vaild_awnsers.py
    │       └── generate_spaql
    │           ├── crate-top-n.py
    │           ├── create_and_run_sparql.py
    │           ├── datasets
    │           │   ├── SPARQL
    │           │   ├── answers
    │           │   │   ├── filterd_awnsers
    │           │   │   │   └── 1000_qestions
    │           │   │   ├── final
    │           │   │   └── merged_dataset
    │           │   │       └── 1000_qestions
    │           │   ├── cleand_datasets
    │           │   ├── eval_results
    │           │   └── failed_queries
    │           ├── eval_results.py
    │           ├── get_example_dataset.py
    │           ├── merge_anwsers.py
    │           ├── simple_noise_reduction.py
    │           └── utils.py
    ├── fine-tuning
    │   └── finetuning.py
    ├── models
    │   ├── README.md
    │   ├── main.py
    │   ├── predict_model.py
    │   ├── prepare_prompts_context.py
    │   ├── train_model.py
    │   ├── verbalizer
    │   │   ├── README.md
    │   │   ├── __init__.py
    │   │   ├── generatePrompt.py
    │   │   ├── prompt_verbalizer.py
    │   │   └── verbalisation_module.py
    │   ├── zero_shot_prompting.py
    │   └── zero_shot_prompting_pipeline.py
    └── visualization
        └── visualize.py

45 directories, 44 files

```
