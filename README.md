# ARDI-Project Scholarly QALD Challenge 2024

This project is a student project developed for the Scholarly QALD Challenge 2024: [https://kgqa.github.io/scholarly-QALD-challenge/2024/](https://kgqa.github.io/scholarly-QALD-challenge/2024/).

Before executing the experiment, ensure to install the source code as a package by running the following command:

```shell
pip install -e .
```

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
├── LICENSE
├── Makefile
├── README.md
├── config.ini
├── data
│   ├── README.md
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw
├── docs
│   └── README.md
├── models
├── now.txt
├── reports
│   ├── README.md
│   ├── comparison_bar_plot_previous.png
│   ├── comparison_prompt_template.png
│   └── mean_loss.png
├── requirements.txt
├── results
│   ├── 4settings10q
│   ├── experiments_10q
│   ├── experiments_T5
│   ├── experiments_templates
│   ├── fine_tuning_preds_epoch_results
│   ├── results_4settings.json
│   └── sparql
├── setup.py
├── src
│   ├── data
│   ├── evaluation
│   ├── features
│   ├── fine-tuning
│   ├── main.py
│   ├── models
│   ├── utils
│   └── visualization
├── tests
│   ├── test_data_extraction.py
│   ├── test_evaluation.py
│   ├── test_evidence_matching.py
│   ├── test_fine_tuning_T5.ipynb
│   ├── test_predictions.py
│   ├── test_prepare_context.py
│   ├── test_qa_pipeline.py
│   └── test_zero_shot_prompting.py
└── tree.txt

30 directories, 28 files

```
