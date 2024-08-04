# ARDI-Project Scholarly QALD Challenge 2024

This project is a student project developed for the Scholarly QALD Challenge 2024: [https://kgqa.github.io/scholarly-QALD-challenge/2024/](https://kgqa.github.io/scholarly-QALD-challenge/2024/).

## Usage
Before executing the experiment, ensure to install the source code as a package by running the following command:

```shell
pip install -e .
```

and then install the required dependencies by running:

```shell
pip install -r requirements.txt
```

### We have created 2 main Pipelines

1. **Demo Pipeline**:
    - **Input**: A question and an Auto-ID.
    - **Process**: 
        - Searches the DBLP and OpenAlex Knowledge Graphs for relevant information (triples).
        - Performs triple extraction, prompt generation, and zero-shot prompting.
    - **Output**: Provides the answer based on the gathered data.

    To use the demo pipeline, run:
    ```shell
    python src/models/qa_pipeline.py --pipeline demo --question "Your Question Here" --auto_id "Your Auto-ID Here"
    ```

2. **Challenge Dataset Creation Pipeline**:
    - **Input**: Parameters required for the dataset generation.
    - **Process**: 
        - Collects and processes data from DBLP and OpenAlex Knowledge Graphs.
        - Generates and formats the dataset suitable for the Scholarly QALD Challenge.
    - **Output**: A dataset ready for use in the challenge.

    To create the challenge dataset, run:
    ```shell
    python tests/test_<component_name>.py 
    ```

### Component Testing
- **Description**: Allows testing of each individual component of the system sequentially.
- **Location**: Scripts are located in the `tests` folder.
- **Instructions**:
    - Navigate to the `tests` folder from the root directory.
    - Execute the desired scripts to verify the functionality of individual components.

    The components should be executed in the following order:

    1. **Noise Reduction**:
        ```shell
        python tests/test_noise_reduction.py
        ```

    2. **Data Extraction**:
        ```shell
        python tests/test_data_extraction.py
        ```

    3. **Prepare Prompt Context**:
        ```shell
        python tests/test_prepare_context.py
        ```

    4. **Zero-Shot Prompting** or **Fine-Tuning T5**:
        - For Zero-Shot Prompting:
          ```shell
          python tests/test_zero_shot_prompting.py
          ```

        - For Fine-Tuning T5:
          ```shell
          python tests/test_fine_tuning_T5.ipynb
          ```

    5. **Evaluation**:
        ```shell
        python tests/test_evaluation.py
        ```

    Additionally, you can test other subcomponents using the following scripts:
    - **Evidence Matching**:
      ```shell
      python tests/test_evidence_matching.py
      ```

    - **Predictions**:
      ```shell
      python tests/test_predictions.py
      ```

    - **QA Pipeline**: (independent component):
      ```shell
      python tests/test_qa_pipeline.py
      ```

    - **Noise Reduction SPARQL Generation** (independent component):
      ```shell
      python tests/test_noise_reduction_SPARQL_gernation.py
      ```

    Note: Before running these components, ensure that the corresponding paths are correctly set in the `config.ini` file.

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
│   ├── test_noise_reduction.py
│   ├── test_noise_reduction_SPARQL_gernation.py
│   ├── test_data_extraction.py
│   ├── test_prepare_context.py
│   ├── test_zero_shot_prompting.py
│   ├── test_fine_tuning_T5.ipynb
│   ├── test_predictions.py
│   ├── test_evidence_matching.py
│   ├── test_evaluation.py
│   └── test_qa_pipeline.py
└── tree.txt

30 directories, 28 files
```