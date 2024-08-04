# ARDI-Project Scholarly QALD Challenge 2024

This project is a student project developed for the Scholarly QALD Challenge 2024: [https://kgqa.github.io/scholarly-QALD-challenge/2024/](https://kgqa.github.io/scholarly-QALD-challenge/2024/).

## Installation 

[!Note]
Python version 3.12.

To install the verbalizer, follow these steps:

1. Download the `output.tar.gz` file from the following [link](https://drive.google.com/file/d/1OW2MkEffc6j-EqiWciMPVN0l56X3bNnS/view?usp=drive_link).

2. Place the downloaded file in the `graph2text` folder:

    ```bash
    cd src/data/verbalizer/graph2text
    ```

3. Unpack the file in the `graph2text` folder:

    ```bash
    tar -xvf output.tar.gz
    ```

4. Run the following commands to combine and extract the model files:

    ```bash
    cd outputs/t5-base_13881/best_tfmr
    cat pytorch_model.bin.tar.gz.parta* > pytorch_model.bin.tar.gz
    tar -xzvf pytorch_model.bin.tar.gz
    ```

Before executing the experiment, ensure to install the source code as a package by running the following command:

```bash
pip install -e .
```

and then install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

### We have created 2 main Pipelines

1. **Demo Pipeline**:
    - **Input**: A question and an Auto-ID.
    - **Process**: 
        - Searches the DBLP and OpenAlex Knowledge Graphs for relevant information (triples).
        - Performs triple extraction, prompt generation, and zero-shot prompting.
    - **Output**: Provides the answer based on the gathered data.

    To use the demo pipeline, run:
    ```bash
    python src/main.py --pipeline demo --question "Your Question Here" --auto_id "Your Auto-ID Here"
    ```

2. **Challenge Dataset Creation Pipeline**:
    - **Input**: Parameters required for the dataset generation.
    - **Process**: 
        - Collects and processes data from DBLP and OpenAlex Knowledge Graphs.
        - Generates and formats the dataset suitable for the Scholarly QALD Challenge.
    - **Output**: A dataset ready for use in the challenge.

    To create the challenge dataset, run:
    ```bash
    python src/main.py --pipeline create_dataset --parameters "Your Parameters Here"
    ```

### Component Testing
- **Description**: Allows testing of each individual component of the system sequentially.
- **Location**: Scripts are located in the `tests` folder.
- **Instructions**:
    - Navigate to the `tests` folder from the root directory.
    - Execute the desired scripts to verify the functionality of individual components.

    The components should be executed in the following order:

    1. **Noise Reduction**:
        ```bash
        python tests/test_noise_reduction.py
        ```

    2. **Data Extraction**:
        ```bash
        python tests/test_data_extraction.py
        ```

    3. **Prepare Prompt Context**:
        ```bash
        python tests/test_prepare_context.py
        ```

    4. **Zero-Shot Prompting** or **Fine-Tuning T5**:
        - For Zero-Shot Prompting:
          ```bash
          python tests/test_zero_shot_prompting.py
          ```

        - For Fine-Tuning T5:
          ```bash
          python tests/test_fine_tuning_T5.ipynb
          ```

    5. **Evaluation**:
        ```bash
        python tests/test_evaluation.py
        ```

    Additionally, you can test other subcomponents using the following scripts:
    - **Evidence Matching**:
      ```bash
      python tests/test_evidence_matching.py
      ```

    - **Predictions**:
      ```bash
      python tests/test_predictions.py
      ```

    - **QA Pipeline**:
      ```bash
      python tests/test_qa_pipeline.py
      ```

    - **Noise Reduction SPARQL Generation** (independent component):
      ```bash
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
├── config.ini                      ---> configuration for dataset, approach and llm and results.
├── data
│   ├── README.md
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw
├── docs
│   └── README.md
├── models
├── now.txt
├── reports
│   ├── README.md
│   ├── comparison_bar_plot_previous.png
│   ├── comparison_prompt_template.png
│   └── mean_loss.png
├── requirements.txt
├── results                         ---> results will be saved here.
│   ├── 4settings10q
│   ├── experiments_10q
│   ├── experiments_T5
│   ├── experiments_templates
│   ├── fine_tuning_preds_epoch_results
│   ├── results_4settings.json
│   └── sparql
├── setup.py
├── src
│   ├── data
│   ├── evaluation
│   ├── features
│   ├── fine-tuning                     
│   ├── models                      # ARDI-Project Scholarly QALD Challenge 2024

This project is a student project developed for the Scholarly QALD Challenge 2024: [https://kgqa.github.io/scholarly-QALD-challenge/2024/](https://kgqa.github.io/scholarly-QALD-challenge/2024/).

## Installation

To install the verbalizer, follow these steps:

1. Download the `output.tar.gz` file from the following [link](https://drive.google.com/file/d/1OW2MkEffc6j-EqiWciMPVN0l56X3bNnS/view?usp=drive_link).

2. Place the downloaded file in the `graph2text` folder:

    ```bash
    cd src/data/verbalizer/graph2text
    ```

3. Unpack the file in the `graph2text` folder:

    ```bash
    tar -xvf output.tar.gz
    ```

4. Run the following commands to combine and extract the model files:

    ```bash
    cd outputs/t5-base_13881/best_tfmr
    cat pytorch_model.bin.tar.gz.parta* > pytorch_model.bin.tar.gz
    tar -xzvf pytorch_model.bin.tar.gz
    ```

Before executing the experiment, ensure to install the source code as a package by running the following command:

```bash
pip install -e .
```

and then install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

### We have created 2 main Pipelines

1. **Demo Pipeline**:
    - **Input**: A question and an Auto-ID.
    - **Process**: 
        - Searches the DBLP and OpenAlex Knowledge Graphs for relevant information (triples).
        - Performs triple extraction, prompt generation, and zero-shot prompting.
    - **Output**: Provides the answer based on the gathered data.

    To use the demo pipeline, run:
    ```bash
    python src/main.py --pipeline demo --question "Your Question Here" --auto_id "Your Auto-ID Here"
    ```

2. **Challenge Dataset Creation Pipeline**:
    - **Input**: Parameters required for the dataset generation.
    - **Process**: 
        - Collects and processes data from DBLP and OpenAlex Knowledge Graphs.
        - Generates and formats the dataset suitable for the Scholarly QALD Challenge.
    - **Output**: A dataset ready for use in the challenge.

    To create the challenge dataset, run:
    ```bash
    python src/main.py --pipeline create_dataset --parameters "Your Parameters Here"
    ```

### Component Testing
- **Description**: Allows testing of each individual component of the system sequentially.
- **Location**: Scripts are located in the `tests` folder.
- **Instructions**:
    - Navigate to the `tests` folder from the root directory.
    - Execute the desired scripts to verify the functionality of individual components.

    The components should be executed in the following order:

    1. **Noise Reduction**:
        ```bash
        python tests/test_noise_reduction.py
        ```

    2. **Data Extraction**:
        ```bash
        python tests/test_data_extraction.py
        ```

    3. **Prepare Prompt Context**:
        ```bash
        python tests/test_prepare_context.py
        ```

    4. **Zero-Shot Prompting** or **Fine-Tuning T5**:
        - For Zero-Shot Prompting:
          ```bash
          python tests/test_zero_shot_prompting.py
          ```

        - For Fine-Tuning T5:
          ```bash
          python tests/test_fine_tuning_T5.ipynb
          ```

    5. **Evaluation**:
        ```bash
        python tests/test_evaluation.py
        ```

    Additionally, you can test other subcomponents using the following scripts:
    - **Evidence Matching**:
      ```bash
      python tests/test_evidence_matching.py
      ```

    - **Predictions**:
      ```bash
      python tests/test_predictions.py
      ```

    - **QA Pipeline**:
      ```bash
      python tests/test_qa_pipeline.py
      ```

    - **Noise Reduction SPARQL Generation** (independent component):
      ```bash
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
├── config.ini                      ---> configuration for dataset, approach and llm and results.
├── requirements.txt
├── setup.py
├── data
│   ├── README.md
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw
├── docs
├── reports
├── results                         ---> results will be saved here.
│   ├── 4settings10q
│   ├── experiments_10q
│   ├── experiments_T5
│   ├── experiments_templates
│   ├── fine_tuning_preds_epoch_results
│   ├── results_4settings.json
│   └── sparql
├── src
│   ├── data
│   ├── evaluation
│   ├── features
│   ├── fine-tuning
│   ├── models                         ---> the pipeline is started with this
│   ├── utils
│   └── visualization
├── tests
│   ├── test_noise_reduction.py
│   ├── test_noise_reduction_SPARQL_gernation.py
│   ├── test_data_extraction.py
│   ├── test_prepare_context.py
│   ├── test_zero_shot_prompting.py
│   ├── test_fine_tuning_T5.ipynb
│   ├── test_predictions.py
│   ├── test_evidence_matching.py
│   ├── test_evaluation.py
│   └── test_qa_pipeline.py
└── tree.txt
│   ├── utils
│   └── visualization
├── tests
│   ├── test_noise_reduction.py
│   ├── test_noise_reduction_SPARQL_generation.py
│   ├── test_data_extraction.py
│   ├── test_prepare_context.py
│   ├── test_zero_shot_prompting.py
│   ├── test_fine_tuning_T5.ipynb
│   ├── test_predictions.py
│   ├── test_evidence_matching.py
│   ├── test_evaluation.py
│---└── test_qa_pipeline.py