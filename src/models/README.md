# Zero-Shot Prompting Pipeline README

## Configuration Explanation

### [FilePaths]

This section defines the file paths used by the pipeline.

- `zero_shot_prompting_result_file`: Path to the file where the zero-shot prompting results will be saved.
- `prompt_template`: Path to the prompt template file. (Not the one for the pipeline)
- `test_data_file`: Path to the test data file.
- `prepared_data4settings_file`: Path to the prepared data file.
- `zero_shot_4settings_results_file`: Path to the file where the zero-shot results will be saved.

### [Token]

This section contains the API tokens required for accessing the respective APIs.

- `llamaapi`: Token for accessing the Llama API.
- `groqapi`: Token for accessing the Groq API.

### [Model]

This section specifies the model configurations.

- `model_id_1`: Identifier for the first model (Llama 3, 7B parameters).
- `model_id_2`: Identifier for the second model (Llama 2, 13B parameters).
- `model_id_3`: Identifier for the third model (Mistral, 7B parameters).

### [Templates]

This section specifies the paths to the prompt templates for each model.

- `model_1_prompt_templates`: Path to the prompt templates for model 1.
- `model_2_prompt_templates`: Path to the prompt templates for model 2.
- `model_3_prompt_templates`: Path to the prompt templates for model 3.

### [Flags]

This section contains configuration flags.

- `use_api`: Boolean flag to indicate whether to use an API (`True` or `False`).
- `api_type`: Specifies which API to use (`llama` or `groq`).

### [Parameters]

This section contains parameters for controlling the input and output lengths.

- `max_length_input`: Maximum length of the input in characters.
- `max_output_length`: Maximum length of the output in tokens.

## Running the Pipeline

### Local Model Inference (Cluster)

To run the pipeline using a local model on a cluster:

1. **Set up the Configuration**:

   - Set `use_api` to `False` in the `[Flags]` section of the `config.ini` file.
   - Ensure that the appropriate model files are available locally on the cluster nodes.
   - Adjust the `[Parameters]` section as needed, particularly `max_length_input` and `max_output_length`.

   Example `config.ini`:

   ```ini
   [FilePaths]
   zero_shot_prompting_result_file = ./results/zero_shot_prompting.json
   prompt_template = ./data/raw/prompt_template.txt
   test_data_file = ./data/processed/test_processed_data_100.json
   prepared_data4settings_file = ./results/prepared_data4settings_file_zero_shot_prompting.json
   zero_shot_4settings_results_file = ./results/zero_shot_prompting.json

   [Token]
   llamaapi = LL-toLj4vcjSC82cwlgF0yf0o5sUs8tTj8tPhbABsKMb1c74TnQud7bJbcFozHTu00n
   groqapi = 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

   [Model]
    model_id_1 =meta-llama/Meta-Llama-3-8B-Instruct
    model_id_2 = meta-llama/Meta-Llama-2-7B
    model_id_3 = mistral/Mistral-7B

   [Templates]
   model_1_prompt_templates = data/raw/prompt_template.txt
   model_2_prompt_templates = data/raw/prompt_template.txt
   model_3_prompt_templates = data/raw/prompt_template.txt

   [Flags]
   use_api = False
   api_type =

   [Parameters]
   max_length_input = 4000
   max_output_length = 50
   ```

### To run the pipeline, execute the following command:

```bash
python3 src/model/main.py
```
