## Noise Reduction in DBLP-QuAD Dataset

This module focuses on reducing noise in the DBLP-QuAD dataset and improving/replacing the answers with the received answers by the generated SPARQL queries.

### Process

1. **Simple Noise Reduction:**
    - The `simple_noise_reduction.py` script cleans the initial dataset by removing entries with empty or nonsensical answers.
    - It filters out answers based on length and the presence of irrelevant characters.

2. **SPARQL Generation:**
    - Similar questions from a pre-processed example dataset are retrieved based on embeddings generated using Sentence Transformers and indexed with FAISS.
    - The `SPARQL-gneration-with-llm.py` script uses a large language model (LLM) to generate SPARQL queries based on the input questions and a few-shot prompt approach.

3. **Answer Extraction:**
    - Generated SPARQL queries are executed against the DBLP SPARQL endpoint.
    - The `create_and_run_sparql.py` script handles the execution and retrieval of answers from the endpoint.

4. **Answer Filtering:**
    - The `filter_for_vaild_awnsers.py` script filters out invalid or nonsensical answers from the SPARQL results.
    - It ensures that only valid and meaningful answers are retained.

5. **Evaluation:**
    - The `eval_results.py` script evaluates the generated SPARQL queries and answers against a gold standard.
    - It provides insights into the performance of the noise reduction and SPARQL generation process.

### Files

- `simple_noise_reduction.py`: Cleans the initial dataset.
- `SPARQL-gneration-with-llm.py`: Generates SPARQL queries using an LLM.
- `create_and_run_sparql.py`: Executes SPARQL queries and retrieves answers.
- `filter_for_vaild_awnsers.py`: Filters out invalid answers from SPARQL results.
- `eval_results.py`: Evaluates the generated SPARQL queries and answers.
- `utils.py`: Provides utility functions for reading data, processing queries, etc.


### Usage 

1. Set up environment variables
    Create a .env file in the root directory of the project and add your OpenAI API key:
````
echo "OPENAI_API_KEY=your_api_key_here" > .env
````
2. Edit the config.ini file to set the necessary paths and parameters. Here is an example configuration:
````
[noise_reduction_parameters]
shot = 5
limit = 10000
with_schema = true
model_name = gpt-3.5-turbo
````
3. Run the script
````
python src/data/noise_reduction/main.py
