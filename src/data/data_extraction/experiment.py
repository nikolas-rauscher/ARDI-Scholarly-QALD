import configparser
import os

class Dataset:
    def __init__(self, experiment_name, config_file='config.ini', direct_input=False):
        # Load the configuration file directly
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.experiment_name = experiment_name
        self.direct_input = direct_input

        # Initialize paths using direct access to configparser
        self.questions_raw = self.config.get('FilePaths', 'raw_path')
        self.questions_path = self.config.get('FilePaths', 'raw_questions_path')
        self.raw_wikipedia_path = self.config.get('FilePaths', 'raw_wikipedia_path')
        self.processed_wikipedia_path = self.config.get('FilePaths', 'processed_wikipedia_path')
        self.dblp_endpoint_url = self.config.get('URLs', 'dblp_endpoint_url')
        self.openalex_endpoint_url = self.config.get('URLs', 'openalex_endpoint_url')

        # Dynamic experiment-specific paths organized by preprocessing and final merged data
        self.dblp_path_outputdata_preprocessed = f"{self.config.get('FilePaths', 'dblp_path_outputdata')}/preprocessed_{self.experiment_name}_dblp.json"
        self.dblp_path_outputdata_postprocessed = f"{self.config.get('FilePaths', 'dblp_path_outputdata')}/postprocessed_{self.experiment_name}_dblp.json"
        self.openalex_path_outputdata_preprocessed = f"{self.config.get('FilePaths', 'openalex_path_outputdata')}/preprocessed_{self.experiment_name}_alex.json"
        self.openalex_path_outputdata_postprocessed = f"{self.config.get('FilePaths', 'openalex_path_outputdata')}/postprocessed_{self.experiment_name}_alex.json"
        self.wikipedia_path_outputdata = f"{self.config.get('FilePaths', 'wikipedia_path_outputdata')}/preprocessed_{self.experiment_name}_wiki.json"
        self.final_path_merged_data = f"{self.config.get('FilePaths', 'merged_triples_and_wikipedia_path')}/final_merged_{self.experiment_name}.json"

    def get(self, section, option):
        return self.config.get(section, option)

