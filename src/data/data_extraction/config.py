class Dataset:
    def __init__(self, config, experiment_name, direct_input = False):
        # Config is an instance of Config class
        self.config = config
        self.experiment_name = experiment_name
        self.direct_input = direct_input

        # Initialize static paths using the Config instance
        if config.questions_path:
             self.questions_path = config.questions_path
        else:
            self.questions_path = config.get('FilePaths', 'raw_questions_path')
        self.questions_raw = config.get('FilePaths', 'raw_path')
        self.raw_wikipedia_path = config.get('FilePaths', 'raw_wikipedia_path')
        self.processed_wikipedia_path = config.get('FilePaths', 'processed_wikipedia_path')
        self.dblp_endpoint_url = config.get('URLs', 'dblp_endpoint_url')
        self.openalex_endpoint_url = config.get('URLs', 'openalex_endpoint_url')

                
        # Dynamic experiment-specific paths organized by preprocessing and final merged data
        self.dblp_path_outputdata_preprocessed = f"{config.get('FilePaths', 'dblp_path_outputdata')}/preprocessed_{self.experiment_name}_dblp.json"
        self.dblp_path_outputdata_postprocessed = f"{config.get('FilePaths', 'dblp_path_outputdata')}/postprocessed_{self.experiment_name}_dblp.json"
        self.openalex_path_outputdata_preprocessed = f"{config.get('FilePaths', 'openalex_path_outputdata')}/preprocessed_{self.experiment_name}_alex.json"
        self.openalex_path_outputdata_postprocessed = f"{config.get('FilePaths', 'openalex_path_outputdata')}/postprocessed_{self.experiment_name}_alex.json"
        self.wikipedia_path_outputdata = f"{config.get('FilePaths', 'wikipedia_path_outputdata')}/preprocessed_{self.experiment_name}_wiki.json"
        self.final_path_merged_data = f"{config.get('FilePaths', 'merged_triples_and_wikipedia_path')}/final_merged_{self.experiment_name}.json"