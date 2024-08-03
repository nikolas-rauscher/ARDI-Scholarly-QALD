import configparser
from src.data.data_extraction.create_context import create_dataset
from src.data.data_extraction.run_question import run_question

class Config:
    def __init__(self, filename='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(filename)

    def get(self, section, option):
        # Get a value from the configuration file
        return self.config.get(section, option)

config = Config()

##################################################################################

#Extract triples and add wikipedia context
#experiment_name = "Test"
#create_dataset(config, experiment_name)

##################################################################################

question = "What is the field of study of Tom Wilson?"
author_dblp_uri = ["https://dblp.org/pid/w/TDWilson"]
question_identifier = "UNO"


run_question(question, author_dblp_uri, question_identifier, config)

##################################################################################