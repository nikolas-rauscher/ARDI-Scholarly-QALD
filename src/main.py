from data.data_extraction.create_context import data_extraction
from data.data_extraction.run_question import run_question

##################################################################################

#Extract triples and add wikipedia context for file of questions (path in config.ini)
experiment_name = "full_dataset"
data_extraction(experiment_name)

##################################################################################

#Extract triples and add wikipedia context for one context
question = "What is the field of study of Tom Wilson?"
author_dblp_uri = ["https://dblp.org/pid/w/TDWilson"]

#run_question(question, author_dblp_uri)

##################################################################################
