from data.data_extraction.create_context import data_extraction

#Extract triples and add wikipedia context for file of questions (path in config.ini)

if __name__ == "__main__":
    experiment_name = "full_dataset"
    data_extraction(experiment_name)