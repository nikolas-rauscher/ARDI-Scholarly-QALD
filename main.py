from src.data.create_dataset_dblp import create_dataset
import config

outputdata_name = "processed_data"
processed_data_name = config.save_processed_data_path
trainingdata_path = config.trainingdata_path
endpoint_url = config.endpoint_url #SPARQL endpoint URL
create_dataset(trainingdata_path, endpoint_url, processed_data_name, outputdata_name) 
