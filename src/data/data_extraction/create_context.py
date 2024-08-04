from .triple_extraction.dblp.create_dataset_dblp import create_dataset_dblp
from .triple_extraction.dblp.postprocess_dataset_dblp import post_process_dblp
from .triple_extraction.openalex.create_dataset_alex import create_alex_dataset 
from .triple_extraction.openalex.postprocess_dataset_alex import post_process_alex_parallelized
from .wikipedia_data.add_wiki_data import add_wikidata
from .merge_data import merge_data

from .experiment import Dataset


def data_extraction(expriment_name):
    dataset = Dataset(expriment_name)
    create_dataset(dataset)

def create_dataset(config):

    # Creating and preprocessing the DBLP dataset
    create_dataset_dblp(config.questions_path, 
                        config.dblp_endpoint_url,
                        config.dblp_path_outputdata_preprocessed, subset=100) 

    post_process_dblp(config.dblp_path_outputdata_preprocessed,
                      config.dblp_path_outputdata_postprocessed)

    # Creating and preprocessing the OpenAlex dataset
    create_alex_dataset(config.dblp_path_outputdata_preprocessed, 
                        config.openalex_path_outputdata_preprocessed, 
                        config.openalex_endpoint_url)

    post_process_alex_parallelized(config.openalex_path_outputdata_preprocessed, 
                                   config.openalex_path_outputdata_postprocessed, 
                                   config.openalex_endpoint_url, processes=8)

    # Adding Wikipedia data
    add_wikidata(config.raw_wikipedia_path, 
                 config.processed_wikipedia_path,
                 config.openalex_path_outputdata_postprocessed, 
                 config.wikipedia_path_outputdata)

    # Merging all data
    merge_data(config.dblp_path_outputdata_postprocessed, 
               config.openalex_path_outputdata_postprocessed, 
               config.wikipedia_path_outputdata,
               config.final_path_merged_data)




