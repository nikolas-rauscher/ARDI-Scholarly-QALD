from .triple_extraction.dblp.create_dataset_dblp import create_dataset_dblp
from .triple_extraction.dblp.postprocess_dataset_dblp import post_process_dblp
from .triple_extraction.openalex.create_dataset_alex import create_alex_dataset 
from .triple_extraction.openalex.postprocess_dataset_alex import post_process_alex_parallelized
from .wikipedia_data.add_wiki_data import add_wikidata
from .merge_data import merge_data

from .experiment import Dataset

def create_dataset(config, expriment_name):
    dataset = Dataset(config, expriment_name)

    """# Creating and preprocessing the DBLP dataset
    create_dataset_dblp(dataset.questions_path, 
                        dataset.dblp_endpoint_url,
                        dataset.dblp_path_outputdata_preprocessed, subset=2) 

    post_process_dblp(dataset.dblp_path_outputdata_preprocessed,
                      dataset.dblp_path_outputdata_postprocessed)

    # Creating and preprocessing the OpenAlex dataset
    create_alex_dataset(dataset.dblp_path_outputdata_preprocessed, 
                        dataset.openalex_path_outputdata_preprocessed, 
                        dataset.openalex_endpoint_url)

    post_process_alex_parallelized(dataset.openalex_path_outputdata_preprocessed, 
                                   dataset.openalex_path_outputdata_postprocessed, 
                                   dataset.openalex_endpoint_url, processes=8)

    # Adding Wikipedia data
    add_wikidata(dataset.raw_wikipedia_path, 
                 dataset.processed_wikipedia_path,
                 dataset.openalex_path_outputdata_postprocessed, 
                 dataset.wikipedia_path_outputdata)"""

    # Merging all data
    merge_data(dataset.dblp_path_outputdata_postprocessed, 
               dataset.openalex_path_outputdata_postprocessed, 
               dataset.wikipedia_path_outputdata,
               dataset.final_path_merged_data)




