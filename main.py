from src.data_extraction.triple_extraction.dblp.create_dataset_dblp import create_dataset_dblp
from src.data_extraction.triple_extraction.dblp.postprocess_dataset_dblp import post_process_dblp
from src.data_extraction.triple_extraction.openalex.create_dataset_alex import create_alex_dataset 
from src.data_extraction.triple_extraction.openalex.postprocess_dataset_alex import post_process_alex_parallelized
from src.data_extraction.wikipedia_data.add_wiki_data import add_wikidata
from src.data_extraction.merge_data import merge_data

import config

def create_dataset(config):
    create_dataset_dblp(config.Qustions_path, 
                        config.DBLP_endpoint_url,
                        config.DBLP_name_outputdata_PREprocessed, subset = 10) 

    post_process_dblp(config.DBLP_name_outputdata_PREprocessed,
                    config.DBLP_name_outputdata_POSTprocessed)

    create_alex_dataset(config.DBLP_name_outputdata_PREprocessed, 
                        config.OpenAlex_name_outputdata_PREprocessed, 
                        config.OpenAlex_endpoint_url)

    post_process_alex_parallelized(config.OpenAlex_name_outputdata_PREprocessed, 
                                config.OpenAlex_name_outputdata_POSTprocessed, 
                                config.OpenAlex_endpoint_url, processes=8)

    add_wikidata(config.RAW_Wikipedia_path, 
                config.OpenAlex_name_outputdata_POSTprocessed, 
                config.Wikipedia_name_outputdata)

    merge_data(config.DBLP_name_outputdata_POSTprocessed, 
            config.OpenAlex_name_outputdata_POSTprocessed, 
            config.Wikipedia_name_outputdata,
            config.Final_name_merged_data)
    
create_dataset(config)

