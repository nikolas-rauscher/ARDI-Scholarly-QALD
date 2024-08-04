from data.noise_reduction.noise_reduction_pipeline import main as noise_reduction_pipeline
import configparser



# Read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

if __name__ == "__main__":
    # Get the paths from the config file
    dataset = "full_dataset"
    input_file_path = config['FilePaths']['raw_questions_path']
    output_file_path = config['FilePaths']['clean_data_path']
    
    # Call the pipeline function from simple_noise_reduction
    noise_reduction_pipeline()