from data.noise_reduction.simple_noise_reduction import clean_and_save_dataset
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

if __name__ == "__main__":
    dataset = "full_dataset"
    input_file_path = config['FilePaths']['raw_questions_path']
    output_file_path = config['FilePaths']['clean_data_path']
    clean_and_save_dataset(input_file_path, output_file_path)
    
