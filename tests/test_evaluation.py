import configparser
from evaluation.metrics import process_metric_for_all_files
from evaluation.postprocess_answers import process_file, process_for_evaluation
config = configparser.ConfigParser()
config.read('config.ini')

def evaluation4settings():
    output_dir="./results/4settings10q"
    process_file(config['FilePaths']['default4settings_results_file'], process_for_evaluation, output_dir)
    process_metric_for_all_files(output_dir,
                                 'data/raw/raw_train_dataset.json')

if __name__ == '__main__':
    evaluation4settings()