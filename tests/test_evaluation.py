from evaluation.metrics import process_metric_for_all_files


if __name__ == '__main__':
    process_metric_for_all_files("/Users/celes/Documents/Projects/ARDI-Scholarly-QALD/results/fine_tuning_preds_epoch_results",
                                 '/Users/celes/Documents/Projects/ARDI-Scholarly-QALD/data/raw/raw_train_dataset.json')
