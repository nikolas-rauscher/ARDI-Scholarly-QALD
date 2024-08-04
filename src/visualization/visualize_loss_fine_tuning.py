import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Setting up the plot style for black and white
plt.rc('font', size=12)
plt.rc('axes', titlesize=12)
plt.rc('axes', labelsize=12)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=12)

# plt.style.use('seaborn-whitegrid')  # Apply a clean, white grid style

def visualize_loss_fine_tuning(files_directory):
    """
    Visualize the evaluation loss during fine-tuning and create a boxplot for model selection.

    This function processes training state files from different model checkpoints in the specified 
    directory, plots the evaluation loss over epochs, and generates a boxplot to compare the 
    distribution of evaluation losses for different model checkpoints.

    Args:
        files_directory (str): The directory containing the model checkpoint subdirectories with 
                               'trainer_state.json' files.

    Returns:
        None: The function saves the generated plots as PNG files and displays them.
    """

    # Loop through the files in the directory
    epoch_losses = {}
    for dicname in os.listdir(files_directory):
        # Filter for the files of interest
        if 'model' not in dicname:
            continue
        path = os.path.join(files_directory, dicname, 'trainer_state.json')
        print(f'Loading data from: {path}')
        with open(path, 'r') as f:
            data = json.load(f)["log_history"]

        # Initialize lists to store epoch numbers and loss values
        for item in data:
            epoch = item['epoch']
            if 'eval_loss' in item:
                epoch = int(epoch)
                if epoch not in epoch_losses:
                    epoch_losses[epoch] = {'eval_loss': [], 'train_loss': []}
                epoch_losses[epoch]['eval_loss'].append(item['eval_loss'])
            if 'loss' in item:
                epoch = int(epoch) + 1
                if epoch not in epoch_losses:
                    epoch_losses[epoch] = {'eval_loss': [], 'train_loss': []}
                epoch_losses[epoch]['train_loss'].append(item['loss'])
        print(epoch_losses)

    mean_epoch_losses = {'epochs': [], 'mean_eval_loss': [], 'std_eval_loss': [], 'mean_train_loss': [], 'std_train_loss': []}
    for epoch in sorted(epoch_losses.keys()):
        mean_epoch_losses['epochs'].append(epoch)
        print(epoch_losses[epoch]['eval_loss'])
        if epoch_losses[epoch]['eval_loss']:
            mean_epoch_losses['mean_eval_loss'].append(np.mean(epoch_losses[epoch]['eval_loss']))
            mean_epoch_losses['std_eval_loss'].append(np.std(epoch_losses[epoch]['eval_loss']))
        else:
            mean_epoch_losses['mean_eval_loss'].append(None)
            mean_epoch_losses['std_eval_loss'].append(None)
        if epoch_losses[epoch]['train_loss']:
            mean_epoch_losses['mean_train_loss'].append(np.mean(epoch_losses[epoch]['train_loss']))
            mean_epoch_losses['std_train_loss'].append(np.std(epoch_losses[epoch]['train_loss']))
        else:
            mean_epoch_losses['mean_train_loss'].append(None)
            mean_epoch_losses['std_train_loss'].append(None)

    plt.figure(figsize=(8, 6))

    epochs = mean_epoch_losses['epochs']
    mean_eval_loss = mean_epoch_losses['mean_eval_loss']
    std_eval_loss = mean_epoch_losses['std_eval_loss']
    mean_train_loss = mean_epoch_losses['mean_train_loss']
    std_train_loss = mean_epoch_losses['std_train_loss']

    # Use black and white colors
    plt.plot(epochs, mean_eval_loss, color='black', label='Mean Validation Loss', marker='o')
    plt.fill_between(epochs, np.array(mean_eval_loss) - np.array(std_eval_loss), np.array(mean_eval_loss) + np.array(std_eval_loss), color='grey', alpha=0.4)

    plt.plot(epochs, mean_train_loss, color='black', linestyle='--', label='Mean Training Loss', marker='o')
    plt.fill_between(epochs, np.array(mean_train_loss) - np.array(std_train_loss), np.array(mean_train_loss) + np.array(std_train_loss), color='grey', alpha=0.4)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Mean Training and Validation Loss Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/mean_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    files_directory = 'results/experiments_T5/fine-tuning'
    visualize_loss_fine_tuning(files_directory)
