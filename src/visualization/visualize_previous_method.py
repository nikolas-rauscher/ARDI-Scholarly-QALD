import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(models, rougeL, meteor, save_path="./reports/comparison_bar_plot_previous.png"):
    """
    Plots a comparison bar chart for performance metrics of different models.
    
    Parameters:
    - models: list : Names of the models
    - rougeL: list : ROUGE-L scores for the models
    - meteor: list : METEOR scores for the models
    - save_path: str : Path to save the plot image
    """
    # Define bar width and positions
    bar_width = 0.35
    index = np.arange(len(models))

    # Creating the bar plots
    fig, ax = plt.subplots(figsize=(12, 3), dpi=250)
    bar1 = ax.bar(index, rougeL, bar_width, label='ROUGE-L', color='black')
    bar2 = ax.bar(index + bar_width, meteor, bar_width, label='METEOR', color='grey')

    # Adding labels, title, and legend
    ax.set_xlabel('Models', fontsize=16)
    ax.set_ylabel('Scores', fontsize=16)
    ax.set_title('Comparison of QA Approaches', fontsize=14)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(models, rotation=0, ha='center', fontsize=14)
    ax.legend(fontsize=14)

    # Function to add the labels on the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=14)

    # Call the function to add the labels
    autolabel(bar1)
    autolabel(bar2)

    # Remove grid and background to keep it clean
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='black')

    # Show the plot
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == '__main__':
    # Example usage
    models = ['Few-shot-SPARQL', 'Zero-shot LLaMA-2', 'Fine-tuned T5']
    rougeL = [0.6712, 24.14, 58.3]
    meteor = [0.48, 19.08, 57.6]

    plot_comparison(models, rougeL, meteor)
