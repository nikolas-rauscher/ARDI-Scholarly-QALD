import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(template1, template2, save_path="./reports/comparison_prompt_template.png"):
    """
    Plots a comparison bar chart for performance metrics of two templates.
    
    Parameters:
    - template1: dict : Performance metrics for the first template
    - template2: dict : Performance metrics for the second template
    - save_path: str : Path to save the plot image

    Both dictionaries must have the same keys.
    """
    # Assure that both dictionaries have the same keys
    assert template1.keys() == template2.keys(), "The templates must have the same keys"

    # Extract the keys and values in order
    metrics = list(template1.keys())
    values1 = list(template1.values())
    values2 = list(template2.values())

    # Define the X axis locations for the groups
    x = np.arange(len(metrics))

    # Define the width of the bars
    bar_width = 0.35

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    rects1 = ax.bar(x - bar_width/2, values1, bar_width, color='black', label='Template 1')
    rects2 = ax.bar(x + bar_width/2, values2, bar_width, color='grey', label='Template 2')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Scores by Metric and Template', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=10)

    # Function to add the labels on the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)

    # Call the function to add the labels
    autolabel(rects1)
    autolabel(rects2)

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
    plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    # Example usage
    template1 = {
        "rouge1": 13.0781,
        "rouge2": 3.2,
        "rougeL": 12.7921,
        "rougeLsum": 13.0305,
        "EM": 9.0,
        "meteor": 9.98
    }

    template2 = {
        "rouge1": 13.9308,
        "rouge2": 4.0,
        "rougeL": 13.6234,
        "rougeLsum": 13.8504,
        "EM": 0.0,
        "meteor": 4.9
    }

    plot_comparison(template1, template2)
