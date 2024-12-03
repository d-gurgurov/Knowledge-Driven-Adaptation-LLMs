import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def combine_plots(plot_paths, save_path):
    # Create a figure with a 2x2 grid for the plots
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))

    # Plot mBERT in top-left
    img_mbert_cn = mpimg.imread(plot_paths['1'])
    axs[0, 0].imshow(img_mbert_cn)
    axs[0, 0].axis('off')

    # Plot XLM-R in top-right
    img_mbert_glot = mpimg.imread(plot_paths['2'])
    axs[0, 1].imshow(img_mbert_glot)
    axs[0, 1].axis('off')

    # Plot additional image 1 in bottom-left
    img_mbert_additional1 = mpimg.imread(plot_paths['3'])
    axs[1, 0].imshow(img_mbert_additional1)
    axs[1, 0].axis('off')

    # Plot additional image 2 in bottom-right
    img_mbert_additional2 = mpimg.imread(plot_paths['4'])
    axs[1, 1].imshow(img_mbert_additional2)
    axs[1, 1].axis('off')

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout()
    
    # Save the combined plot
    plt.savefig(save_path, dpi=300)
    plt.show()

# Define the paths to your individual plots
plot_paths = {
    '1': '/netscratch/dgurgurov/thesis/src/cor_task_data/ner_vs_data_mbert.png',
    '2': '/netscratch/dgurgurov/thesis/src/cor_task_data/ner_vs_data_xlm-r.png',
    '3': '/netscratch/dgurgurov/thesis/src/cor_task_data/ner_vs_data_cn_mbert.png',  # New plot 1
    '4': '/netscratch/dgurgurov/thesis/src/cor_task_data/ner_vs_data_cn_xlm-r.png',  # New plot 2
}

# Save the combined plot
save_path = './ner_vs_data.png'
combine_plots(plot_paths, save_path)
