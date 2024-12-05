import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def combine_plots(plot_paths, save_path):
    # Create a figure with a 2x2 grid for the plots
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Plot mBERT in top-left
    img_mbert_cn = mpimg.imread(plot_paths['1'])
    axs[0].imshow(img_mbert_cn)
    axs[0].axis('off')

    # Plot XLM-R in top-right
    img_mbert_glot = mpimg.imread(plot_paths['2'])
    axs[1].imshow(img_mbert_glot)
    axs[1].axis('off')

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout()
    
    # Save the combined plot
    plt.savefig(save_path, dpi=300)
    plt.show()

# Define the paths to your individual plots
plot_paths = {
    '1': 'sa_vs_training_exs_mbert_with_fit.png',
    '2': 'sa_vs_training_exs_xlm-r_with_fit.png',
}

# Save the combined plot
save_path = './sa_vs_data.png'
combine_plots(plot_paths, save_path)
