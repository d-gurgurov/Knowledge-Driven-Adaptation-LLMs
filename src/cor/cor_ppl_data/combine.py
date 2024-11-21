import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def combine_plots(plot_paths, save_path):
    # Create a figure with a 1x2 grid for the plots
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    # Plot mBERT in top-left
    img_mbert_cn = mpimg.imread(plot_paths['1'])
    axs[0].imshow(img_mbert_cn)
    axs[0].axis('off')

    # Plot XLM-R in bottom-left
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
    '1': '/netscratch/dgurgurov/thesis/src/cor_ppl_data/ppl_vs_wiki_colored.png',
    '2': '/netscratch/dgurgurov/thesis/src/cor_ppl_data/ppl_vs_cc_colored.png',
}

# Save the combined plot
save_path = './ppl_vs_size.png'
combine_plots(plot_paths, save_path)
