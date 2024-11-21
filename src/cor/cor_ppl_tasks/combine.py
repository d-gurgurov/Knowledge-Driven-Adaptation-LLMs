import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def combine_plots(plot_paths, save_path):
    # Create a figure with a 2x3 grid for the plots
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))

    # Plot mBERT (top row)
    img_mbert_ner = mpimg.imread(plot_paths['mbert_ner'])
    axs[0, 0].imshow(img_mbert_ner)
    axs[0, 0].axis('off')
    
    img_mbert_sa = mpimg.imread(plot_paths['mbert_sa'])
    axs[0, 1].imshow(img_mbert_sa)
    axs[0, 1].axis('off')
    
    img_mbert_tc = mpimg.imread(plot_paths['mbert_tc'])
    axs[0, 2].imshow(img_mbert_tc)
    axs[0, 2].axis('off')
    
    # Plot XLM-R (bottom row)
    img_xlm_ner = mpimg.imread(plot_paths['xlmr_ner'])
    axs[1, 0].imshow(img_xlm_ner)
    axs[1, 0].axis('off')
    
    img_xlm_sa = mpimg.imread(plot_paths['xlmr_sa'])
    axs[1, 1].imshow(img_xlm_sa)
    axs[1, 1].axis('off')
    
    img_xlm_tc = mpimg.imread(plot_paths['xlmr_tc'])
    axs[1, 2].imshow(img_xlm_tc)
    axs[1, 2].axis('off')

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout()
    
    # Save the combined plot
    plt.savefig(save_path, dpi=300)
    plt.show()

# Define the paths to your individual plots
plot_paths = {
    'mbert_ner': '/netscratch/dgurgurov/thesis/src/cor_ppl_tasks/base_cor_ner_mbert_log_with_adapted.png',
    'mbert_sa': '/netscratch/dgurgurov/thesis/src/cor_ppl_tasks/base_cor_sa_mbert_log_with_adapted.png',
    'mbert_tc': '/netscratch/dgurgurov/thesis/src/cor_ppl_tasks/base_cor_tc_mbert_log_with_adapted.png',
    'xlmr_ner': '/netscratch/dgurgurov/thesis/src/cor_ppl_tasks/base_cor_ner_xlm-r_log_with_adapted.png',
    'xlmr_sa': '/netscratch/dgurgurov/thesis/src/cor_ppl_tasks/base_cor_sa_xlm-r_log_with_adapted.png',
    'xlmr_tc': '/netscratch/dgurgurov/thesis/src/cor_ppl_tasks/base_cor_tc_xlm-r_log_with_adapted.png',
}

# Save the combined plot
save_path = './base_cor_tasks.png'
combine_plots(plot_paths, save_path)
