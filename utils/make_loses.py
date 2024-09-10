from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator # type: ignore
import math

def load_tensorboard_data(logdir):
    event_acc = event_accumulator.EventAccumulator(logdir)
    event_acc.Reload()
    return event_acc

def plot_metrics(ax, event_accs, metric_names, title, colors, styles=None):
    for (label, event_acc), color in zip(event_accs.items(), colors):
        for metric_name in metric_names:
            steps = [event.step for event in event_acc.scalars.Items(metric_name)]
            values = [event.value for event in event_acc.scalars.Items(metric_name)]
            style = styles[metric_name] if styles else '-'
            label_with_metric = f"{label} - {'Train' if 'train' in metric_name else 'Eval'}"
            ax.plot(steps, values, label=label_with_metric, color=color, linestyle=style)

    ax.set_title(title)
    ax.set_xlabel('Steps')
    ax.legend()

def visualize_multiple_languages(logdirs_dict, languages_per_page=10, save=False):
    num_languages = len(logdirs_dict)
    num_pages = (num_languages + languages_per_page - 1) // languages_per_page  # Ceiling division

    for page in range(num_pages):
        start_idx = page * languages_per_page
        end_idx = min(start_idx + languages_per_page, num_languages)

        languages_subset = list(logdirs_dict.keys())[start_idx:end_idx]
        logdirs_subset = {lang: logdirs_dict[lang] for lang in languages_subset}

        # Determine the number of rows required
        num_rows = math.ceil(len(languages_subset) / 2)

        # Create subplots grid with two columns (one column per language)
        fig, axs = plt.subplots(num_rows, 4, figsize=(20, 6 * num_rows))  # 4 subplots per row (2 languages, each with loss & accuracy)

        colors = ['blue', 'green', 'red']
        styles = {'train/loss': '-', 'eval/loss': '--'}

        for i, language in enumerate(languages_subset):
            logdirs = logdirs_subset[language]
            event_accs = {name: load_tensorboard_data(logdir) for name, logdir in logdirs.items()}

            row = i // 2  # Determine the row index
            col = (i % 2) * 2  # Determine the starting column index (0 or 2)

            # Plot losses and add the language code to the title
            plot_metrics(axs[row][col], event_accs, ['train/loss', 'eval/loss'], f'Losses - {language}', colors, styles)
            # Plot accuracies and add the language code to the title
            plot_metrics(axs[row][col + 1], event_accs, ['eval/accuracy'], f'Accuracies - {language}', colors)

        plt.tight_layout()

        if save:
            fig.savefig(f"metrics_page_{page + 1}.png", pad_inches=0.1, bbox_inches='tight', dpi=100)

        plt.show()

if __name__ == "__main__":
    languages = ['am', 'uz', 'su', 'cy', 'mr', 'te', 'ku', 'mk', 'bn', 'ka', 'sk', 'el', 'th', 'az', 'lv', 'sl', 
                 'he', 'ro', 'da', 'ur', 'si', 'yo', 'sw', 'ug', 'bo', 'mt', 'jv', 'ne', 'ms', 'bg']

    # Prepare the log directories for each language
    logdirs_dict = {}
    for language in languages:
        logdirs_dict[language] = {
            "LoRA": f"/netscratch/dgurgurov/thesis/lang_adapters/conceptnet/xlm-r/lora/{language}/logs",
            "Seq_bn": f"/netscratch/dgurgurov/thesis/lang_adapters/conceptnet/xlm-r/seq_bn/{language}/logs",
            "Seq_bn_inv": f"/netscratch/dgurgurov/thesis/lang_adapters/conceptnet/xlm-r/seq_bn_inv/{language}/logs"
        }

    # Visualize the tensorboard logs for multiple languages (10 languages per page)
    visualize_multiple_languages(logdirs_dict, languages_per_page=10, save=True)
