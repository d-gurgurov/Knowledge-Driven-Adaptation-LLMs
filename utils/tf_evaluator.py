from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator # type: ignore

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
            # Update the label to include both the adapter type and the type of loss (train/eval)
            label_with_metric = f"{label} - {'Train' if 'train' in metric_name else 'Eval'}"
            ax.plot(steps, values, label=label_with_metric, color=color, linestyle=style)

    ax.set_title(title)
    ax.set_xlabel('Steps')
    ax.legend()

def visualize_tensorboard_logs(logdirs, save=False, save_path=None):
    event_accs = {name: load_tensorboard_data(logdir) for name, logdir in logdirs.items()}

    # Print all available metrics for each log
    """
    for name, event_acc in event_accs.items():
        print(f"Available metrics for {name}:")
        print(event_acc.scalars.Keys())
        print()
    """

    # Specify the metrics to plot
    loss_metrics = ['train/loss', 'eval/loss']
    accuracy_metrics = ['eval/accuracy']  # Update this based on the actual keys printed above

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Define colors and line styles for loss metrics
    colors = ['blue', 'green', 'red']  # Colors for each adapter type
    styles = {'train/loss': '-', 'eval/loss': '--'}  # Solid for train, dashed for eval

    # Plot all losses on the left subplot with the same color for train and eval but different line styles
    plot_metrics(axs[0], event_accs, loss_metrics, 'Losses', colors, styles) # type: ignore

    # Plot all accuracies on the right subplot (unchanged)
    plot_metrics(axs[1], event_accs, accuracy_metrics, 'Accuracies', colors) # type: ignore

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(save_path, pad_inches=0.1, bbox_inches='tight', dpi=100) # type: ignore

if __name__ == "__main__":

    languages = ['am', 'uz', 'su', 'cy', 'mr', 'te', 'ku', 'mk', 'bn', 'ka', 'sk', 'el', 'th', 'az', 'lv', 'sl', 
                            'he', 'ro', 'da', 'ur', 'si', 'yo', 'sw', 'ug', 'bo', 'mt', 'jv', 'ne', 'ms', 'bg']

    for language in languages:

        logdirs = {
            "LoRA": f"/netscratch/dgurgurov/thesis/lang_adapters/conceptnet/mbert/lora/{language}/logs",
            "Seq_bn": f"/netscratch/dgurgurov/thesis/lang_adapters/conceptnet/mbert/seq_bn/{language}/logs",
            "Seq_bn_inv": f"/netscratch/dgurgurov/thesis/lang_adapters/conceptnet/mbert/seq_bn_inv/{language}/logs"
        }

        visualize_tensorboard_logs(logdirs, save=True, save_path=f"mbert_cn_{language}.png")
