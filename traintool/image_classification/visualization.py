"""
Plots to visualize training results in tensorboard.
"""

# from tensorboardX import SummaryWriter
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import io


def plot_confusion_matrix() -> mpl.figure.Figure:
    """Plot confusion matrix to tensorboard."""
    fig = plt.figure()
    plt.plot([1, 3, 2])
    # writer.add_figure("confusion-matrix", plt.gcf(), epoch)
    return fig


def plot_samples(
    images: np.ndarray, labels: np.ndarray, predictions: np.ndarray,
) -> mpl.figure.Figure:
    """Plot a few sample images and classification results."""
    num_samples = len(images)
    num_classes = predictions.shape[1]

    fig, axes = plt.subplots(2, num_samples, figsize=(10, 3))
    # fig.suptitle("Samples and probabilites from train_data. Red is ground truth.")

    # Plot images.
    for i, ax in enumerate(axes[0]):
        plt.sca(ax)
        plt.axis("off")
        # TODO: Allow RGB (needs channels-last I think).
        plt.imshow(images[i][0], cmap="gray")

    # Plot predictions in bar charts.
    for i, ax in enumerate(axes[1]):
        plt.sca(ax)
        bars = ax.bar(np.arange(num_classes), predictions[i], zorder=3)
        bars[labels[i]].set_color("red")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xticks(np.arange(num_classes))
        # TODO: Set class labels via ax.set_xticklabels(class_names, rotation=0)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", zorder=0)
        ax.yaxis.set_tick_params(length=0)
        if i > 0:
            ax.yaxis.set_tick_params(labelleft=False)
        if i == 0:
            ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.get_xticklabels()[labels[i]].set_color("red")

    # Add common x label by drawing above the figure.
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel("Class (red = ground truth)")

    return fig


def figure_to_array(fig):
    """Convert matplotlib figure to RGBA numpy array (channels-first format)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="rgba")
    buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    buf.close()
    plt.close(fig)  # close here so it doesn't show up in jupyter notebook
    img_arr = img_arr.transpose((2, 0, 1))  # make channels first for tensorboardX
    return img_arr
