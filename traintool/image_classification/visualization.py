"""
Plots to visualize training results in tensorboard.
"""

# from tensorboardX import SummaryWriter
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import io
from tensorboardX import SummaryWriter


def plot_confusion_matrix(writer: SummaryWriter, name: str, epoch: int) -> None:
    """Plot confusion matrix to tensorboard."""
    fig = plt.figure()
    plt.plot([1, 3, 2])
    writer.add_image(name, figure_to_array(fig), epoch)


def plot_samples(
    writer: SummaryWriter,
    name: str,
    epoch: int,
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> None:
    """Plot a few sample images and classification results to tensorboard."""
    num_samples = len(images)
    num_classes = predictions.shape[1]
    #print(num_samples)

    fig, axes = plt.subplots(2, num_samples, figsize=(10, 3))
    # fig.suptitle("Samples and probabilites from train_data. Red is ground truth.")

    # Scale images to [0, 1] (they may have another range for classification).
    images = (images - np.min(images)) / np.ptp(images)

    # Plot images.
    for i, ax in enumerate(axes[0]):
        plt.sca(ax)
        plt.axis("off")
        if images.shape[1] == 1:  # grayscale
            plt.imshow(images[i][0], cmap="gray")
        elif images.shape[1] == 3:  # RGB
            plt.imshow(images[i].transpose(1, 2, 0))
        else:
            raise RuntimeError()

    # Plot predictions in bar charts.
    for i, ax in enumerate(axes[1]):
        plt.sca(ax)
        bars = ax.bar(np.arange(num_classes), predictions[i], zorder=3)
        bars[int(labels[i])].set_color("red")
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
        ax.get_xticklabels()[int(labels[i])].set_color("red")

    # Add common x label by drawing above the figure.
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel("Class (red = ground truth)")

    writer.add_image(name, figure_to_array(fig), epoch)


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
