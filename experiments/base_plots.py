import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 14
FONTSIZE_LEGEND = 13
FONTSIZE_TICK = 12


def plot_loss(file, losses, lr=None, labels=None, logy=True):
    if len(losses[1]) == 0:  # catch no-validations case
        losses = [losses[0]]
        labels = [labels[0]]
    labels = [None for _ in range(len(losses))] if labels is None else labels
    iterations = range(1, len(losses[0]) + 1)
    fig, ax = plt.subplots()
    for loss, label in zip(losses, labels, strict=False):
        if len(loss) == len(iterations):
            its = iterations
        else:
            frac = len(losses[0]) / len(loss)
            its = np.arange(1, len(loss) + 1) * frac
        ax.plot(its, loss, label=label)

    if logy:
        ax.set_yscale("log")
    if lr is not None:
        axright = ax.twinx()
        axright.plot(iterations, lr, label="learning rate", color="crimson")
        axright.set_ylabel("Learning rate", fontsize=FONTSIZE)
    ax.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    ax.set_ylabel("Loss", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper right")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_metric(file, metrics, metric_label, labels=None, logy=False):
    labels = [None for _ in range(len(metrics))] if labels is None else labels
    fig, ax = plt.subplots()
    for metric, label in zip(metrics, labels, strict=False):
        iterations = range(1, len(metric) + 1)
        if len(metric) == len(iterations):
            its = iterations
        else:
            frac = len(metrics[0]) / len(metric)
            its = np.arange(1, len(metric) + 1) * frac
        ax.plot(its, metric, label=label)

    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(metric_label, fontsize=FONTSIZE)
    ax.set_xlabel("Number of iterations", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE_LEGEND, frameon=False, loc="upper left")
    fig.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()
