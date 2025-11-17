# load fonts
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np

# fontsize
FONTSIZE = 14
FONTSIZE_LEGEND = 13
TICKLABELSIZE = 10


def dup_last(a):
    return np.append(a, a[-1])


def plot_histogram(
    file,
    train,
    test,
    model,
    title,
    xlabel,
    xrange,
    model_label,
    logy=False,
    n_bins=60,
    error_range=[0.85, 1.15],
    error_ticks=[0.9, 1.0, 1.1],
    weights=None,
    mask_dict=None,
):
    """
    Plotting code used for all 1d distributions

    Parameters:
    file: str
    train: np.ndarray of shape (nevents)
    test: np.ndarray of shape (nevents)
    model: np.ndarray of shape (nevents)
    title: str
    xlabel: str
    xrange: tuple with 2 floats
    model_label: str
    logy: bool
    n_bins: int
    error_range: tuple with 2 floats
    error_ticks: tuple with 3 floats
    weights: np.ndarray of shape (nevents)
    mask_dict: dict
        mask (np.ndarray), condition (str)
    """
    # construct labels and colors
    labels = ["Train", "Test", model_label]
    colors = ["black", "#0343DE", "#A52A2A"]

    # construct histograms
    y_trn, bins = np.histogram(train, bins=n_bins, range=xrange)
    y_tst, _ = np.histogram(test, bins=bins)
    y_mod, _ = np.histogram(model, bins=bins)
    hists = [y_trn, y_tst, y_mod]
    hist_errors = [np.sqrt(y_trn), np.sqrt(y_tst), np.sqrt(y_mod)]

    if weights is not None:
        labels.append(f"Rew. {model_label}")
        colors.append("darkorange")
        assert model.shape == weights.shape
        y_weighted = np.histogram(model, bins=bins, weights=weights)[0]
        hists.append(y_weighted)
        hist_errors.append(np.sqrt(y_weighted))

    if mask_dict is not None:
        labels.append(f"{model_label} {mask_dict['condition']}")
        colors.append("violet")
        y_masked = np.histogram(model[mask_dict["mask"]], bins=bins)[0]
        hists.append(y_masked)
        hist_errors.append(np.sqrt(y_masked))

    integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
    scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]

    if mask_dict is None:
        _, axs = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(6, 4),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.00},
        )
    else:
        _, ax = plt.subplots(figsize=(6, 4))
        axs = [ax]

    for y, y_err, scale, label, color in zip(
        hists, hist_errors, scales, labels, colors, strict=False
    ):
        axs[0].step(
            bins,
            dup_last(y) * scale,
            label=label,
            color=color,
            linewidth=1.0,
            where="post",
        )
        axs[0].step(
            bins,
            dup_last(y + y_err) * scale,
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[0].step(
            bins,
            dup_last(y - y_err) * scale,
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[0].fill_between(
            bins,
            dup_last(y - y_err) * scale,
            dup_last(y + y_err) * scale,
            facecolor=color,
            alpha=0.3,
            step="post",
        )

        if label == "Train":
            axs[0].fill_between(
                bins,
                dup_last(y) * scale,
                0.0 * dup_last(y),
                facecolor=color,
                alpha=0.1,
                step="post",
            )
            continue

        if mask_dict is not None:
            continue

        ratio = (y * scale) / (hists[0] * scales[0])
        ratio_err = np.sqrt((y_err / y) ** 2 + (hist_errors[0] / hists[0]) ** 2)
        ratio_isnan = np.isnan(ratio)
        ratio[ratio_isnan] = 1.0
        ratio_err[ratio_isnan] = 0.0

        axs[1].step(bins, dup_last(ratio), linewidth=1.0, where="post", color=color)
        axs[1].step(
            bins,
            dup_last(ratio + ratio_err),
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[1].step(
            bins,
            dup_last(ratio - ratio_err),
            color=color,
            alpha=0.5,
            linewidth=0.5,
            where="post",
        )
        axs[1].fill_between(
            bins,
            dup_last(ratio - ratio_err),
            dup_last(ratio + ratio_err),
            facecolor=color,
            alpha=0.3,
            step="post",
        )

    axs[0].legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
    axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)

    if logy:
        axs[0].set_yscale("log")

    _, ymax = axs[0].get_ylim()
    axs[0].set_ylim(0.0, ymax)
    axs[0].set_xlim(xrange)
    axs[0].tick_params(axis="both", labelsize=TICKLABELSIZE)
    plt.xlabel(rf"${{{xlabel}}}$", fontsize=FONTSIZE)
    axs[0].text(
        0.02,
        0.95,
        s=title,
        horizontalalignment="left",
        verticalalignment="top",
        transform=axs[0].transAxes,
        fontsize=FONTSIZE,
    )

    if mask_dict is None:
        axs[1].set_ylabel(
            rf"$\frac{{\mathrm{{{model_label}}}}}{{\mathrm{{Test}}}}$", fontsize=FONTSIZE
        )
        axs[1].set_yticks(error_ticks)
        axs[1].set_ylim(error_range)
        axs[1].axhline(y=error_ticks[0], c="black", ls="dotted", lw=0.5)
        axs[1].axhline(y=error_ticks[1], c="black", ls="--", lw=0.7)
        axs[1].axhline(y=error_ticks[2], c="black", ls="dotted", lw=0.5)
        axs[1].tick_params(axis="both", labelsize=TICKLABELSIZE)

    plt.savefig(file, bbox_inches="tight", format="pdf")
    plt.close()


def plot_histogram_2d(
    file,
    test,
    model,
    title,
    xlabel,
    ylabel,
    xrange,
    yrange,
    model_label,
    n_bins=100,
):
    data = [test, model]
    weights = [None, None]
    subtitles = ["Test", model_label]

    fig, axs = plt.subplots(1, len(data), figsize=(4 * len(data), 4))
    for ax, dat, weight, subtitle in zip(axs, data, weights, subtitles, strict=False):
        ax.set_title(subtitle)
        ax.hist2d(
            dat[:, 0],
            dat[:, 1],
            bins=n_bins,
            range=[xrange, yrange],
            rasterized=True,
            weights=weight,
        )
        ax.set_xlabel(rf"${{{xlabel}}}$")
        ax.set_ylabel(rf"${{{ylabel}}}$")
    fig.suptitle(title)
    plt.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def plot_calibration(file, prob_true, prob_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prob_true, prob_pred, color="#A52A2A", marker="o", markersize=3, linewidth=1)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("classifier probability for true events", fontsize=FONTSIZE)
    ax.set_ylabel("true fraction of true events", fontsize=FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICKLABELSIZE)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    plt.savefig(file, bbox_inches="tight", format="pdf")
    plt.close()


def plot_roc(file, tpr, fpr, auc):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#A52A2A", linewidth=1.0)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("false positive rate", fontsize=FONTSIZE)
    ax.set_ylabel("true positive rate", fontsize=FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICKLABELSIZE)
    ax.text(
        0.95,
        0.05,
        f"AUC = {auc:.4f}",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
    )
    plt.savefig(file, format="pdf", bbox_inches="tight")
    plt.close()


def simple_histogram(file, data, labels, xrange, xlabel, logx=False, logy=False, n_bins=80):
    assert len(data) == 2 and len(labels) == 2
    colors = ["#0343DE", "#A52A2A"]

    data = [np.clip(data_i.clone(), xrange[0], xrange[1]) for data_i in data]
    if logx:
        data = [np.log(data_i) for data_i in data]
        xrange = np.log(xrange)

    bins = np.histogram(data[0], bins=n_bins, range=xrange)[1]
    hists = [np.histogram(data_i, bins=bins, range=xrange)[0] for data_i in data]
    integrals = [np.sum((bins[1:] - bins[:-1]) * y) for y in hists]
    scales = [1 / integral if integral != 0.0 else 1.0 for integral in integrals]
    if logx:
        bins = np.exp(bins)
        xrange = np.exp(xrange)

    fig, ax = plt.subplots(figsize=(5, 4))
    for y, scale, label, color in zip(hists, scales, labels, colors, strict=False):
        ax.step(
            bins,
            dup_last(y) * scale,
            label=label,
            color=color,
            linewidth=1.0,
            where="post",
        )
    ax.legend(loc="upper right", frameon=False, fontsize=FONTSIZE_LEGEND)
    ax.set_ylabel("Normalized", fontsize=FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)

    if logy:
        ax.set_yscale("log")
    else:
        _, ymax = ax.get_ylim()
        ax.set_ylim(0.0, ymax)
    if logx:
        ax.set_xscale("log")
    ax.set_xlim(xrange)
    ax.tick_params(axis="both", labelsize=TICKLABELSIZE)
    plt.savefig(file, bbox_inches="tight", format="pdf")
    plt.close()
