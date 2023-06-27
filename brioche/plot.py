import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams.update({"figure.autolayout": True})

rcParams["figure.figsize"] = (16, 9)

import numpy as np

import pandas as pd

import warnings

import seaborn as sns

from typing import List, Union, Dict, Any

colors = sns.color_palette("Set2")


def plotEnrichmentBars(pval_categories: dict, name: str = None) -> tuple:
    """
    Plots a bar chart to visualize the enrichment p-values of each category.

    Args:
        pval_categories (dict): A dictionary with categories as keys and their associated p-values as values.
        name (str, optional): The name of the file where the plot will be saved. If None, the plot will not be saved. Defaults to None.

    Returns:
        tuple: A tuple containing the Figure and Axes instances of the plot.
    """

    pvals_sorted = {
        n: v for n, v in sorted(list(pval_categories.items()), key=lambda x: x[1])
    }

    pvalsDF = pd.DataFrame({"cat": pvals_sorted.keys(), "pval": pvals_sorted.values()})

    fig, ax = plt.subplots()

    ax.bar(
        list(range(len(pvalsDF))),
        pvalsDF["pval"],
        tick_label=pvalsDF["cat"],
        color=colors[0],
    )

    ax.set_xticks(rotation=90)
    ax.set_ylabel("Enrichment p-value", fontsize=18)
    ax.set_xticks(fontsize=14)
    ax.set_yticks(fontsize=14)

    if name:
        plt.savefig(f"{name}.pdf", dpi=300)
        plt.savefig(f"{name}.png", dpi=300)

    return fig, ax


def plotModelHists(samples: dict, data: np.ndarray, name: str = ""):
    """
    Plots histograms for the frequency of rows and columns from the given samples.

    Args:
        samples (dict): A dictionary containing samples of frequency distributions.
        data (np.ndarray): The data to be modeled.
        name (str, optional): The name of the file where the plot will be saved. If empty, the plot will not be saved. Defaults to "".

    Returns:
        None
    """

    # Rows

    bins = np.linspace(np.min(samples["freq_rows"]), np.max(samples["freq_rows"]), 100)

    for i in samples["freq_rows"].squeeze().T:
        plt.hist(np.array(i), bins=bins, histtype="step")
    plt.axvline(np.mean(samples["freq_rows"]))

    plt.savefig(f"{name}freqRows.pdf")
    plt.clf()

    # Columns

    bins = np.linspace(np.min(samples["freq_cols"]), np.max(samples["freq_cols"]), 100)

    for i in samples["freq_cols"].squeeze().T:
        plt.hist(np.array(i), bins=bins, histtype="step")
    plt.axvline(np.mean(samples["freq_cols"]))

    plt.savefig(f"{name}freqCols.pdf")
    plt.clf()


def plotModelArrays(
    samples: dict,
    data: np.ndarray,
    x_labels: list = [],
    y_labels: list = [],
    name: str = "",
    likelihood_type: str = "sum",
):
    """
    Plot heatmap visualizations of the model parameters, normalized residuals and deviation parameters.

    Args:
        samples (dict): A dictionary containing samples of frequency distributions.
        data (np.ndarray): The data to be modeled.
        x_labels (list, optional): List of labels for the x-axis. Defaults to [].
        y_labels (list, optional): List of labels for the y-axis. Defaults to [].
        name (str, optional): The base name of the file where the plots will be saved. If empty, the plots will not be saved. Defaults to "".
        likelihood_type (str, optional): The type of likelihood function to use; "sum" or "prod". Defaults to "sum".

    Returns:
        None
    """

    # Parameters

    sns.heatmap(
        np.mean(samples["freq"], 0),
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap="viridis",
    )
    plt.xticks(rotation=90)
    plt.savefig(f"{name}freqMean.pdf")
    plt.clf()

    sns.heatmap(
        np.std(samples["freq"], 0),
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap="viridis",
    )
    plt.xticks(rotation=90)
    plt.savefig(f"{name}freqStd.pdf")
    plt.clf()

    # Normalised residuals

    pred_mean = np.mean(np.random.poisson(np.clip(samples["freq"], 0, np.inf)))
    pred_std = np.std(np.random.poisson(np.clip(samples["freq"], 0, np.inf)))

    sig = np.array((pred_mean - data) / pred_std)

    sns.heatmap(sig, xticklabels=x_labels, yticklabels=y_labels, cmap="PRGn")
    plt.xticks(rotation=90)
    plt.savefig(f"{name}residuals.pdf")
    plt.clf()

    # Deviation parameters

    pred_mean = (
        (np.mean(samples["freq_dev"], 0) - 1.0)
        if likelihood_type == "prod"
        else np.mean(samples["freq_dev"], 0)
    )
    pred_std = np.std(samples["freq_dev"], 0)

    dev = np.array(pred_mean / pred_std)

    sns.heatmap(dev, xticklabels=x_labels, yticklabels=y_labels, cmap="PRGn")
    plt.xticks(rotation=90)
    plt.savefig(f"{name}deviations.pdf")
    plt.clf()


def plotDeviations(
    samples: dict,
    threshold: float = 2,
    x_labels: list = [],
    y_labels: list = [],
    name: str = "",
    likelihood_type: str = "sum",
):
    """
    Plots kernel density estimate (KDE) plots of the deviation parameters and saves them in PDF and PNG formats.

    Args:
        samples (dict): A dictionary containing samples of frequency distributions.
        threshold (float, optional): The threshold value for determining the 'best' cells based on absolute z-score. Defaults to 2.
        x_labels (list, optional): List of labels for the x-axis. Defaults to [].
        y_labels (list, optional): List of labels for the y-axis. Defaults to [].
        name (str, optional): The base name of the file where the plots will be saved. If empty, the plots will not be saved. Defaults to "".
        likelihood_type (str, optional): The type of likelihood function to use; "sum" or "prod". Defaults to "sum".

    Returns:
        None
    """

    dataDict = {}
    for i in range(samples["freq_dev"].shape[1]):
        for j in range(samples["freq_dev"].shape[2]):
            dataDict[f"{y_labels[i]}_{x_labels[j]}"] = list(
                np.array(samples["freq_dev"][:, i, j])
            )

    df = pd.DataFrame(dataDict).melt(value_name="deviation", var_name="cell")

    stats = df.groupby("cell").agg({"deviation": ["mean", "std"]}).reset_index()

    stats.columns = ["_".join(col).strip("_") for col in stats.columns.values]

    stats["z"] = stats["deviation_mean"] / stats["deviation_std"]
    stats["abs_z"] = np.abs(stats["z"])

    df_stats = df.reset_index(drop=True).merge(
        stats.reset_index(drop="True"), on="cell"
    )

    # Get the best (> threshold) cells and set their colour

    bestCells = list(stats[np.abs(stats["z"]) > threshold]["cell"].values)
    otherCells = list(stats[np.abs(stats["z"]) < threshold]["cell"].values)
    colours = {}
    for c in otherCells:
        colours[c] = sns.color_palette("Blues")[0]
    for i, c in enumerate(bestCells):
        colours[c] = sns.color_palette("Set2")[i]

    # Split data and plot separately

    bestData = df_stats[df_stats["cell"].isin(bestCells)]
    otherData = df_stats[df_stats["cell"].isin(otherCells)]

    sns.kdeplot(
        data=otherData.sort_values("abs_z", ascending=False),
        x="deviation",
        hue="cell",
        legend=False,
        palette=colours,
        hue_order=stats.sort_values("abs_z", ascending=False)["cell"].values,
        linewidth=1,
        common_norm=False,
    )
    g = sns.kdeplot(
        data=bestData.sort_values("abs_z", ascending=False),
        x="deviation",
        hue="cell",
        legend=True,
        palette=colours,
        hue_order=stats.sort_values("abs_z", ascending=False)["cell"].values,
        linewidth=3,
        common_norm=False,
    )

    # A hack so that only the 'best' cells are annotated

    warnings.filterwarnings("ignore")

    plt.legend(labels=["_" for c in otherCells] + [c for c in bestCells])

    plt.ylabel("Density", fontsize=18)
    plt.xlabel("Deviation", fontsize=18)

    g.set_yticklabels(g.get_yticks(), size=14)
    g.set_xticklabels(g.get_xticks(), size=14)

    warnings.resetwarnings()

    plt.savefig(f"{name}deviations-kde.pdf")
    plt.savefig(f"{name}deviations-kde.png", dpi = 300)
    plt.clf()
