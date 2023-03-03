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

import seaborn as sns
colors = sns.color_palette("Set2")

def plotEnrichmentBars(pval_categories, name = None):

    pvals_sorted = {n : v for n, v in sorted(list(pval_categories.items()), key = lambda x : x[1])}

    pvalsDF = pd.DataFrame({'cat' : pvals_sorted.keys(), 'pval' : pvals_sorted.values()})

    fig,ax = plt.subplots()

    ax.bar(list(range(len(pvalsDF))), pvalsDF['pval'], tick_label = pvalsDF['cat'], color = colors[0])

    ax.set_xticks(rotation = 90);
    ax.set_ylabel('Enrichment p-value', fontsize = 18);
    ax.set_xticks(fontsize=14);
    ax.set_yticks(fontsize=14);

    if name:
        plt.savefig(f'{name}.pdf', dpi = 300)
        plt.savefig(f'{name}.png', dpi = 300)

    return fig, ax

def plotModelHists(samples, data, name = ''):

    # Rows

    bins = np.linspace(np.min(samples['freq_rows']), np.max(samples['freq_rows']), 100)

    for i in samples['freq_rows'].squeeze().T:
        plt.hist(np.array(i), bins = bins, histtype = 'step')
    plt.axvline(np.mean(samples['freq_rows']))

    plt.savefig(f'{name}freqRows.pdf')
    plt.clf()

    # Columns

    bins = np.linspace(np.min(samples['freq_cols']), np.max(samples['freq_cols']), 100)

    for i in samples['freq_cols'].squeeze().T:
        plt.hist(np.array(i), bins = bins, histtype = 'step')
    plt.axvline(np.mean(samples['freq_cols']))

    plt.savefig(f'{name}freqCols.pdf')
    plt.clf()

def plotModelArrays(samples, data, x_labels = [], y_labels = [], name = ''):

    # Parameters

    sns.heatmap(np.mean(samples['freq'], 0), xticklabels = x_labels, yticklabels = y_labels, cmap = 'viridis')
    plt.xticks(rotation = 90)
    plt.savefig(f'{name}freqMean.pdf')
    plt.clf()

    sns.heatmap(np.std(samples['freq'], 0), xticklabels = x_labels, yticklabels = y_labels, cmap = 'viridis')
    plt.xticks(rotation = 90)
    plt.savefig(f'{name}freqsStd.pdf')
    plt.clf()

    # Normalised residuals

    pred_mean = np.mean(np.random.poisson(samples['freq']))
    pred_std = np.std(np.random.poisson(samples['freq']))

    sig = np.array((pred_mean - data)  / pred_std)

    sns.heatmap(sig, xticklabels = x_labels, yticklabels = y_labels, cmap = 'PRGn')
    plt.xticks(rotation = 90)
    plt.savefig(f'{name}residuals.pdf')
    plt.clf()

    # Deviation parameters

    pred_mean = np.mean(samples['freq_dev'], 0) - 1.
    pred_std = np.std(samples['freq_dev'], 0)

    dev = np.array(pred_mean / pred_std)

    sns.heatmap(dev, xticklabels = x_labels, yticklabels = y_labels, cmap = 'PRGn')
    plt.xticks(rotation = 90)
    plt.savefig(f'{name}deviations.pdf')
    plt.clf()
