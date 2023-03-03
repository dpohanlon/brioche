import numpy as np

import numpyro
import numpyro.distributions as dist

import jax.numpy as jnp

from jax import random
from numpyro.infer import MCMC, NUTS

from jax.config import config; config.update("jax_enable_x64", True)

import pandas as pd

from pprint import pprint

from brioche.models import multisetModel

class MultisetEnrichment(object):
    """Two-way test for enrichment on nxm mutually-exclusive sets. """

    def __init__(self, data, row_names = [], col_names = [], row_constraint = False, col_constraint = False):

        self.data = data
        self.row_names = row_names
        self.col_names = col_names

        if len(data.shape) != 2:
            raise ValueError('Multiset enrichment is only for 2d arrays of counts!')

        if isinstance(data, float):
            data = data.astype(np.int)

        self.col_constraint = col_constraint
        self.row_constraint = row_constraint

        self.priors = self.getPriors(self.data)

        self.model = multisetModel

    def getPriors(self, data, dev_mean = 1.0, dev_std = 0.1):

        # Just use the row/col means and stds to init the priors

        rowMeans = np.mean(np.sqrt(data), axis = 1)
        colMeans = np.mean(np.sqrt(data), axis = 0)

        rowStds = np.std(np.sqrt(data), axis = 1)
        colStds = np.std(np.sqrt(data), axis = 0)

        priors = {}

        priors['row_means'] = rowMeans.flatten()
        priors['col_means'] = colMeans.flatten()

        priors['row_stds'] = rowStds.flatten()
        priors['col_stds'] = colStds.flatten()

        priors['dev_mean'] = dev_mean
        priors['dev_std'] = dev_std

        return priors

    def runMCMC(self, num_samples = 10000):

        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)

        num_warmup = num_samples // 10

        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_warmup = num_warmup, num_samples = num_samples)

        mcmc.run(rng_key_, self.data, self.priors, self.row_constraint, self.col_constraint)

        # mcmc.print_summary()
        samples = mcmc.get_samples()

        return samples

    def getSummary(self, samples):

        xs, ys = np.meshgrid(self.row_names, self.col_names)

        pred_mean = np.mean(samples['freq_dev'], 0) - 1.
        pred_std = np.std(samples['freq_dev'], 0)

        dev = np.array(pred_mean / pred_std)

        dataList = list(zip(xs.flatten(), ys.flatten(), dev.flatten()))
        sortedList = sorted(dataList, key = lambda x : np.abs(x[2]), reverse = True)

        print('Top 10 enrichments:')
        pprint(sortedList[:10])

        # Add more model data here?

        results = pd.DataFrame({'Rows' : [x[0] for x in sortedList], 'Cols' : [x[1] for x in sortedList], 'Deviance significance' : [x[2] for x in sortedList]})

        return results
