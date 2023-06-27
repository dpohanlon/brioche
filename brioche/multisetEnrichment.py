import numpy as np

import numpyro
import numpyro.distributions as dist

import jax.numpy as jnp

from jax import random
from numpyro.infer import MCMC, NUTS

from jax.config import config

config.update("jax_enable_x64", True)

import pandas as pd

from pprint import pprint

from typing import List, Union, Dict, Any

from brioche.models import multisetModel


class MultisetEnrichment(object):
    """
    A class for performing a two-way test for enrichment on nxm mutually-exclusive sets.

    This class uses Markov Chain Monte Carlo (MCMC) methods to perform the enrichment analysis,
    providing methods to compute priors, run the MCMC process, and summarize the results.

    Attributes:
    likelihood_type (str): The type of likelihood to use, either "sum" or "prod".
    data (np.ndarray): The data used for the analysis.
    row_names (List[str]): The names of the rows in the data.
    col_names (List[str]): The names of the columns in the data.
    col_constraint (bool): Whether or not a column constraint is applied.
    row_constraint (bool): Whether or not a row constraint is applied.
    priors (Dict[str, Union[float, np.ndarray]]): The priors used in the MCMC process.
    model (Callable): The model used in the MCMC process.

    Methods:
    __init__(data, row_names, col_names, row_constraint, col_constraint, likelihood_type):
        Initializes the MultisetEnrichment object.
    getPriors(data, dev_mean, dev_std): Computes the priors for the model using the data.
    runMCMC(num_samples): Runs the MCMC process and returns the samples.
    getSummary(samples): Generates a summary of the MCMC samples and returns it as a DataFrame.
    """

    def __init__(
        self,
        data: np.ndarray,
        row_names: List[str] = [],
        col_names: List[str] = [],
        row_constraint: bool = False,
        col_constraint: bool = False,
        likelihood_type: str = "sum",
    ) -> None:
        """
        Initializes an instance of the MultisetEnrichment class.

        Parameters:
        data (np.ndarray): The input data.
        row_names (List[str], optional): The row names. Defaults to an empty list.
        col_names (List[str], optional): The column names. Defaults to an empty list.
        row_constraint (bool, optional): If there is a row constraint. Defaults to False.
        col_constraint (bool, optional): If there is a column constraint. Defaults to False.
        likelihood_type (str, optional): The likelihood type, "sum" or "prod". Defaults to "sum".
        """

        self.likelihood_type = likelihood_type

        self.data = data
        self.row_names = row_names
        self.col_names = col_names

        if len(data.shape) != 2:
            raise ValueError("Multiset enrichment is only for 2d arrays of counts!")

        if isinstance(data, float):
            data = data.astype(np.int)

        self.col_constraint = col_constraint
        self.row_constraint = row_constraint

        self.priors = self.getPriors(
            self.data, dev_mean=0.0 if self.likelihood_type == "sum" else 1.0
        )

        self.model = multisetModel

    def getPriors(self, data: np.ndarray, dev_mean: float = 0.0, dev_std: float = 0.01) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute the priors for the model using the input data.

        Parameters:
        data (np.ndarray): The input data.
        dev_mean (float, optional): The deviation mean. Defaults to 0.0.
        dev_std (float, optional): The deviation standard deviation. Defaults to 0.01.

        Returns:
        dict: The calculated priors.
        """

        # Just use the row/col means and stds to init the priors

        if self.likelihood_type == "prod":

            rowMeans = np.mean(np.sqrt(data), axis=1)
            colMeans = np.mean(np.sqrt(data), axis=0)

            rowStds = np.std(np.sqrt(data), axis=1)
            colStds = np.std(np.sqrt(data), axis=0)

        else:

            rowMeans = np.mean(data / 2.0, axis=1)
            colMeans = np.mean(data / 2.0, axis=0)

            rowStds = np.std(data / 2.0, axis=1)
            colStds = np.std(data / 2.0, axis=0)

        priors = {}

        priors["row_means"] = rowMeans.flatten()
        priors["col_means"] = colMeans.flatten()

        priors["row_stds"] = rowStds.flatten()
        priors["col_stds"] = colStds.flatten()

        priors["dev_mean"] = dev_mean
        priors["dev_std"] = dev_std

        if self.likelihood_type == "prod":

            priors["dev_mean"] = data / (
                rowMeans.reshape(-1, 1) @ colMeans.reshape(-1, 1).T
            )

        else:

            priors["dev_mean"] = data - (
                rowMeans.reshape(-1, 1) + colMeans.reshape(-1, 1).T
            )

        priors["dev_std"] = np.max(priors["dev_mean"])

        self.priors = priors

        return priors

    def runMCMC(self, num_samples: int = 10000) -> Dict[str, np.ndarray]:
        """
        Runs the Markov chain Monte Carlo (MCMC) process.

        Parameters:
        num_samples (int, optional): The number of samples. Defaults to 10000.

        Returns:
        dict: The samples obtained from the MCMC process.
        """

        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)

        num_warmup = num_samples // 10

        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)

        mcmc.run(
            rng_key_,
            self.data,
            self.priors,
            self.likelihood_type,
            self.row_constraint,
            self.col_constraint,
        )

        samples = mcmc.get_samples()

        return samples

    def getSummary(self, samples: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Generates a summary of the MCMC samples.

        Parameters:
        samples (dict): The samples obtained from the MCMC process.

        Returns:
        pd.DataFrame: A DataFrame containing the results.
        """

        xs, ys = np.meshgrid(self.row_names, self.col_names)

        pred_mean = (
            (np.mean(samples["freq_dev"], 0) - 1.0)
            if self.likelihood_type == "prod"
            else np.mean(samples["freq_dev"], 0)
        )
        pred_std = np.std(samples["freq_dev"], 0)

        dev = np.array(pred_mean / pred_std)

        dataList = list(zip(xs.flatten(), ys.flatten(), dev.flatten()))
        sortedList = sorted(dataList, key=lambda x: np.abs(x[2]), reverse=True)

        print("Top 10 enrichments:")
        pprint(sortedList[:10])

        # Add more model data here?

        results = pd.DataFrame(
            {
                "Rows": [x[0] for x in sortedList],
                "Cols": [x[1] for x in sortedList],
                "Deviance significance": [x[2] for x in sortedList],
            }
        )

        return results
