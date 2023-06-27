"""

Brioche: Tests for set enrichment.

Test a n x 2 (n categories with 2 outcomes) contingency table with fixed marginals.

Test n categories in two datasets ('test' vs 'reference'), assuming that the number
of total 'reference' and total 'test' observations are fixed (but categories are
free to vary within them, up to the maximum in 'reference').

Performs a test using the chi-squared test statistic, either with a resampled
distribution, or assuming a chi-squared distribution (>~ 5 counts per category).

"""

import numpy as np

from scipy.stats import chisquare as chisquare_test
from scipy.stats import fisher_exact

from typing import List, Union, Dict, Any, Tuple

from statsmodels.stats.multitest import fdrcorrection


def countOcurrances(categoryIndices: dict, data: np.ndarray) -> np.ndarray:
    """
    Counts the occurrences of each category in the data.

    Args:
    categoryIndices (dict): A dictionary mapping categories to indices.
    data (np.ndarray): The data, where each element is a category.

    Returns:
    np.ndarray: An array where the i-th element is the count of the i-th category.
    """

    n_categories = len(categoryIndices)

    observations = np.zeros(n_categories)

    dataCats, dataCounts = map(list, np.unique(data, return_counts=True))

    for cat, count in list(zip(dataCats, dataCounts)):
        observations[categoryIndices[cat]] = count

    return observations


def chi2(data: np.ndarray, expected: np.ndarray, signed: bool=False) -> np.ndarray:
    """
    Calculate the Chi-square test statistic.

    Args:
    data (np.ndarray): The observed data.
    expected (np.ndarray): The expected data.
    signed (bool, optional): Whether to return a signed value. Defaults to False.

    Returns:
    np.ndarray: The Chi-square test statistic for each element.
    """

    val = (data - expected) ** 2 / expected

    if signed:
        return np.sign(data - expected) * val
    else:
        return val


def chi2Statistic(data: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """
    Compute the total Chi-square test statistic.

    Args:
    data (np.ndarray): The observed data.
    expected (np.ndarray): The expected data.

    Returns:
    np.ndarray: The total Chi-square test statistic.
    """

    return np.sum(np.atleast_2d(chi2(data, expected)), axis=1).squeeze()


def calculateExpectation(occurances: np.ndarray, nObservations: int) -> np.ndarray:
    """
    Calculate the expected occurrences based on a reference distribution.

    Args:
    occurances (np.ndarray): The occurrences in the reference data.
    nObservations (int): The total number of observations.

    Returns:
    np.ndarray: The expected occurrences.
    """

    # If the categories are equally represented, the expectation is that
    # this is just a scaling of the 'underlying' reference distribution

    return (occurances / np.sum(occurances)) * nObservations


def nullDistribution(nObservations: int, reference: np.ndarray, nSamples: int=1) -> np.ndarray:
    """
    Generate samples from the null distribution.

    Args:
    nObservations (int): The total number of observations.
    reference (np.ndarray): The reference data.
    nSamples (int, optional): The number of samples to generate. Defaults to 1.

    Returns:
    np.ndarray: The sampled data.
    """

    nullProbs = reference / np.sum(reference)

    samples = np.random.multinomial(nObservations, nullProbs, size=nSamples)

    return samples


def fisherTest2x2(observations: np.ndarray, refOcurrances: np.ndarray, catIdx: int) -> float:
    """
    Perform a Fisher's exact test for a 2x2 contingency table.

    Args:
    observations (np.ndarray): The observed data.
    refOcurrances (np.ndarray): The reference occurrences.
    catIdx (int): The index of the category to test.

    Returns:
    float: The p-value from the test.
    """

    obs = observations[catIdx]
    ref = refOcurrances[catIdx]

    table = np.zeros((2, 2))
    table[0][0] = obs
    table[0][1] = ref
    table[1][0] = np.sum(observations) - obs
    table[1][1] = np.sum(refOcurrances) - ref

    _, pval = fisher_exact(table)

    return pval


def enrichment(reference: np.ndarray, test: np.ndarray, nTestSamples: int=1000) -> Tuple:
    """
    Perform an enrichment analysis.

    Args:
    reference (np.ndarray): The reference data.
    test (np.ndarray): The test data.
    nTestSamples (int, optional): The number of test samples to generate. Defaults to 1000.

    Returns:
    tuple: A tuple containing the Chi-square statistic, Chi-square values per category,
           p-values per category, and the Chi-square statistic for the null distribution.
    """

    categories = np.unique(reference)

    # Category strings to indices
    categories_to_index = {c: i for i, c in enumerate(categories)}

    # Array of observation counts for category (index) i
    observations = countOcurrances(categories_to_index, test)
    nObservations = np.sum(observations)

    refOcurrances = countOcurrances(categories_to_index, reference)
    expectations = calculateExpectation(refOcurrances, nObservations)

    chi2Stat = chi2Statistic(observations, expectations)

    nullSamples = nullDistribution(nObservations, refOcurrances, nTestSamples)
    nullChi2Stats = chi2Statistic(nullSamples, expectations)

    pval = len(nullChi2Stats[nullChi2Stats > chi2Stat]) / float(len(nullChi2Stats))

    pval_chi2 = chisquare_test(observations, expectations)

    # Test individual categories also

    chi2vals = chi2(observations, expectations, signed=True)

    chi2_categories = {categories[i]: chi2vals[i] for i in range(len(categories))}

    pvals = [
        fisherTest2x2(observations, refOcurrances, i) for i in range(len(observations))
    ]

    _, pvals_corrected = fdrcorrection(pvals)

    # pval_categories = {categories[i] : fisherTest2x2(observations, refOcurrances, i) for i in range(len(observations))}
    pval_categories = {
        categories[i]: pvals_corrected[i] for i in range(len(observations))
    }

    return chi2Stat, chi2_categories, pval_categories, nullChi2Stats
