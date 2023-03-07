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

from statsmodels.stats.multitest import fdrcorrection


def countOcurrances(categoryIndices, data):

    n_categories = len(categoryIndices)

    observations = np.zeros(n_categories)

    dataCats, dataCounts = map(list, np.unique(data, return_counts=True))

    for cat, count in list(zip(dataCats, dataCounts)):
        observations[categoryIndices[cat]] = count

    return observations


def chi2(data, expected, signed=False):

    val = (data - expected) ** 2 / expected

    if signed:
        return np.sign(data - expected) * val
    else:
        return val


def chi2Statistic(data, expected):

    return np.sum(np.atleast_2d(chi2(data, expected)), axis=1).squeeze()


def calculateExpectation(occurances, nObservations):

    # If the categories are equally represented, the expectation is that
    # this is just a scaling of the 'underlying' reference distribution

    return (occurances / np.sum(occurances)) * nObservations


def nullDistribution(nObservations, reference, nSamples=1):

    nullProbs = reference / np.sum(reference)

    samples = np.random.multinomial(nObservations, nullProbs, size=nSamples)

    return samples


def fisherTest2x2(observations, refOcurrances, catIdx):

    obs = observations[catIdx]
    ref = refOcurrances[catIdx]

    table = np.zeros((2, 2))
    table[0][0] = obs
    table[0][1] = ref
    table[1][0] = np.sum(observations) - obs
    table[1][1] = np.sum(refOcurrances) - ref

    _, pval = fisher_exact(table)

    return pval


def enrichment(reference, test, nTestSamples=1000):

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
