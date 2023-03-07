import numpy as np

from brioche.multisetEnrichment import MultisetEnrichment

from brioche.plot import plotModelArrays, plotModelHists


def runAll():

    testNoConstraintProd()
    testNoConstraintSum()


def testNoConstraintProd():

    nx = 30
    ny = 15

    means_rows = np.ones((nx, 1))
    means_cols = np.ones((ny, 1))

    means_rows[1] += 0.1
    means_rows[3] += 0.1
    means_cols[2] -= 0.1

    means = means_rows @ means_cols.T

    means *= 100

    means[7, 7] += 50
    means[3, 5] -= 50

    data = np.random.poisson(means).astype(int)

    row_names = [str(x) for x in np.random.randint(0, 1000, size=data.shape[0])]
    col_names = [str(x) for x in np.random.randint(0, 1000, size=data.shape[1])]

    enrichment = MultisetEnrichment(data, row_names, col_names, likelihood_type="prod")

    samples = enrichment.runMCMC(num_samples=1000)

    plotModelHists(samples, data, name="prodLH-")
    plotModelArrays(samples, data, name="prodLH-")

    results = enrichment.getSummary(samples)

    print(results)


def testNoConstraintSum():

    nx = 10
    ny = 15

    means_rows = np.ones((nx, 1)) * 10
    means_cols = np.ones((ny, 1)) * 10

    means_rows[1] += 2
    means_rows[3] += 1
    means_cols[2] -= 2
    means = means_rows + means_cols.T

    data = np.random.poisson(means).astype(int)

    data[7, 7] += 10
    data[3, 5] -= 20
    data[2, 4] -= 10

    data = np.clip(data, 0, np.inf)

    row_names = [str(x) for x in np.random.randint(0, 1000, size=data.shape[0])]
    col_names = [str(x) for x in np.random.randint(0, 1000, size=data.shape[1])]

    enrichment = MultisetEnrichment(data, row_names, col_names, likelihood_type="sum")

    samples = enrichment.runMCMC(num_samples=10000)

    plotModelHists(samples, data, name="sumLH-")
    plotModelArrays(samples, data, name="sumLH-")

    results = enrichment.getSummary(samples)

    print(results)


if __name__ == "__main__":

    runAll()
