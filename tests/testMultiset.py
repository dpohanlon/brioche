import numpy as np

from brioche.multisetEnrichment import MultisetEnrichment

from brioche.plot import plotModelArrays, plotModelHists

def runAll():

    testNoConstraint()

def testNoConstraint():

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

    row_names = [str(x) for x in np.random.randint(0, 1000, size = data.shape[0])]
    col_names = [str(x) for x in np.random.randint(0, 1000, size = data.shape[1])]

    enrichment = MultisetEnrichment(data, row_names, col_names)

    samples = enrichment.runMCMC(num_samples = 1000)

    plotModelHists(samples, data)
    plotModelArrays(samples, data)

    results = enrichment.getSummary(samples)

    print(results)

if __name__ == '__main__':

    runAll()
