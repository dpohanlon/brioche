<p align="center">
  <img width="602" height="200" src="https://github.com/dpohanlon/brioche/blob/main/assets/brioche.png">
  <br>
  Bayesian tests for set enrichment.
</p>

![Tests!](https://github.com/dpohanlon/brioche/actions/workflows/python-package.yml/badge.svg)

Installation
---
Install from PyPI
```bash
pip install brioche-enrichment
```

or install from the Github repository
```bash
git clone git@github.com:dpohanlon/brioche.git
pip install -r requirements.txt .
```

Usage
---
Prepare some data in a contingency table format, with row and column set annotations
```python
row_names = ["a", "b", "c", ...]
col_names = ["l", "m", "n", ...]

data = np.array([[30, 27, 10, ...], [28, 25, 11, ...], [31, 29, 15, ...], ...])
```

<p align="center">
  <img width="728" height="450" src="https://github.com/dpohanlon/brioche/blob/main/assets/data.png">
</p>

Import Brioche and create the multi-set enrichment object
```python
from brioche.multisetEnrichment import MultisetEnrichment

enrichment = MultisetEnrichment(data, row_names, col_names, likelihood_type="sum")
```
Optionally, specify whether there are total-sum constraints on the rows or columns
```python
enrichment = MultisetEnrichment(data, row_names, col_names, row_constraint = True,
								col_constraint = False, likelihood_type="sum")
```
Run the Markov-chain Monte Carlo inference on the enrichment model, retrieve samples from the posterior, and return a Pandas DataFrame of enrichment z scores
```python
samples = enrichment.runMCMC(num_samples=10000)

results = enrichment.getSummary(samples)
```

Plot model parameters and enrichment posterior probability distributions
```python
from brioche.plot import plotDeviations

plotDeviations(samples, threshold=2, x_labels=col_names, y_labels=row_names, name="test-")
```

<p align="center">
  <img width="800" height="450" src="https://github.com/dpohanlon/brioche/blob/main/assets/deviations.png">
</p>
