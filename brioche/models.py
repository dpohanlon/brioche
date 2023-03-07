import jax

import numpyro
import numpyro.distributions as dist

import jax.numpy as jnp


def likelihoodSum(freq_rows, freq_cols, freq_dev):
    return (freq_rows + freq_cols.T) + freq_dev


def likelihoodProd(freq_rows, freq_cols, freq_dev):
    return (freq_rows @ freq_cols.T) * freq_dev


def multisetModel(
    data, priors, likelihood_type="sum", row_constraint=False, col_constraint=False
):

    likelihood = likelihoodSum if likelihood_type == "sum" else likelihoodProd

    if row_constraint and col_constraint:

        # Not implemented yet

        return

    if row_constraint:

        freq_cols = numpyro.sample(
            "freq_cols",
            dist.TruncatedNormal(priors["col_means"], priors["col_stds"], low=0.0),
        ).reshape(-1, 1)
        freq_rows = numpyro.sample(
            "freq_rows",
            dist.TruncatedNormal(
                priors["row_means"][:-1], priors["row_stds"][:-1], low=0.0
            ),
        ).reshape(-1, 1)

        freq_dev = numpyro.sample(
            "freq_dev",
            dist.Normal(priors["dev_mean"], priors["dev_std"]),
            sample_shape=(data.shape[0] - 1, data.shape[1]),
        )

        freq = numpyro.deterministic("freq", likelihood(freq_rows, freq_cols, freq_dev))
        numpyro.sample("obs", dist.Poisson(jax.nn.softplus(freq)), obs=data[:-1, :])

        col_sum = jnp.sum(data, 0)

        last_row = numpyro.deterministic("last_row", col_sum - jnp.sum(freq, 0))

        numpyro.sample(
            "obs_row", dist.Poisson(jax.nn.softplus(last_row)), obs=data[-1, :]
        )

    elif col_constraint:

        freq_cols = numpyro.sample(
            "freq_cols",
            dist.TruncatedNormal(
                priors["col_means"][:-1], priors["col_stds"][:-1], low=0.0
            ),
        ).reshape(-1, 1)
        freq_rows = numpyro.sample(
            "freq_rows",
            dist.TruncatedNormal(priors["row_means"], priors["row_stds"], low=0.0),
        ).reshape(-1, 1)

        freq_dev = numpyro.sample(
            "freq_dev",
            dist.Normal(priors["dev_mean"], priors["dev_std"]),
            sample_shape=(data.shape[0], data.shape[1] - 1),
        )

        freq = numpyro.deterministic("freq", likelihood(freq_rows, freq_cols, freq_dev))
        numpyro.sample("obs", dist.Poisson(jax.nn.softplus(freq)), obs=data[:, :-1])

        row_sum = jnp.sum(data, 1)

        last_col = numpyro.deterministic("last_row", row_sum - jnp.sum(freq, 1))

        numpyro.sample(
            "obs_col", dist.Poisson(jax.nn.softplus(last_col)), obs=data[:, -1]
        )

    else:

        freq_cols = numpyro.sample(
            "freq_cols",
            dist.TruncatedNormal(priors["col_means"], priors["col_stds"], low=0.0),
        ).reshape(-1, 1)
        freq_rows = numpyro.sample(
            "freq_rows",
            dist.TruncatedNormal(priors["row_means"], priors["row_stds"], low=0.0),
        ).reshape(-1, 1)

        freq_dev = numpyro.sample(
            "freq_dev", dist.Normal(priors["dev_mean"], priors["dev_std"])
        )

        freq = numpyro.deterministic("freq", likelihood(freq_rows, freq_cols, freq_dev))
        numpyro.sample("obs", dist.Poisson(jax.nn.softplus(freq)), obs=data)
