import jax
import numpy as np
import pytest
from scoringrules._energy import energy_score, weenergy_score

from .conftest import BACKENDS

ENSEMBLE_SIZE = 51
N = 100
N_VARS = 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_weenergy_score(backend):
    rng = np.random.default_rng(53)
    obs = rng.normal(size=(N, N_VARS))
    fct = np.expand_dims(obs, axis=-2) + rng.normal(size=(N, ENSEMBLE_SIZE, N_VARS))
    fct_wt = np.abs(rng.normal(size=(N, ENSEMBLE_SIZE)))

    res = weenergy_score(obs, fct, fct_wt, backend=backend)

    if backend in ["numpy", "numba"]:
        assert isinstance(res, np.ndarray)
    elif backend == "jax":
        assert isinstance(res, jax.Array)


@pytest.mark.parametrize("backend", BACKENDS)
def test_we_vs_es_equal_weights(backend):
    rng = np.random.default_rng(75)
    obs = rng.normal(size=(N, N_VARS))
    fct = np.expand_dims(obs, axis=-2) + rng.normal(size=(N, ENSEMBLE_SIZE, N_VARS))
    fct_wt = np.full((N, ENSEMBLE_SIZE), 1 / ENSEMBLE_SIZE)

    res = energy_score(obs, fct, backend=backend)
    res_we = weenergy_score(obs, fct, fct_wt, backend=backend)

    np.testing.assert_allclose(res, res_we, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_we_vs_es_unequal_weights(backend):
    rng = np.random.default_rng(42)
    obs = rng.normal(size=(N, N_VARS))
    fct = np.expand_dims(obs, axis=-2) + rng.normal(size=(N, ENSEMBLE_SIZE, N_VARS))

    # Generate sets of random repeat counts that all sum to the same value (N_WITH_REPEATS)
    # so that they are stackable.
    N_WITH_REPEATS = ENSEMBLE_SIZE * 6
    p = rng.dirichlet(np.ones(ENSEMBLE_SIZE))
    fct_wt = rng.multinomial(N_WITH_REPEATS, p, size=N)

    fct_repeated = []
    for idx in range(N):
        scenarios = fct[idx]
        n_repeat = fct_wt[idx]
        fct_repeated.append(np.repeat(scenarios, n_repeat, -2))
    fct_repeated = np.stack(fct_repeated, axis=-3)
    assert fct_repeated.shape == (N, N_WITH_REPEATS, N_VARS)

    res = energy_score(obs, fct_repeated, backend=backend)
    res_we = weenergy_score(obs, fct, fct_wt, backend=backend)
    np.testing.assert_allclose(res, res_we, atol=1e-6)
