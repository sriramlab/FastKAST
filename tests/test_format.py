import numpy as np
import pytest
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics.pairwise import additive_chi2_kernel
# from sklearn.kernel_approximation import AdditiveChi2Sampler
import numpy as np

from FastKAST.Compute.est import getfullComponent, getfullComponentMulti, getfullComponentPerm, getRLComponent, getmleComponent  # Import your functions

# Define some sample data for testing
def generate_test_data(n_samples=100, n_features=5, n_traits=1):
    """
    Generates random test data for your functions.

    Args:
        n_samples: Number of samples.
        n_features: Number of features in X.
        n_traits: Number of traits in y.

    Returns:
        X: Covariates matrix (n_samples x n_features).
        Z: Genotype matrix (n_samples x n_features).
        y: Phenotype matrix (n_samples x n_traits).
    """
    X = np.random.randn(n_samples, n_features)
    Z = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, n_traits)
    return X, Z, y


# Test each estimation case
@pytest.mark.parametrize("X, Z, y", [generate_test_data() for _ in range(5)])
def test_getfullComponentPerm(X, Z, y):
    """
    Tests the getfullComponentPerm function.
    """
    results = getfullComponentPerm(X, Z, y, Perm=10)
    assert "pval" in results
    assert isinstance(results["pval"], list)
    assert all(isinstance(p_value[0], float) for p_value in results["pval"])


# @pytest.mark.parametrize("X, Z, y", [generate_test_data(n_traits=5) for _ in range(5)])
# def test_getfullComponentMulti(X, Z, y):
#     """
#     Tests the getfullComponentMulti function.
#     """
#     results = getfullComponentMulti(X, Z, y)
#     assert "pvals" in results
#     assert isinstance(results["pvals"][0], float)


# @pytest.mark.parametrize("X, Z, y", [generate_test_data() for _ in range(5)])
# def test_getRLComponent(X, Z, y):
#     """
#     Tests the getRLComponent function.
#     """
#     p_value, p_value_perm = getRLComponent(X, Z, y)
#     assert isinstance(p_value, float)
#     assert isinstance(p_value_perm, float)

# @pytest.mark.parametrize("X, Z, y", [generate_test_data() for _ in range(5)])
# def test_getmleComponent(X, Z, y):
#     """
#     Tests the getmleComponent function.
#     """
#     results = getmleComponent(X, Z, y)
#     if results:  # Check if results are not empty
#         assert len(results) == 2
#         assert isinstance(results[0], float)
#         assert isinstance(results[1], float)

# Run the tests
if __name__ == "__main__":
    pytest.main()