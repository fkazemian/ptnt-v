def test_fit_imports():
    from ptnt.tn.fit import compute_likelihood
    assert callable(compute_likelihood)
