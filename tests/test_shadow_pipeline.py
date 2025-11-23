def test_preprocess_imports():
    from ptnt.preprocess.shadow import shadow_results_to_data_vec
    assert callable(shadow_results_to_data_vec)
