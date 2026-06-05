import pytest
from bvhar.model import VarOls, VharOls
from bvhar.datasets import load_vix
import numpy as np

def test_var():
    num_data = 30
    win_size = 30
    dim_data = 3
    var_lag = 2
    etf_vix = load_vix()
    data = etf_vix.to_numpy()[:num_data, :dim_data]
    n_ahead = 5
    data_out = etf_vix.to_numpy()[num_data:(num_data + n_ahead), :dim_data]

    fit_var = VarOls(data, var_lag, True, "nor")
    fit_var_llt = VarOls(data, var_lag, True, "chol")
    fit_var_qr = VarOls(data, var_lag, True, "qr")
    fit_var.fit()
    fit_var_llt.fit()
    fit_var_qr.fit()

    assert fit_var.n_features_in_ == dim_data
    assert fit_var.coef_.shape == (dim_data * var_lag + 1, dim_data)
    assert fit_var.intercept_.shape == (dim_data,)

    pred_out = fit_var.predict(n_ahead)
    roll_out = fit_var.roll_forecast(1, data_out, 1)
    expand_out = fit_var.expand_forecast(1, data_out, 1)

    assert pred_out['forecast'].shape == (n_ahead, dim_data)
    assert roll_out['forecast'].shape == (n_ahead, dim_data)
    assert expand_out['forecast'].shape == (n_ahead, dim_data)

    sp_out = fit_var.spillover(n_ahead)
    assert sp_out['connect'].shape == (dim_data, dim_data)
    assert sp_out['net_pairwise'].shape == (dim_data, dim_data)
    # assert sp_out['tot'].shape == (dim_data,)
    assert sp_out['to'].shape == (dim_data,)
    assert sp_out['from'].shape == (dim_data,)
    assert sp_out['net'].shape == (dim_data,)
    dynamic_out = fit_var.dynamic_spillover(win_size, n_ahead)
    assert dynamic_out['tot'].shape == (data.shape[0] - win_size + 1,)
    assert dynamic_out['to'].shape == (data.shape[0] - win_size + 1, dim_data)
    assert dynamic_out['from'].shape == (data.shape[0] - win_size + 1, dim_data)
    assert dynamic_out['net'].shape == (data.shape[0] - win_size + 1, dim_data)

    data = np.random.randn(var_lag - 1, dim_data)
    with pytest.raises(ValueError, match=f"'data' rows must be larger than 'lag' = {var_lag}"):
        fit_var = VarOls(data, var_lag, True, "nor")

def test_vhar():
    num_data = 50
    win_size = 30
    dim_data = 3
    week = 5
    month = 22
    etf_vix = load_vix()
    data = etf_vix.to_numpy()[:num_data, :dim_data]
    n_ahead = 5
    data_out = etf_vix.to_numpy()[num_data:(num_data + n_ahead), :dim_data]

    fit_vhar = VharOls(data, week, month, True, "nor")
    fit_vhar_llt = VharOls(data, week, month, True, "chol")
    fit_vhar_qr = VharOls(data, week, month, True, "qr")
    fit_vhar.fit()
    fit_vhar_llt.fit()
    fit_vhar_qr.fit()

    assert fit_vhar.n_features_in_ == dim_data
    assert fit_vhar.coef_.shape == (dim_data * 3 + 1, dim_data)
    assert fit_vhar.intercept_.shape == (dim_data,)

    pred_out = fit_vhar.predict(n_ahead)
    roll_out = fit_vhar.roll_forecast(1, data_out, 1)
    expand_out = fit_vhar.expand_forecast(1, data_out, 1)

    assert pred_out['forecast'].shape == (n_ahead, dim_data)
    assert roll_out['forecast'].shape == (n_ahead, dim_data)
    assert expand_out['forecast'].shape == (n_ahead, dim_data)

    sp_out = fit_vhar.spillover(n_ahead)
    assert sp_out['connect'].shape == (dim_data, dim_data)
    assert sp_out['net_pairwise'].shape == (dim_data, dim_data)
    # assert sp_out['tot'].shape == (dim_data,)
    assert sp_out['to'].shape == (dim_data,)
    assert sp_out['from'].shape == (dim_data,)
    assert sp_out['net'].shape == (dim_data,)
    dynamic_out = fit_vhar.dynamic_spillover(win_size, n_ahead)
    assert dynamic_out['tot'].shape == (data.shape[0] - win_size + 1,)
    assert dynamic_out['to'].shape == (data.shape[0] - win_size + 1, dim_data)
    assert dynamic_out['from'].shape == (data.shape[0] - win_size + 1, dim_data)
    assert dynamic_out['net'].shape == (data.shape[0] - win_size + 1, dim_data)

    data = np.random.randn(month - 1, dim_data)
    with pytest.raises(ValueError, match=f"'data' rows must be larger than 'month' = {month}"):
        fit_vhar = VharOls(data, week, month, True, "nor")