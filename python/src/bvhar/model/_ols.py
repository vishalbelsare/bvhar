from ..utils._misc import check_np, get_var_intercept
from .._src._ols import OlsVar, OlsVhar
from .._src._ols import OlsForecast, OlsVarRoll, OlsVarExpand, OlsVharRoll, OlsVharExpand
from .._src._ols import OlsSpillover, OlsDynamicSpillover

class _Vectorautoreg:
    """Base class for OLS"""
    def __init__(self, data, lag, p, fit_intercept = True, method = "nor"):
        if method not in ["nor", "chol", "qr"]:
            raise ValueError(f"Argument ('method') '{method}' is not valid: Choose between {['nor', 'chol', 'qr']}")
        if lag == p:
            lag_name = "lag"
        else:
            lag_name = "month"
        self.method = {
            "nor": 1,
            "chol": 2,
            "qr": 3
        }.get(method, None)
        self.y_ = check_np(data)
        self.n_features_in_ = self.y_.shape[1]
        # if self.y_.shape[0] <= lag:
        #     raise ValueError(f"'data' rows must be larger than '{lag_name}' = {lag}")
        # self.p = lag
        self.p_ = p # 3 in VHAR
        self.lag_ = lag # month in VHAR
        if self.y_.shape[0] <= self.lag_:
            raise ValueError(f"'data' rows must be larger than '{lag_name}' = {self.lag_}")
        self.fit_intercept = fit_intercept
        self._model = None
        self.is_fitted_ = False
        self.coef_ = None
        self.intercept_ = None
        self.cov_ = None

    def fit(self):
        """Fit the model
        Returns
        -------
        self : object
            An instance of the estimator.
        """
        fit = self._model.returnOlsRes()
        self.coef_ = fit.get("coefficients")
        self.intercept_ = get_var_intercept(self.coef_, self.p_, self.fit_intercept)
        self.cov_ = fit.get("covmat")
        self.design_ = fit.get("design")
        self.response_ = fit.get("y0")
        self.is_fitted_ = True

    def predict(self):
        pass

    def roll_forecast(self):
        pass

    def expand_forecast(self):
        pass

    def spillover(self):
        pass

    def dynamic_spillover(self):
        pass

class VarOls(_Vectorautoreg):
    """OLS for Vector autoregressive model

    Fits VAR model using OLS.

    Parameters
    ----------
    data : array-like
        Time series data of which columns indicate the variables
    lag : int
        VAR lag, by default 1
    fit_intercept : bool
        Include constant term in the model, by default True
    method : str
        Normal equation solving method
        - "nor": projection matrix (default)
        - "chol": LU decomposition
        - "qr": QR decomposition)
    
    Attributes
    ----------
    coef_ : ndarray
        VAR coefficient matrix.

    intercept_ : ndarray
        VAR model constant vector.
    
    cov_ : ndarray
        VAR covariance matrix.

    n_features_in_ : int
        Number of variables.
    """
    def __init__(self, data, lag = 1, fit_intercept = True, method = "nor"):
        super().__init__(data, lag, lag, fit_intercept, method)
        self._model = OlsVar(self.y_, self.p_, self.fit_intercept, self.method)
    
    def predict(self, n_ahead: int):
        if not self.is_fitted_:
            raise RuntimeError("The model has not been fitted yet.")
        forecaster = OlsForecast(self.lag_, n_ahead, self.y_, self.coef_, self.fit_intercept)
        y_distn = forecaster.returnForecast()
        return {
            "forecast": y_distn
        }

    def roll_forecast(self, n_ahead: int, test, n_thread = 1):
        test = check_np(test)
        forecaster = OlsVarRoll(self.y_, self.p_, self.fit_intercept, n_ahead, test, self.method, n_thread)
        y_distn = forecaster.returnForecast()
        return {
            "forecast": y_distn
        }

    def expand_forecast(self, n_ahead: int, test, n_thread = 1):
        test = check_np(test)
        forecaster = OlsVarExpand(self.y_, self.p_, self.fit_intercept, n_ahead, test, self.method, n_thread)
        y_distn = forecaster.returnForecast()
        return {
            "forecast": y_distn
        }

    def spillover(self, n_ahead = 10):
        spo = OlsSpillover(self.p_, n_ahead, self.coef_, self.cov_)
        out_spillover = spo.returnSpillover()
        return {
            "connect": out_spillover['connect'],
            "net_pairwise": out_spillover['net_pairwise'],
            "tot": out_spillover['tot'],
            "to": out_spillover['to'],
            "from": out_spillover['from'],
            "net": out_spillover['net']
        }

    def dynamic_spillover(self, window: int, n_ahead = 10, n_thread = 1):
        spo = OlsDynamicSpillover(self.y_, window, n_ahead, self.p_, self.fit_intercept, self.method, n_thread)
        out_spillover = spo.returnSpillover()
        return {
            "tot": out_spillover['tot'],
            "to": out_spillover['to'],
            "from": out_spillover['from'],
            "net": out_spillover['net']
        }

class VharOls(_Vectorautoreg):
    """OLS for Vector heterogeneous autoregressive model

    Fits VHAR model using OLS.

    Parameters
    ----------
    data : array-like
        Time series data of which columns indicate the variables
    week : int
        VHAR weekly order, by default 5
    month : int
        VHAR monthly order, by default 22
    fit_intercept : bool
        Include constant term in the model, by default True
    method : str
        Normal equation solving method
        - "nor": projection matrix (default)
        - "chol": LU decomposition
        - "qr": QR decomposition)
    
    Attributes
    ----------
    coef_ : ndarray
        VHAR coefficient matrix.

    intercept_ : ndarray
        VHAR model constant vector.
    
    cov_ : ndarray
        VHAR covariance matrix.

    n_features_in_ : int
        Number of variables.
    """
    def __init__(self, data, week = 5, month = 22, fit_intercept = True, method = "nor"):
        super().__init__(data, month, 3, fit_intercept, method)
        self.week_ = week
        self.month_ = self.lag_ # or self.lag_ = [week, month]
        self._model = OlsVhar(self.y_, week, self.lag_, self.fit_intercept, self.method)
    
    def predict(self, n_ahead: int):
        if not self.is_fitted_:
            raise RuntimeError("The model has not been fitted yet.")
        forecaster = OlsForecast(self.week_, self.month_, n_ahead, self.y_, self.coef_, self.fit_intercept)
        y_distn = forecaster.returnForecast()
        return {
            "forecast": y_distn
        }

    def roll_forecast(self, n_ahead: int, test, n_thread = 1):
        test = check_np(test)
        forecaster = OlsVharRoll(self.y_, self.week_, self.month_, self.fit_intercept, n_ahead, test, self.method, n_thread)
        y_distn = forecaster.returnForecast()
        return {
            "forecast": y_distn
        }

    def expand_forecast(self, n_ahead: int, test, n_thread = 1):
        test = check_np(test)
        forecaster = OlsVharExpand(self.y_, self.week_, self.month_, self.fit_intercept, n_ahead, test, self.method, n_thread)
        y_distn = forecaster.returnForecast()
        return {
            "forecast": y_distn
        }

    def spillover(self, n_ahead = 10):
        spo = OlsSpillover(self.week_, self.month_, n_ahead, self.coef_, self.cov_)
        out_spillover = spo.returnSpillover()
        return {
            "connect": out_spillover['connect'],
            "net_pairwise": out_spillover['net_pairwise'],
            "tot": out_spillover['tot'],
            "to": out_spillover['to'],
            "from": out_spillover['from'],
            "net": out_spillover['net']
        }

    def dynamic_spillover(self, window: int, n_ahead = 10, n_thread = 1):
        spo = OlsDynamicSpillover(self.y_, window, n_ahead, self.month_, self.fit_intercept, self.method, n_thread, self.week_)
        out_spillover = spo.returnSpillover()
        return {
            "tot": out_spillover['tot'],
            "to": out_spillover['to'],
            "from": out_spillover['from'],
            "net": out_spillover['net']
        }