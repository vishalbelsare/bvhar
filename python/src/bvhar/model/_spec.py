import numpy as np
from math import sqrt

class LdltConfig:
    """Prior for Covariance Matrix

    Specifies inverse-gamma prior for cholesky diagonal vector.

    Parameters
    ----------
    ig_shape : float
        Inverse-Gamma shape of Cholesky diagonal vector, by default 3
    ig_scale : float
        Inverse-Gamma scale of Cholesky diagonal vector, by default .01
    
    Attributes
    ----------
    shape : float
        Inverse-Gamma shape
    scale : float
        Inverse-Gamma scale
    """
    def __init__(self, ig_shape = 3, ig_scale = .01):
        self.process = "Homoskedastic"
        self.prior = "Cholesky"
        self.shape = self._validate(ig_shape, "ig_shape")
        self.scale = self._validate(ig_scale, "ig_scale")
    
    def _validate(self, value, member):
        if isinstance(value, int):
            return [float(value)]
        elif isinstance(value, (float, np.number)):
            return [value]
        elif isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                raise ValueError(f"'{member}' cannot be empty.")
            return np.array(value)
        else:
            raise TypeError(f"'{member}' should be a number or a numeric array.")
    
    def validate(self, value, member):
        self._validate(self, value, member)
    
    def update(self, n_dim: int):
        if len(self.shape) == 1:
            self.shape = np.repeat(self.shape, n_dim)
        if len(self.scale) == 1:
            self.scale = np.repeat(self.scale, n_dim)

    def to_dict(self):
        return {
            "shape": self.shape,
            "scale": self.scale
        }

class SvConfig(LdltConfig):
    def __init__(self, ig_shape = 3, ig_scale = .01, initial_mean = 1, initial_prec = .1):
        super().__init__(ig_shape, ig_scale)
        self.process = "SV"
        self.initial_mean = self.validate(initial_mean, "initial_mean", 1)
        self.initial_prec = self.validate(initial_prec, "initial_prec", 2)
    
    def validate(self, value, member, n_dim):
        if isinstance(value, (int, float, np.number)):
            return [value] if n_dim == 1 else value * np.identity(1)
        elif isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                raise ValueError(f"'{member} cannot be empty.")
            value_array = np.array(value)
            if value_array.ndim > 2:
                raise ValueError(f"'{member} has wrong dim = {n_dim}.")
            elif value_array.ndim != n_dim:
                raise ValueError(f"'{member}' should be {n_dim}-dim.")
            return value_array
        else:
            raise TypeError(f"'{member}' should be a number or a numeric array.")
    
    def update(self, n_dim: int):
        super().update(n_dim)
        if isinstance(self.initial_mean, (int, float, np.number)) or (hasattr(self.initial_mean, '__len__') and len(self.initial_mean) == 1):
            self.initial_mean = np.repeat(self.initial_mean, n_dim)
        if isinstance(self.initial_prec, (int, float, np.number)) or (hasattr(self.initial_prec, '__len__') and len(self.initial_prec) == 1):
            # self.initial_prec = self.initial_prec[0] * np.identity(n_dim)
            self.initial_prec = np.repeat(self.initial_prec[0], n_dim)

    def to_dict(self):
        return {
            "shape": self.shape,
            "scale": self.scale,
            "initial_mean": self.initial_mean,
            "initial_prec": self.initial_prec
        }

def get_cov_init(init_ : list, cov_spec_ : LdltConfig, n_features_in_ : int, n_design : int):
    if type(cov_spec_) == LdltConfig:
        for init in init_:
            init.update({
                'init_diag': np.exp(np.random.uniform(-1, 1, n_features_in_))
            })
    elif type(cov_spec_) == SvConfig:
        for init in init_:
            init.update({
                'lvol_init': np.random.uniform(-1, 1, n_features_in_),
                'lvol': np.exp(np.random.uniform(-1, 1, n_features_in_ * n_design)).reshape(n_features_in_, -1).T,
                'lvol_sig': [np.exp(np.random.uniform(-1, 1))]
            })
    return init_

class InterceptConfig:
    """Prior for Constant term

    Specifies normal prior for constant term.
    """
    def __init__(self, mean = 0, sd = .1):
        self.process = "Intercept"
        self.prior = "Normal"
        self.mean_non = self.validate(mean, "mean")
        self.sd_non = self.validate(sd, "sd")
    
    def validate(self, value, member):
        # if isinstance(value, int):
        #     return [float(value)]
        # elif isinstance(value, (float, np.number)):
        #     return [value]
        if isinstance(value, int):
            return float(value)
        if isinstance(value, (float, np.number)):
            return value
        elif isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                raise ValueError(f"'{member}' cannot be empty.")
            return np.array(value)
        else:
            raise TypeError(f"'{member}' should be a number or a numeric array.")
    
    def update(self, n_dim: int):
        if isinstance(self.mean_non, (int, float, np.number)) or (hasattr(self.mean_non, '__len__') and len(self.mean_non) == 1):
            self.mean_non = np.repeat(self.mean_non, n_dim)
        if not isinstance(self.sd_non, (int, float, np.number)) or (hasattr(self.sd_non, '__len__') and len(self.sd_non) > 1):
            raise ValueError("'sd_non' should be a number.")

    def to_dict(self):
        return {
            "mean_non": self.mean_non,
            "sd_non": self.sd_non
        }

class _BayesConfig:
    """Base class for coefficient prior configuration"""
    def __init__(self, prior):
        self.prior = prior
    
    def validate(self, value, member, n_size = None):
        # if isinstance(value, int):
        #     return [float(value)]
        # elif isinstance(value, (float, np.number)):
        #     return [value]
        if isinstance(value, (int, float, np.number)):
            return value
        elif isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                raise ValueError(f"'{member}' cannot be empty.")
            elif n_size is not None and len(value) != n_size:
                raise ValueError(f"'{member}' length must be {n_size}.")
            value_array = np.array(value)
            if value_array.ndim > 2:
                raise ValueError(f"'{member} has wrong dim = {value_array.ndim}.")
            return value_array
        else:
            raise TypeError(f"'{member}' should be a number or a numeric array.")

    def update(self):
        pass

    def to_dict(self):
        pass

class SsvsConfig(_BayesConfig):
    """SSVS prior configuration

    Specifies SSVS prior for coefficient.
    """
    def __init__(
        self,
        spike_grid = 100, slab_shape = .01, slab_scl = .01,
        s1 = [1, 1], s2 = [1, 1]
    ):
        super().__init__("SSVS")
        self.spike_grid = self.validate(spike_grid, "spike_grid")
        self.slab_shape = self.validate(slab_shape, "slab_shape")
        self.slab_scl = self.validate(slab_scl, "slab_scl")
        self.s1 = self.validate(s1, "s1", 2)
        self.s2 = self.validate(s2, "s1", 2)
        # self.chol_s1 = self.validate(chol_s1, "chol_s1")
        # self.chol_s2 = self.validate(chol_s2, "chol_s1")

    def update(self, grp_id: np.array, own_id: np.array, cross_id: np.array):
        if len(self.s1) == 2:
            s1 = np.zeros(len(grp_id))
            s1[np.isin(grp_id, own_id)] = self.s1[0]
            s1[np.isin(grp_id, cross_id)] = self.s1[1]
            self.s1 = s1
        if len(self.s2) == 2:
            s2 = np.zeros(len(grp_id))
            s2[np.isin(grp_id, own_id)] = self.s2[0]
            s2[np.isin(grp_id, cross_id)] = self.s2[1]
            self.s2 = s2

    def to_dict(self):
        return {
            "grid_size": self.spike_grid,
            "slab_shape": self.slab_shape,
            "slab_scl": self.slab_scl,
            "s1": self.s1,
            "s2": self.s2
        }

class HorseshoeConfig(_BayesConfig):
    """Horseshoe prior configuration

    Specifies Horseshoe prior for coefficient.
    """
    def __init__(self):
        super().__init__("Horseshoe")

    def to_dict(self):
        return dict()

class MinnesotaConfig(_BayesConfig):
    """Minnesota prior configuration

    Specifies Minnesota prior for coefficient.
    """
    def __init__(self, sig = None, lam = .1, delt = None, is_long = False, eps = 1e-04):
        super().__init__("Minnesota")
        self.sig = None
        if sig is not None:
            self.sig = self.validate(sig, "sig")
        # self.lam = self.validate(lam, "lam")
        self.lam = lam
        self.hierarchical = False
        if type(self.lam) == LambdaConfig:
            self.lam = [lam.shape_, lam.rate_, lam.grid_size] # shape and rate
            self.prior = "HMN"
            self.hierarchical = True
        self.delt = delt
        if delt is not None or isinstance(delt, np.ndarray):
            self.delt = self.validate(delt, "delt")
        self.eps = self.validate(eps, "eps")
        self.p = None
        self.num = None
        if is_long:
            self.weekly = None
            self.monthly = None
    
    def update(self, y: np.array, p, n_dim: int):
        self.p = p
        self.num = n_dim
        if self.sig is None:
            self.sig = np.apply_along_axis(np.std, 0, y)
        if self.delt is not None and len(self.delt) == 1:
            self.delt = np.repeat(self.delt, n_dim)
        elif self.delt is None:
            self.delt = np.repeat(0, n_dim)
        if hasattr(self, "weekly"):
            self.weekly = np.repeat(0, n_dim)
            self.monthly = np.repeat(0, n_dim)

    def to_dict(self):
        res = {
            "p": self.p,
            "num": self.num,
            "sigma": self.sig,
            "eps": self.eps,
            "hierarchical": self.hierarchical
        }
        if hasattr(self, "weekly"):
            delt_term = {
                "daily": self.delt,
                "weekly": self.weekly,
                "monthly": self.monthly
            }
            res.update(delt_term)
        else:
            delt_term = {"delta": self.delt}
            res.update(delt_term)
        if self.p > 0:
            del res["num"]
        else:
            del res["p"]
        if self.hierarchical:
            lam_term = {
                "shape": self.lam[0],
                "rate": self.lam[1],
                "grid_size": self.lam[2]
            }
            res.update(lam_term)
        else:
            lam_term = {"lambda": self.lam}
            res.update(lam_term)
        return res

class LambdaConfig:
    r"""Hierarchical structure of Minnesota prior

    Specifies prior for :math:`\lambda` in Minnesota prior.
    """
    def __init__(self, shape = .01, rate = .01, grid_size = 100, eps = 1e-04):
        self.shape_ = self.validate(shape, "shape")
        self.rate_ = self.validate(rate, "rate")
        self.grid_size = self.validate(grid_size, "grid_size")
    
    def validate(self, value, member):
        if isinstance(value, int):
            return float(value)
        if isinstance(value, (float, np.number)):
            return value
        elif isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                raise ValueError(f"'{member}' cannot be empty.")
            return np.array(value)
        else:
            raise TypeError(f"'{member}' should be a number or a numeric array.")
    
    def update(self, mode, sd):
        self.shape_ = (2 + mode ** 2 / (sd ** 2) + sqrt((2 + mode ** 2 / (sd ** 2)) ** 2 - 4)) / 2
        self.rate_ = sqrt(self.shape_) / sd

class DlConfig(_BayesConfig):
    """Dirichlet-Laplace prior configuration

    Specifies Dirichlet-Laplace prior for coefficient.
    """
    def __init__(self, dir_grid: int = 100, shape = .01, scale = .01):
        super().__init__("DL")
        self.grid_size = self.validate(dir_grid, "dir_grid")
        self.shape = self.validate(shape, "shape")
        self.scale = self.validate(scale, "scale")

    def to_dict(self):
        return {
            "grid_size": self.grid_size,
            "shape": self.shape,
            "scale": self.scale
        }

class NgConfig(_BayesConfig):
    """Normal-Gamma prior configuration

    Specifies Normal-Gamma prior for coefficient.
    """
    def __init__(
        self, shape_sd = .01, group_shape = .01, group_scale = .01,
        global_shape = .01, global_scale = .01
    ):
        super().__init__("NG")
        self.shape_sd = self.validate(shape_sd, "shape_sd")
        self.group_shape = self.validate(group_shape, "group_shape")
        self.group_scale = self.validate(group_scale, "group_scale")
        self.global_shape = self.validate(global_shape, "global_shape")
        self.global_scale = self.validate(global_scale, "global_scale")
        # self.contem_global_shape = self.validate(contem_global_shape, "contem_global_shape")
        # self.contem_global_scale = self.validate(contem_global_scale, "contem_global_scale")

    def to_dict(self):
        return {
            "shape_sd": self.shape_sd,
            "group_shape": self.group_shape,
            "group_scale": self.group_scale,
            "global_shape": self.global_shape,
            "global_scale": self.global_scale
        }
    
class GdpConfig(_BayesConfig):
    """Generalized Double Pareto prior configuration

    Specifies GDP prior for coefficient.
    """
    def __init__(self, shape_grid: int = 100, rate_grid: int = 100):
        super().__init__("GDP")
        self.grid_shape = self.validate(shape_grid, "shape_grid")
        self.grid_rate = self.validate(rate_grid, "rate_grid")

    def to_dict(self):
        return {
            "grid_shape": self.grid_shape,
            "grid_rate": self.grid_rate
        }

def validate_spec(bayes_spec_: _BayesConfig, y: np.array, p: int, n_features_in_: int, group_id = None, own_id = None, cross_id = None):
    if type(bayes_spec_) == SsvsConfig:
        if group_id is not None:
            bayes_spec_.update(group_id, own_id, cross_id)
    elif type(bayes_spec_) == MinnesotaConfig:
        bayes_spec_.update(y, p, n_features_in_)
    return bayes_spec_

def get_init(init_: list, bayes_spec_: _BayesConfig, n_alpha: int, n_grp: int):
    res = [dict(init) for init in init_]
    if type(bayes_spec_) == SsvsConfig:
        for init in res:
            init_mixture = np.random.uniform(-1, 1, n_grp)
            init_mixture = np.exp(init_mixture) / (1 + np.exp(init_mixture))
            init_dummy = np.random.binomial(1, 0.5, n_alpha)
            init_slab = np.exp(np.random.uniform(-1, 1, n_alpha))
            init.update({
                'dummy': init_dummy,
                'mixture': init_mixture,
                'slab': init_slab,
                'spike_scl': np.random.uniform(0, 1)
            })
    elif type(bayes_spec_) == HorseshoeConfig:
        for init in res:
            local_sparsity = np.exp(np.random.uniform(-1, 1, n_alpha))
            global_sparsity = np.exp(np.random.uniform(-1, 1))
            group_sparsity = np.exp(np.random.uniform(-1, 1, n_grp))
            init.update({
                'local_sparsity': local_sparsity,
                'global_sparsity': global_sparsity,
                'group_sparsity': group_sparsity
            })
    elif type(bayes_spec_) == MinnesotaConfig:
        for init in res:
            init.update({
                'own_lambda': np.random.uniform(0, 1),
                'cross_lambda': np.random.uniform(0, 1)
            })
    elif type(bayes_spec_) == DlConfig:
        for init in res:
            local_sparsity = np.exp(np.random.uniform(-1, 1, n_alpha))
            global_sparsity = np.exp(np.random.uniform(-1, 1))
            group_sparsity = np.exp(np.random.uniform(-1, 1, n_grp))
            init.update({
                'local_sparsity': local_sparsity,
                'global_sparsity': global_sparsity,
                'group_sparsity': group_sparsity
            })
    elif type(bayes_spec_) == NgConfig:
        for init in res:
            local_sparsity = np.exp(np.random.uniform(-1, 1, n_alpha))
            global_sparsity = np.exp(np.random.uniform(-1, 1))
            group_sparsity = np.exp(np.random.uniform(-1, 1, n_grp))
            local_shape = np.random.uniform(0, 1, n_grp)
            init.update({
                'local_shape': local_shape,
                'local_sparsity': local_sparsity,
                'global_sparsity': global_sparsity,
                'group_sparsity': group_sparsity
            })
    elif type(bayes_spec_) == GdpConfig:
        for init in res:
            local_sparsity = np.exp(np.random.uniform(-1, 1, n_alpha))
            group_rate = np.exp(np.random.uniform(-1, 1, n_grp))
            coef_shape = np.random.uniform(0, 1)
            coef_rate = np.random.uniform(0, 1)
            init.update({
                'local_sparsity': local_sparsity,
                'group_rate': group_rate,
                'gamma_shape': coef_shape,
                'gamma_rate': coef_rate
            })
    return res

def enumerate_prior_type(bayes_spec: _BayesConfig):
    return {
        "Minnesota": 1,
        "SSVS": 2,
        "Horseshoe": 3,
        "HMN": 4,
        "NG": 5,
        "DL": 6,
        "GDP": 7
    }.get(bayes_spec.prior)