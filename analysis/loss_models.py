"""
Loss-separation models fitted on the same scale as the log-log models.

A curved log-log loss characteristic needs no new law in itself: the
classical separation of the loss into a hysteresis term proportional to f
and an eddy-current term proportional to f^2 already produces one, with
two parameters instead of three.  This module fits those competitors so
that the comparison can be made with the same information criterion.

Loss models, P in W at fixed Bmax and T:

  power       P = a f^b                        (2 parameters)
  two-term    P = k_h f + k_e f^2              (2 parameters)
  bertotti    P = k_h f + k_x f^1.5 + k_e f^2  (3 parameters)
  quadratic   ln P = a2 (ln f)^2 + a1 ln f+a0  (3 parameters)

Conductivity models.  Under the homogenised eddy-current relation
P = k_geom sigma f^2 Bmax^2 every loss-separation model maps onto

  sigma_eff proportional to P / f^2 ,

so the two-term model becomes c_h/f + c_e and the Bertotti model
c_h/f + c_x f^-0.5 + c_e.  Both decrease monotonically towards a
positive constant and can never reproduce a minimum followed by a rise,
which is what the identified conductivity of the widest-gap core does.

All models are fitted by minimising the sum of squared residuals of
ln y, the same objective as loglog_fit.py, so that AICc values are
directly comparable.  The physical coefficients are constrained to be
non-negative.
"""
import numpy as np
from scipy.optimize import least_squares, nnls

from loglog_fit import aicc_from_ssres

LOSS_MODELS = {
    "two-term": (lambda f: np.column_stack([f, f ** 2]),
                 r"$k_h f + k_e f^2$"),
    "bertotti": (lambda f: np.column_stack([f, f ** 1.5, f ** 2]),
                 r"$k_h f + k_x f^{1.5} + k_e f^2$"),
}

SIGMA_MODELS = {
    "two-term": (lambda f: np.column_stack([1.0 / f, np.ones_like(f)]),
                 r"$c_h/f + c_e$"),
    "bertotti": (lambda f: np.column_stack([1.0 / f, f ** -0.5,
                                            np.ones_like(f)]),
                 r"$c_h/f + c_x f^{-1/2} + c_e$"),
}


def fit_nonneg_log(f, y, basis):
    """Least squares in ln y for y = sum_j c_j g_j(f) with all c_j >= 0.

    The linear-space NNLS solution seeds a bounded nonlinear solve of the
    log-space problem; a second start from the scaled NNLS solution
    guards against the poor conditioning of the 50 Hz point.
    """
    f = np.asarray(f, dtype=float)
    y = np.asarray(y, dtype=float)
    G = basis(f)
    lny = np.log(y)

    seeds = []
    c0, _ = nnls(G, y)
    seeds.append(np.maximum(c0, 1e-30))
    w = 1.0 / y
    c1, _ = nnls(G * w[:, None], y * w)
    seeds.append(np.maximum(c1, 1e-30))

    def resid(logc):
        return np.log(basis(f) @ np.exp(logc)) - lny

    best = None
    for s in seeds:
        try:
            sol = least_squares(resid, np.log(s), method="lm", max_nfev=20000)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    coef = np.exp(best.x)
    r = resid(best.x)
    ssres = float(r @ r)
    n, k = len(f), G.shape[1]
    aic, aicc = aicc_from_ssres(ssres, n, k)
    resid_rel = 100.0 * (np.exp(-r) - 1.0)
    return {
        "coef": coef, "ssres": ssres, "aic": aic, "aicc": aicc,
        "n": n, "k": k,
        "r2": 1.0 - ssres / float(np.sum((lny - lny.mean()) ** 2)),
        "rms": float(np.sqrt(np.mean(resid_rel ** 2))),
        "max": float(np.max(np.abs(resid_rel))),
        "resid_rel": resid_rel,
        "predict": (lambda ff, c=coef, b=basis: b(np.asarray(ff, float)) @ c),
    }


def fit_family(f, y, family):
    """Fit every member of a model family; returns {name: result}."""
    return {name: fit_nonneg_log(f, y, basis)
            for name, (basis, _) in family.items()}
