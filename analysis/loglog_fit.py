"""
Polynomial regression of ln(y) on ln(x) with small-sample statistics.

Models (natural logarithms throughout, x = f in Hz):

  classical  ln y = b ln f + ln a                  y = a f^b
  extended   ln y = a2 (ln f)^2 + a1 ln f + a0     n_eff(f) = a1 + 2 a2 ln f

Both are fitted by ordinary least squares on ln y, which is the maximum
likelihood estimator when the relative error of y is constant.  With
n = 7 frequency points the asymptotic AIC is not usable, so the
small-sample corrected AICc of Burnham and Anderson is reported instead;
the number of estimated parameters K counts the regression coefficients
plus the residual variance.

The classical fit is the same regression that
data/fem/T13_2026_02/Script_analyse_02_26_01.m runs through its
loglog_lin_ident(), an Octave function that is not part of the archive,
so the two cannot be compared directly here.  The quadratic model, AICc,
leverage and the coefficient covariance that the difference tests in
run_analysis.py need have no counterpart on the Octave side.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class FitResult:
    kind: str                 # 'lin' | 'quad'
    beta: np.ndarray          # polynomial coefficients, highest order first
    se: np.ndarray            # standard errors
    ci95: np.ndarray          # half-widths of the 95 % confidence intervals
    cov: np.ndarray           # coefficient covariance matrix
    r2: float
    r2_adj: float
    aic: float
    aicc: float
    bic: float
    ssres: float
    dof: int
    n: int
    leverage: np.ndarray      # diagonal of the hat matrix
    resid_rel: np.ndarray     # 100 (y / yhat - 1)
    p_values: np.ndarray = field(default=None)

    def predict(self, x):
        lx = np.log(np.asarray(x, dtype=float))
        return np.exp(np.polyval(self.beta, lx))

    def n_eff(self, x):
        """Local log-log slope d ln y / d ln f."""
        lx = np.log(np.asarray(x, dtype=float))
        return np.polyval(np.polyder(self.beta), lx)


def aicc_from_ssres(ssres, n, k_reg):
    """AICc for least squares; K = k_reg regression coefficients + variance."""
    K = k_reg + 1
    aic = n * np.log(ssres / n) + 2.0 * K
    if n - K - 1 <= 0:
        return aic, np.inf
    return aic, aic + 2.0 * K * (K + 1) / (n - K - 1)


def loglog_polyfit(x, y, degree):
    """OLS fit of ln y on a polynomial in ln x, with t-based 95 % CIs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = (x > 0) & (y > 0)
    X, Y = np.log(x[keep]), np.log(y[keep])
    n, k = len(X), degree + 1

    M = np.vander(X, k)
    beta, *_ = np.linalg.lstsq(M, Y, rcond=None)
    resid = Y - M @ beta
    ssres = float(resid @ resid)
    dof = n - k
    s2 = ssres / dof
    XtXinv = np.linalg.inv(M.T @ M)
    cov = s2 * XtXinv
    se = np.sqrt(np.diag(cov))
    ci95 = stats.t.ppf(0.975, dof) * se

    sstot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1.0 - ssres / sstot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / dof
    aic, aicc = aicc_from_ssres(ssres, n, k)
    bic = n * np.log(ssres / n) + (k + 1) * np.log(n)
    p_values = 2.0 * stats.t.sf(np.abs(beta) / se, dof)
    leverage = np.einsum("ij,jk,ik->i", M, XtXinv, M)
    resid_rel = 100.0 * (np.exp(resid) - 1.0)

    return FitResult("lin" if degree == 1 else "quad",
                     beta, se, ci95, cov, r2, r2_adj, aic, aicc, bic,
                     ssres, dof, n, leverage, resid_rel, p_values)


def fit_both(x, y):
    """Classical and extended fit of the same data set."""
    return loglog_polyfit(x, y, 1), loglog_polyfit(x, y, 2)


def summary_row(x, y):
    """Flat dictionary with everything the tables and figures need."""
    lin, quad = fit_both(x, y)
    return {
        "b_lin": lin.beta[0], "b_lin_ci": lin.ci95[0],
        "lna_lin": lin.beta[1], "lna_lin_ci": lin.ci95[1],
        "a_lin": float(np.exp(lin.beta[1])),
        "r2_lin": lin.r2,
        "rms_lin": float(np.sqrt(np.mean(lin.resid_rel ** 2))),
        "max_lin": float(np.max(np.abs(lin.resid_rel))),
        "a2": quad.beta[0], "a2_ci": quad.ci95[0],
        "a2_se": quad.se[0], "a2_p": quad.p_values[0],
        "a1": quad.beta[1], "a1_ci": quad.ci95[1],
        "a0": quad.beta[2], "a0_ci": quad.ci95[2],
        "r2_quad": quad.r2,
        "rms_quad": float(np.sqrt(np.mean(quad.resid_rel ** 2))),
        "max_quad": float(np.max(np.abs(quad.resid_rel))),
        "dAIC": lin.aic - quad.aic,
        "dAICc": lin.aicc - quad.aicc,
        "dBIC": lin.bic - quad.bic,
        "lev50": float(quad.leverage[0]),
        "lin": lin, "quad": quad,
    }


def a2_difference_test(row1, row2):
    """Two-sided t test of a2(1) - a2(2) for two independent fits.

    Welch-type pooling of the two a2 standard errors; the degrees of
    freedom are taken from the Welch-Satterthwaite formula.
    """
    d = row1["a2"] - row2["a2"]
    s1, s2 = row1["a2_se"], row2["a2_se"]
    v1, v2 = row1["quad"].dof, row2["quad"].dof
    se = np.hypot(s1, s2)
    dof = se ** 4 / (s1 ** 4 / v1 + s2 ** 4 / v2)
    t = d / se
    return d, se, float(t), float(dof), float(2.0 * stats.t.sf(abs(t), dof))
