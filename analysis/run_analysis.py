"""
Complete statistical analysis of the measured losses and the identified
effective conductivity.

Reads the data from core_data.py, fits every model, runs the consistency
and robustness checks, and writes

  results/fit_coefficients.csv     every coefficient with its standard
                                   error, confidence interval, leverage
                                   and residual statistics
  results/model_comparison.csv     pooled residuals and AICc differences
                                   of the four competing descriptions
  results/curvature_excess.csv     a2 of P, a2 of sigma_eff, and their
                                   difference, per configuration and
                                   temperature
  results/sigma_minimum.csv        location and depth of the minimum of
                                   sigma_eff(f)
  results/fit_results.json         machine-readable dump of the fits
  results/analysis_report.txt      the console output of this script

Run from anywhere:  python3 analysis/run_analysis.py
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core_data import CORES, F, GEOM, T                       # noqa: E402
from loglog_fit import a2_difference_test, summary_row        # noqa: E402
from loss_models import LOSS_MODELS, SIGMA_MODELS, fit_family  # noqa: E402
from paths import CASE, CASE_SUPERSEDED, FEM, RESULTS         # noqa: E402

OUT = RESULTS
OUT.mkdir(exist_ok=True)
ORDER = ["full", "gap017", "gap047"]
QTY = [("P", "P_loss"), ("S", "sigma_eff")]
CORE_LABEL = {"full": "no gap", "gap017": "2 x 0.17 mm",
              "gap047": "2 x 0.47 mm"}

# uncertainty budget of sigma_eff, JCGM 100:2008 (GUM); reported in full
# further down and quoted where the non-monotonicity is assessed
U_MEAS = 3.5                        # loss reading, 1-2 % plus 2 x 1 % amplitude
U_OUTER = 2.0 / np.sqrt(3.0)        # outer-loop threshold, rectangular
U_INNER = 4.0 / np.sqrt(3.0)        # inner-loop threshold through P ~ B^2
U_MESH = 0.4 / np.sqrt(3.0)         # mesh convergence at 20 kHz
U_C = float(np.hypot(np.hypot(U_MEAS, U_OUTER), np.hypot(U_INNER, U_MESH)))

report = io.StringIO()


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    report.write(line + "\n")


# ---------------------------------------------------------------- fits
rows = {}
for qty, _ in QTY:
    for core in ORDER:
        M = CORES[core][qty]
        for j, temp in enumerate(T):
            r = summary_row(F, M[:, j])
            r["no50"] = summary_row(F[1:], M[1:, j])
            r["sub10k"] = summary_row(F[F <= 10000], M[F <= 10000, j])
            rows[(qty, core, j)] = r

say("=" * 78)
say("EXTENDED STEINMETZ ANALYSIS -- full numerical report")
say("=" * 78)
say(f"grid: {len(F)} frequencies {F[0]:.0f}-{F[-1]:.0f} Hz, "
    f"{len(T)} temperatures {T[0]:.0f}-{T[-1]:.0f} C, "
    f"{len(ORDER)} configurations = {len(F)*len(T)*len(ORDER)} operating points")
say(f"core: OD {GEOM['OD_mm']} mm, ID {GEOM['ID_mm']} mm, h {GEOM['H_mm']} mm, "
    f"A_geom {GEOM['area_geom_cm2']} cm2, A_eff {GEOM['area_eff_cm2']} cm2, "
    f"{GEOM['n_layers']} ribbon layers")

# ------------------------------------------------- per-fit parameter tables
for qty, _ in QTY:
    say("\n" + "=" * 78)
    say(f"{qty}: classical and extended fits")
    say("-" * 78)
    say(f"{'core':8s} {'T':>4s} | {'b':>7s} {'R2lin':>7s} | {'a2':>8s} "
        f"{'+-CI':>7s} {'a1':>8s} {'a0':>9s} {'p(a2)':>8s} {'R2quad':>7s} "
        f"{'dAIC':>7s} {'dAICc':>7s} {'lev50':>6s}")
    for core in ORDER:
        for j, temp in enumerate(T):
            r = rows[(qty, core, j)]
            say(f"{core:8s} {int(temp):4d} | {r['b_lin']:7.3f} {r['r2_lin']:7.4f} | "
                f"{r['a2']:8.4f} {r['a2_ci']:7.4f} {r['a1']:8.3f} {r['a0']:9.3f} "
                f"{r['a2_p']:8.4f} {r['r2_quad']:7.4f} {r['dAIC']:7.1f} "
                f"{r['dAICc']:7.1f} {r['lev50']:6.3f}")

# --------------------------------------------------- model selection counts
say("\n" + "=" * 78)
say("Model selection: how the small-sample correction changes the verdict")
say("-" * 78)
say("K counts the regression coefficients plus the residual variance,")
say("so K = 3 (classical) and K = 4 (extended); at n = 7 the AICc penalty")
say(f"difference is fixed at {2*4*5/(7-4-1) - 2*3*4/(7-3-1):.1f}.")
for qty, _ in QTY:
    for core in ORDER:
        rs = [rows[(qty, core, j)] for j in range(len(T))]
        n_p = sum(r["a2_p"] < 0.05 for r in rs)
        n_aic = sum(r["dAIC"] > 0 for r in rs)
        n_aicc = sum(r["dAICc"] > 0 for r in rs)
        say(f"  {qty} {core:8s}: p(a2)<0.05 in {n_p}/6, dAIC>0 in {n_aic}/6, "
            f"dAICc>0 in {n_aicc}/6  "
            f"[dAICc {min(r['dAICc'] for r in rs):6.1f} .. "
            f"{max(r['dAICc'] for r in rs):6.1f}]")

# multiplicity
pvals = np.array([rows[(q, c, j)]["a2_p"]
                  for q, _ in QTY for c in ORDER for j in range(len(T))])
order = np.argsort(pvals)
holm = np.empty_like(pvals)
m = len(pvals)
running = 0.0
for i, idx in enumerate(order):
    running = max(running, (m - i) * pvals[idx])
    holm[idx] = min(1.0, running)
say(f"\n  36 simultaneous t-tests of a2 = 0: {int((pvals < 0.05).sum())} "
    f"significant at 5 %, {int((holm < 0.05).sum())} after Holm correction")

# ------------------------------------------- practical size of the curvature
say("\n" + "=" * 78)
say("Practical size of the curvature: change of n_eff across the band")
say("-" * 78)
span = np.log(F[-1] / F[0])
for qty, _ in QTY:
    for core in ORDER:
        a2 = np.array([rows[(qty, core, j)]["a2"] for j in range(len(T))])
        dn = 2.0 * a2.mean() * span
        say(f"  {qty} {core:8s}: mean a2 = {a2.mean():7.4f}  "
            f"-> delta n_eff over 50 Hz..20 kHz = {dn:6.3f}")

# ------------------------------------------------ consistency relation, da2
say("\n" + "=" * 78)
say("Consistency with the unshielded eddy-current limit")
say("-" * 78)
say("  P = k sigma f^2 B^2 implies b_S = b_P - 2, a1(S) = a1(P) - 2 and")
say("  a2(S) = a2(P) exactly, so da2 = a2(S) - a2(P) measures the part of")
say("  the curvature that the mapping does not produce by itself.")
da2 = {}
for core in ORDER:
    bP = np.mean([rows[("P", core, j)]["b_lin"] for j in range(len(T))])
    bS = np.mean([rows[("S", core, j)]["b_lin"] for j in range(len(T))])
    d = np.array([rows[("S", core, j)]["a2"] - rows[("P", core, j)]["a2"]
                  for j in range(len(T))])
    da2[core] = d
    say(f"  {core:8s}: <b_P> = {bP:6.3f}, <b_S> = {bS:6.3f}, "
        f"<b_P>-2 = {bP-2:6.3f}, deviation = {bS-(bP-2):6.3f}")
    say(f"            da2 per temperature: "
        + " ".join(f"{v:6.3f}" for v in d)
        + f"   mean {d.mean():6.3f}  spread {d.max()-d.min():6.3f}")

# ------------------------------------------------ competing physical models
say("\n" + "=" * 78)
say("Competing models fitted on the same objective (least squares in ln y)")
say("-" * 78)
comp = {}
for qty, _ in QTY:
    fam = LOSS_MODELS if qty == "P" else SIGMA_MODELS
    for core in ORDER:
        M = CORES[core][qty]
        for j in range(len(T)):
            comp[(qty, core, j)] = fit_family(F, M[:, j], fam)
for qty, _ in QTY:
    say(f"\n  {qty}: AICc of competitor minus AICc of the extended model")
    say("      (positive: the extended model is preferred)")
    for core in ORDER:
        for name in ("two-term", "bertotti"):
            d = np.array([comp[(qty, core, j)][name]["aicc"]
                          - rows[(qty, core, j)]["quad"].aicc
                          for j in range(len(T))])
            wins = int((d > 0).sum())
            say(f"    {core:8s} vs {name:9s}: {d.min():7.1f} .. {d.max():7.1f} "
                f"(median {np.median(d):6.1f}), extended preferred in {wins}/6")
        d = np.array([rows[(qty, core, j)]["dAICc"] for j in range(len(T))])
        say(f"    {core:8s} vs {'power law':9s}: {d.min():7.1f} .. {d.max():7.1f} "
            f"(median {np.median(d):6.1f}), extended preferred in "
            f"{int((d > 0).sum())}/6")
    say(f"  {qty}: relative residuals of each model, pooled over temperature")
    for core in ORDER:
        rms_c = np.sqrt(np.mean(np.concatenate(
            [rows[(qty, core, j)]["lin"].resid_rel for j in range(len(T))]) ** 2))
        rms_q = np.sqrt(np.mean(np.concatenate(
            [rows[(qty, core, j)]["quad"].resid_rel for j in range(len(T))]) ** 2))
        out = [f"classical {rms_c:6.1f}"]
        for name in ("two-term", "bertotti"):
            r = np.sqrt(np.mean(np.concatenate(
                [comp[(qty, core, j)][name]["resid_rel"]
                 for j in range(len(T))]) ** 2))
            out.append(f"{name} {r:6.1f}")
        out.append(f"extended {rms_q:6.1f}")
        say(f"    {core:8s}: " + ", ".join(out) + "  (rms %)")

# -------------------------------------- non-monotonicity from the raw data
say("\n" + "=" * 78)
say("Non-monotonicity of sigma_eff read from the identified values, not the fit")
say("-" * 78)
nonmono = {}
minima = {}
for core in ORDER:
    S = CORES[core]["S"]
    say(f"  {core}:")
    rises, rec = [], []
    for j, temp in enumerate(T):
        col = S[:, j]
        i = int(np.argmin(col))
        rise = 100.0 * (col[-1] / col[i] - 1.0)
        rises.append(rise)
        rec.append((float(temp), float(F[i]), float(col[i]), float(col[-1]),
                    float(rise)))
        say(f"    T={int(temp):3d} C: min {col[i]:8.1f} S/m at "
            f"{F[i]:7.0f} Hz, value at 20 kHz {col[-1]:8.1f} S/m, "
            f"rise {rise:+6.1f} %")
    nonmono[core] = np.array(rises)
    minima[core] = rec
say(f"  the combined standard uncertainty of sigma_eff is {U_C:.1f} % (k=1), "
    f"{2*U_C:.1f} % (k=2); the budget is below")

# ------------------------------------------------------- robustness at 50 Hz
say("\n" + "=" * 78)
say("Leverage of the 50 Hz point and the fit without it")
say("-" * 78)
say(f"  ln f = " + " ".join(f"{np.log(f):.2f}" for f in F))
for qty, _ in QTY:
    for core in ORDER:
        lev = [rows[(qty, core, j)]["lev50"] for j in range(len(T))]
        d = [rows[(qty, core, j)]["no50"]["a2"] - rows[(qty, core, j)]["a2"]
             for j in range(len(T))]
        pn = [rows[(qty, core, j)]["no50"]["a2_p"] for j in range(len(T))]
        say(f"  {qty} {core:8s}: leverage {min(lev):.3f}-{max(lev):.3f}, "
            f"a2 shift {min(d):+.3f}..{max(d):+.3f}, "
            f"max p(a2) without 50 Hz {max(pn):.3f}")

# ------------------------------------------------------------ sub-band fit
say("\n" + "=" * 78)
say("Classical fit restricted to 50 Hz - 10 kHz against the full band")
say("-" * 78)
for qty, _ in QTY:
    for core in ORDER:
        full = [rows[(qty, core, j)]["r2_lin"] for j in range(len(T))]
        sub = [rows[(qty, core, j)]["sub10k"]["r2_lin"] for j in range(len(T))]
        say(f"  {qty} {core:8s}: R2 full band {min(full):.4f}-{max(full):.4f}, "
            f"<=10 kHz {min(sub):.4f}-{max(sub):.4f}")

# ------------------------------------------- differences between the cores
say("\n" + "=" * 78)
say("Difference of a2 between configurations (two-sided t test)")
say("-" * 78)
for qty, _ in QTY:
    for c1, c2 in (("full", "gap017"), ("gap017", "gap047"), ("full", "gap047")):
        ps = []
        for j in range(len(T)):
            _, _, _, _, p = a2_difference_test(rows[(qty, c2, j)],
                                               rows[(qty, c1, j)])
            ps.append(p)
        say(f"  {qty} {c2} vs {c1}: p = " + " ".join(f"{p:.4f}" for p in ps))

# -------------------------------------------------- temperature trend of a2
say("\n" + "=" * 78)
say("Trend of a2 with temperature (OLS of a2 on T, six points)")
say("-" * 78)
for qty, _ in QTY:
    for core in ORDER:
        a2 = np.array([rows[(qty, core, j)]["a2"] for j in range(len(T))])
        sl, ic, r, p, se = stats.linregress(T, a2)
        say(f"  {qty} {core:8s}: slope {sl*100:+.4f} per 100 K, p = {p:.3f}")

# ---------------------------------------------- temperature coefficient of P
say("\n" + "=" * 78)
say("Change of the loss between 20 and 120 C")
say("-" * 78)
for core in ORDER:
    M = CORES[core]["P"]
    ch = 100.0 * (M[:, -1] / M[:, 0] - 1.0)
    say(f"  {core:8s} (Bmax = {CORES[core]['B']} T): per frequency " +
        " ".join(f"{v:+6.1f}" for v in ch) +
        f"  | at 1 kHz {ch[2]:+.1f} %  ({M[2,0]*1e3:.3g} -> {M[2,-1]*1e3:.3g} mW)")

# ------------------------------------------------------------------ n_eff
say("\n" + "=" * 78)
say("Local exponent at the band edges")
say("-" * 78)
for qty, _ in QTY:
    for core in ORDER:
        lo = [rows[(qty, core, j)]["quad"].n_eff(F[0]) for j in range(len(T))]
        hi = [rows[(qty, core, j)]["quad"].n_eff(F[-1]) for j in range(len(T))]
        say(f"  {qty} {core:8s}: n_eff(50 Hz) {min(lo):6.2f}..{max(lo):6.2f}, "
            f"n_eff(20 kHz) {min(hi):6.2f}..{max(hi):6.2f}")
for core in ORDER:
    zc = np.array([np.exp(-rows[("S", core, j)]["a1"]
                          / (2 * rows[("S", core, j)]["a2"]))
                   for j in range(len(T))])
    inside = (zc >= F[0]) & (zc <= F[-1])
    if inside.all():
        say(f"  sigma_eff stationary point of {core:8s}: "
            f"{zc.min():.0f} .. {zc.max():.0f} Hz")
    else:
        say(f"  sigma_eff stationary point of {core:8s}: outside the measured "
            f"band at {int((~inside).sum())}/6 temperatures "
            f"(extrapolated {zc.min():.3g} .. {zc.max():.3g} Hz, "
            f"not a physical prediction)")

# ---------------------------- the minimum against the superseded geometry
say("\n" + "=" * 78)
say("The 0.47 mm minimum against the superseded 0.32 mm spacer geometry")
say("-" * 78)
old_path = CASE_SUPERSEDED / "res_step_ST.txt"
if old_path.exists():
    old = np.loadtxt(old_path)
    new = CORES["gap047"]["S"]
    ratio = new / old
    say(f"  corrected / superseded conductivity, mean over temperature: "
        + " ".join(f"{v:.3f}" for v in ratio.mean(axis=1)))
    say(f"  (rows are {', '.join(f'{f:.0f} Hz' for f in F)})")
    for label, M in (("superseded 0.32 mm", old), ("corrected 0.47 mm", new)):
        fmin = [F[int(np.argmin(M[:, j]))] for j in range(len(T))]
        rise = [100.0 * (M[-1, j] / M[:, j].min() - 1.0) for j in range(len(T))]
        r2 = [summary_row(F, M[:, j])["r2_lin"] for j in range(len(T))]
        say(f"  {label}: minimum at {min(fmin):.0f}-{max(fmin):.0f} Hz, "
            f"rise to 20 kHz {min(rise):+.1f}..{max(rise):+.1f} %, "
            f"classical R2 {min(r2):.3f}-{max(r2):.3f}")
else:
    say("  superseded results not present, check skipped")


# --------------------------------------------------- radial flux profiles
say("\n" + "=" * 78)
say("Radial profile of |B| in the core at 20 kHz and 120 C, from the")
say("SaveLine output of the final FEM runs (0.1 mm node spacing)")
say("-" * 78)
R_IN, R_OUT = 15.5, 24.45
profiles = {}
for core in ORDER:
    path = CASE[core] / "line000.dat"
    if not path.exists():
        say(f"  {core}: line000.dat missing, skipped")
        continue
    d = np.loadtxt(path)
    r = d[:, 3] * 1000.0
    b = np.sqrt((d[:, 15:18] ** 2).sum(1) + (d[:, 18:21] ** 2).sum(1))
    sel = (r >= R_IN) & (r <= R_OUT)
    r, b = r[sel], b[sel] / CORES[core]["B"]
    profiles[core] = (r, b)
    i = int(np.argmin(b))
    say(f"  {core:8s}: |B|/Bmax at the inner edge {b[0]:.2f}, at the outer "
        f"edge {b[-1]:.2f}, minimum {b[i]:.2f} at r = {r[i]:.1f} mm, "
        f"mid-width value {np.interp(20.0, r, b):.2f}")
if "full" in profiles:
    r, b = profiles["full"]
    say(f"  the ungapped profile against a 1/r reference: "
        f"b(r_in)/b(r_out) = {b[0]/b[-1]:.3f}, r_out/r_in = {r[-1]/r[0]:.3f}")
say("  the identification controls the flux integral over a window enclosing")
say(f"  the cross-section divided by A0 = {GEOM['area_ref_cm2']:.2f} cm2; with")
say(f"  A_geom = {GEOM['area_geom_cm2']:.2f} cm2 the cross-section average of")
say("  |B| is 1.10 Bmax, verified on the stored 3-D field of the 0.47 mm case")

# --------------------------------------------------- uncertainty budget
say("\n" + "=" * 78)
say("Uncertainty budget of sigma_eff, JCGM 100:2008 (GUM)")
say("-" * 78)
say(f"  measurement of P                 {U_MEAS:5.2f} %")
say(f"  outer loop, 2 % bound / sqrt(3)  {U_OUTER:5.2f} %")
say(f"  inner loop, 4 % bound / sqrt(3)  {U_INNER:5.2f} %")
say(f"  discretisation, 0.4 % / sqrt(3)  {U_MESH:5.2f} %")
say(f"  combined standard uncertainty    {U_C:5.2f} % (k = 1)")
say(f"  expanded uncertainty             {2*U_C:5.2f} % (k = 2)")
say(f"  upper bound under linear summation "
    f"{U_MEAS + 2.0 + 4.0 + 0.4:5.1f} %")

# ------------------------------------------------- penetration depth check
say("\n" + "=" * 78)
say("Penetration depth at 20 kHz and the observed surface layer")
say("-" * 78)
hb = np.loadtxt(CASE["full"] / "HB")
mu0 = 4e-7 * np.pi
for core in ORDER:
    bmax = CORES[core]["B"]
    h_at_b = float(np.interp(bmax, hb[:, 0], hb[:, 1]))
    mu = bmax / h_at_b
    sig = CORES[core]["S"][-1, -1]
    delta = np.sqrt(2.0 / (2 * np.pi * F[-1] * mu * sig))
    line = (f"  {core:8s}: mu_r = {mu/mu0:7.0f} at Bmax, sigma_eff(20 kHz, "
            f"120 C) = {sig:6.1f} S/m, delta = {delta*1e3:.2f} mm")
    if core in profiles:
        r, b = profiles[core]
        thr = b[0] / np.e
        below = np.where(b <= thr)[0]
        if len(below):
            line += f", |B| falls to 1/e of the edge value by {r[below[0]]-r[0]:.1f} mm"
    say(line)
say(f"  largest element edge in the core: 0.50 mm (maxh of pierscien01.geo)")

# ------------------------------------------------------------ result files
def write_csv(name, header, rows_out):
    (OUT / name).write_text("\n".join([header] + rows_out) + "\n",
                            encoding="utf-8")
    say(f"  wrote {name}")


say("\n" + "=" * 78)
say("Output files")
say("-" * 78)

# pooled residuals and AICc differences of the four descriptions
lines = []
for qty, sym in QTY:
    for core in ORDER:
        pooled = {
            "power law": np.concatenate(
                [rows[(qty, core, j)]["lin"].resid_rel for j in range(len(T))]),
            "extended": np.concatenate(
                [rows[(qty, core, j)]["quad"].resid_rel for j in range(len(T))]),
        }
        for name in ("two-term", "bertotti"):
            pooled[name] = np.concatenate(
                [comp[(qty, core, j)][name]["resid_rel"] for j in range(len(T))])
        dq = {
            "power law": np.array([rows[(qty, core, j)]["dAICc"]
                                   for j in range(len(T))]),
            "extended": np.zeros(len(T)),
        }
        for name in ("two-term", "bertotti"):
            dq[name] = np.array([comp[(qty, core, j)][name]["aicc"]
                                 - rows[(qty, core, j)]["quad"].aicc
                                 for j in range(len(T))])
        for name in ("power law", "two-term", "bertotti", "extended"):
            r, d = pooled[name], dq[name]
            lines.append(",".join([
                sym, core, f"{CORES[core]['gap_mm']}", name,
                f"{np.sqrt(np.mean(r ** 2)):.2f}", f"{np.max(np.abs(r)):.2f}",
                f"{np.median(d):.2f}", f"{d.min():.2f}", f"{d.max():.2f}",
                f"{int((d > 0).sum())}"]))
write_csv("model_comparison.csv",
          "quantity,core,gap_mm,model,rms_resid_pct,max_resid_pct,"
          "daicc_vs_extended_median,daicc_vs_extended_min,"
          "daicc_vs_extended_max,extended_preferred_of_6", lines)

# curvature of both quantities and the excess
lines = []
for core in ORDER:
    for j, temp in enumerate(T):
        a2P = rows[("P", core, j)]["a2"]
        a2S = rows[("S", core, j)]["a2"]
        lines.append(",".join([
            core, f"{CORES[core]['gap_mm']}", f"{temp:.0f}",
            f"{a2P:.6g}", f"{rows[('P', core, j)]['a2_ci']:.6g}",
            f"{a2S:.6g}", f"{rows[('S', core, j)]['a2_ci']:.6g}",
            f"{a2S - a2P:.6g}"]))
write_csv("curvature_excess.csv",
          "core,gap_mm,T_C,a2_P,a2_P_ci95,a2_sigma,a2_sigma_ci95,"
          "curvature_excess", lines)

# where sigma_eff turns and by how much it rises again
lines = []
for core in ORDER:
    for temp, fmin, smin, s20k, rise in minima[core]:
        lines.append(",".join([
            core, f"{CORES[core]['gap_mm']}", f"{temp:.0f}",
            f"{fmin:.0f}", f"{smin:.6g}", f"{s20k:.6g}", f"{rise:.2f}"]))
write_csv("sigma_minimum.csv",
          "core,gap_mm,T_C,f_min_Hz,sigma_min_S_per_m,"
          "sigma_20kHz_S_per_m,rise_to_20kHz_pct", lines)

csv = ["quantity,core,gap_mm,T_C,Bmax_T,b_lin,b_lin_se,ln_a_lin,ln_a_lin_se,"
       "r2_lin,a2,a2_se,a2_ci95,a1,a1_se,a1_ci95,a0,a0_se,a0_ci95,p_a2,"
       "r2_quad,dAIC,dAICc,leverage_50Hz,rms_lin_pct,max_lin_pct,"
       "rms_quad_pct,max_quad_pct"]
for qty, _ in QTY:
    for core in ORDER:
        for j, temp in enumerate(T):
            r = rows[(qty, core, j)]
            q, l = r["quad"], r["lin"]
            csv.append(",".join([
                qty, core, f"{CORES[core]['gap_mm']}", f"{temp:.0f}",
                f"{CORES[core]['B']}",
                f"{l.beta[0]:.6g}", f"{l.se[0]:.6g}",
                f"{l.beta[1]:.6g}", f"{l.se[1]:.6g}", f"{l.r2:.6f}",
                f"{q.beta[0]:.6g}", f"{q.se[0]:.6g}", f"{r['a2_ci']:.6g}",
                f"{q.beta[1]:.6g}", f"{q.se[1]:.6g}", f"{r['a1_ci']:.6g}",
                f"{q.beta[2]:.6g}", f"{q.se[2]:.6g}", f"{r['a0_ci']:.6g}",
                f"{r['a2_p']:.4g}", f"{q.r2:.6f}",
                f"{r['dAIC']:.3f}", f"{r['dAICc']:.3f}", f"{r['lev50']:.4f}",
                f"{r['rms_lin']:.3f}", f"{r['max_lin']:.3f}",
                f"{r['rms_quad']:.3f}", f"{r['max_quad']:.3f}"]))
(OUT / "fit_coefficients.csv").write_text("\n".join(csv) + "\n",
                                          encoding="utf-8")
say("  wrote fit_coefficients.csv")

dump = {"frequencies_Hz": F.tolist(), "temperatures_C": T.tolist(),
        "sigma_uncertainty_pct": {"k1": U_C, "k2": 2 * U_C},
        "geometry": {k: v for k, v in GEOM.items()},
        "units": {"P": "W", "sigma_eff": "S/m", "f": "Hz",
                  "model": "ln y = a2 (ln f)^2 + a1 ln f + a0"},
        "fits": {}, "curvature_excess": {k: v.tolist() for k, v in da2.items()},
        "sigma_rise_from_minimum_pct": {k: v.tolist()
                                        for k, v in nonmono.items()}}
for qty, _ in QTY:
    for core in ORDER:
        for j, temp in enumerate(T):
            r = rows[(qty, core, j)]
            dump["fits"][f"{qty}|{core}|{temp:.0f}"] = {
                "b_lin": r["b_lin"], "ln_a_lin": r["lna_lin"],
                "r2_lin": r["r2_lin"],
                "a2": r["a2"], "a2_ci95": r["a2_ci"], "a2_p": r["a2_p"],
                "a1": r["a1"], "a1_ci95": r["a1_ci"],
                "a0": r["a0"], "a0_ci95": r["a0_ci"],
                "r2_quad": r["r2_quad"],
                "dAIC": r["dAIC"], "dAICc": r["dAICc"],
                "leverage_50Hz": r["lev50"],
                "a2_without_50Hz": r["no50"]["a2"],
                "p_a2_without_50Hz": r["no50"]["a2_p"]}
(OUT / "fit_results.json").write_text(json.dumps(dump, indent=1), encoding="utf-8")
say("  wrote fit_results.json")

(OUT / "analysis_report.txt").write_text(report.getvalue(), encoding="utf-8")
print(f"\nwrote analysis_report.txt to {OUT}")
