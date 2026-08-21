"""
Internal consistency check of the repository.

core_data.py is a hand transcription of the finite-element output, and the
figures, tables and statistics all descend from it. This script checks that
the transcription still matches the files it came from, that the archived
finite-element results are converged and complete, that the geometry
constants agree with the CSG source, and that the data obey the relations
that the physics requires.

Every check prints PASS, FAIL or NOTE. A NOTE is a known property of the
archive, not an error. The script exits 1 if anything fails.

Run from anywhere:  python3 analysis/verify_data.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core_data import CORES, F, GEOM, T                 # noqa: E402
from loglog_fit import summary_row                      # noqa: E402
from paths import CASE, CASE_SUPERSEDED, FEM, ROOT     # noqa: E402

CASE_LABEL = {"full": "no gap", "gap017": "2 x 0.17 mm",
              "gap047": "2 x 0.47 mm"}
GAP_HALF_WIDTH_MM = {"full": None, "gap017": 0.085, "gap047": 0.235}
TOL_CONVERGENCE = 0.02          # the identification stops at 2 %
U_EXPANDED = 8.7                # expanded uncertainty of sigma_eff, k = 2,
                                # from the budget in run_analysis.py
failures = []
notes = []


def check(ok, what, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {what}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(what)


def note(what, detail=""):
    print(f"  [NOTE] {what}" + (f"   {detail}" if detail else ""))
    notes.append(what)


def head(title):
    print("\n" + "=" * 74)
    print(title)
    print("-" * 74)


# ------------------------------------------------------------ 1. the grids
head("1. Measurement grid and matrix shapes")
check(np.all(np.diff(F) > 0), "frequencies strictly increasing",
      f"{F[0]:.0f} Hz to {F[-1]:.0f} Hz, {len(F)} points")
check(np.all(np.diff(T) > 0), "temperatures strictly increasing",
      f"{T[0]:.0f} C to {T[-1]:.0f} C, {len(T)} points")
for core, c in CORES.items():
    check(c["P"].shape == (len(F), len(T)) and c["S"].shape == (len(F), len(T)),
          f"{core}: both matrices are {len(F)} by {len(T)}")
    check(np.all(c["P"] > 0) and np.all(c["S"] > 0),
          f"{core}: every loss and conductivity is positive")
    check(np.all(np.isfinite(c["P"])) and np.all(np.isfinite(c["S"])),
          f"{core}: no missing values")

# ------------------------------------------------- 2. against the FEM files
head("2. core_data.py against the archived finite-element output")
for core, path in CASE.items():
    tgt = np.loadtxt(path / "res_step_Pc_IMN.txt")
    st = np.loadtxt(path / "res_step_ST.txt")
    P, S = CORES[core]["P"], CORES[core]["S"]
    check(np.allclose(P, tgt, rtol=1e-9, atol=0),
          f"{core}: P matches res_step_Pc_IMN.txt",
          f"max deviation {np.max(np.abs(P / tgt - 1)):.2e}")
    filled = st != 0.0
    if filled.all():
        check(np.allclose(S[filled], st[filled], rtol=1e-9, atol=0),
              f"{core}: sigma_eff matches res_step_ST.txt (all 42 points)")
    else:
        check(np.allclose(S[filled], st[filled], rtol=1e-9, atol=0),
              f"{core}: sigma_eff matches res_step_ST.txt where the file is "
              f"populated ({int(filled.sum())}/{filled.size} points)")
        note(f"{core}: res_step_ST.txt records "
             f"{int(filled.sum())}/{filled.size} points, the archive of a "
             f"resumed run that began at the fourth temperature; the rest "
             f"survive only in T13_2026_02/Script_analyse_02_26_01.m")

sup_tgt = np.loadtxt(CASE_SUPERSEDED / "res_step_Pc_IMN.txt")
check(np.allclose(sup_tgt, CORES["gap047"]["P"], rtol=1e-9, atol=0),
      "superseded 0.47 mm case was identified against the same targets")

# --------------------------------------------------------- 3. convergence
head("3. Convergence of the identification")
for core, path in CASE.items():
    tgt = np.loadtxt(path / "res_step_Pc_IMN.txt")
    got = np.loadtxt(path / "res_step_Pc.txt")
    filled = got != 0.0
    dev = np.abs(got[filled] / tgt[filled] - 1.0)
    check(dev.max() <= TOL_CONVERGENCE,
          f"{core}: modelled loss within {TOL_CONVERGENCE:.0%} of the target "
          f"at all {int(filled.sum())} recorded points",
          f"worst {dev.max():.2%}")

# ----------------------------------------------------------- 4. geometry
head("4. Geometry constants against the CSG source")
r_in, r_out, h = GEOM["ID_mm"] / 2, GEOM["OD_mm"] / 2, GEOM["H_mm"]
area = (r_out - r_in) * h / 100.0                      # mm^2 -> cm^2
check(abs(area - GEOM["area_geom_cm2"]) < 5e-3,
      "geometric cross-section follows from OD, ID and height",
      f"{area:.3f} cm2 against {GEOM['area_geom_cm2']:.2f} cm2")
check(abs(GEOM["area_eff_cm2"] - GEOM["fill_factor"] * GEOM["area_geom_cm2"])
      < 5e-3, "effective cross-section is the fill factor times the geometric")
path_len = 2 * np.pi * GEOM["mean_radius_mm"] / 1000.0
check(abs(path_len - GEOM["path_len_m"]) < 1e-4,
      "magnetic path length is the mean circumference",
      f"{path_len:.5f} m against {GEOM['path_len_m']:.5f} m")
check(abs(GEOM["mean_radius_mm"] - 0.5 * (r_in + r_out)) < 1e-9,
      "mean radius is halfway between the bores")
layers = 0.5 * (GEOM["OD_mm"] - GEOM["ID_mm"]) * 1000.0 \
    / (GEOM["ribbon_um"] / GEOM["fill_factor"])
check(abs(layers - GEOM["n_layers"]) < 1.0,
      "ribbon layer count follows from the radial build and the pitch",
      f"{layers:.1f} against {GEOM['n_layers']}")

for core, path in CASE.items():
    geo = (path / "pierscien01.geo").read_text(encoding="utf-8")
    check(f"{r_in:.1f}" in geo and f"{r_out:.1f}" in geo,
          f"{core}: pierscien01.geo carries both core radii")
    half = GAP_HALF_WIDTH_MM[core]
    if half is None:
        check("spacer" not in geo, f"{core}: no spacer in the geometry")
    else:
        want = f"orthobrick  (-{half}, 13, -7; {half}, 27, 7)"
        check(want in geo,
              f"{core}: spacer is symmetric and {2 * half:.2f} mm wide")

# ------------------------------------------------------- 5. the B(H) curve
head("5. The measured B(H) curve")
hb_ref = None
for core, path in CASE.items():
    hb = np.loadtxt(path / "HB")
    check(hb.shape[1] == 2 and np.all(np.diff(hb[:, 0]) > 0)
          and np.all(np.diff(hb[:, 1]) > 0),
          f"{core}: HB is a strictly increasing B against H table",
          f"{hb.shape[0]} points, up to {hb[-1, 0]:.2f} T at "
          f"{hb[-1, 1]:.0f} A/m")
    if hb_ref is None:
        hb_ref = hb
    else:
        check(np.array_equal(hb, hb_ref),
              f"{core}: HB is the same curve as the ungapped case")
mu0 = 4e-7 * np.pi
mu_r = np.diff(hb_ref[:, 0]) / np.diff(hb_ref[:, 1]) / mu0
check(mu_r.max() < 1e6 and mu_r.min() > 0,
      "differential permeability stays positive and physical",
      f"{mu_r.min():.0f} to {mu_r.max():.0f} times mu0")
for core in CORES:
    bmax = CORES[core]["B"]
    check(bmax < hb_ref[-1, 0],
          f"{core}: Bmax = {bmax} T is inside the tabulated B(H) range")

# ------------------------------------------------------ 6. the flux profile
head("6. The radial flux profiles")
profiles = {}
for core, path in CASE.items():
    f_line = path / "line000.dat"
    if not f_line.exists():
        note(f"{core}: line000.dat absent, do_elmer_clear removes it")
        continue
    d = np.loadtxt(f_line)
    x = d[:, 3] * 1000.0
    check(d.shape[1] >= 21, f"{core}: line000.dat has the field columns",
          f"{d.shape[0]} nodes, {d.shape[1]} columns")
    step = np.diff(x)
    check(np.allclose(step, step[0], atol=1e-6),
          f"{core}: sampling line is uniform",
          f"{step[0]:.3f} mm spacing, x from {x[0]:.1f} to {x[-1]:.1f} mm")
    b = np.sqrt((d[:, 15:18] ** 2).sum(1) + (d[:, 18:21] ** 2).sum(1))
    sel = (x >= r_in) & (x <= r_out - 0.05)
    r, ratio = x[sel], b[sel] / CORES[core]["B"]
    profiles[core] = (r, ratio)
    check(np.all(ratio > 0) and np.all(np.isfinite(ratio)),
          f"{core}: |B| is positive across the core",
          f"{ratio.min():.2f} to {ratio.max():.2f} times Bmax")
    check(ratio[0] > np.interp(20.0, r, ratio),
          f"{core}: |B| is largest at the inner edge, as skin effect and "
          f"the 1/r fall of H both require")

if "full" in profiles:
    r, ratio = profiles["full"]
    check(abs(ratio[0] / ratio[-1] - r[-1] / r[0]) < 0.1,
          "ungapped profile follows the 1/r law of an unshielded toroid",
          f"edge ratio {ratio[0] / ratio[-1]:.3f} against "
          f"r_out/r_in = {r[-1] / r[0]:.3f}")
if len(profiles) == 3:
    contrast = [profiles[c][1].max() / profiles[c][1].min()
                for c in ("full", "gap017", "gap047")]
    check(contrast[0] < contrast[1] < contrast[2],
          "the flux becomes less uniform as the gap widens",
          "peak-to-trough " + ", ".join(f"{c:.1f}" for c in contrast))

# ------------------------------------------------- 7. physical consistency
head("7. Relations the physics requires")
dev_b = {}
for core in CORES:
    P, S = CORES[core]["P"], CORES[core]["S"]
    bP = np.array([summary_row(F, P[:, j])["b_lin"] for j in range(len(T))])
    bS = np.array([summary_row(F, S[:, j])["b_lin"] for j in range(len(T))])
    d = bS - (bP - 2.0)
    dev_b[core] = d
    check(np.all(d > 0) and d.max() < 0.5,
          f"{core}: b_sigma exceeds b_P - 2, the sign shielding demands",
          f"{d.min():+.3f} to {d.max():+.3f}")
check(dev_b["full"].mean() < dev_b["gap017"].mean() < dev_b["gap047"].mean(),
      "the departure from the unshielded relation grows with the gap",
      "mean " + ", ".join(f"{dev_b[c].mean():+.3f}"
                          for c in ("full", "gap017", "gap047")))

for core in CORES:
    S = CORES[core]["S"]
    check(np.all(S[:, -1] < S[:, 0]),
          f"{core}: sigma_eff at 120 C is below its value at 20 C at every "
          f"frequency",
          f"ratio {(S[:, -1] / S[:, 0]).min():.2f} to "
          f"{(S[:, -1] / S[:, 0]).max():.2f}")
    steps = np.diff(S, axis=1)
    up = int((steps > 0).sum())
    if up:
        note(f"{core}: {up}/{steps.size} steps in temperature go the other "
             f"way, which is the scatter of the loss measurement, not a trend")

for core in CORES:
    P = CORES[core]["P"]
    check(np.all(np.diff(P, axis=0) > 0),
          f"{core}: the loss grows with frequency at every temperature")

deltas = {}
for core in CORES:
    bmax = CORES[core]["B"]
    h_at_b = float(np.interp(bmax, hb_ref[:, 0], hb_ref[:, 1]))
    mu = bmax / h_at_b
    sig = CORES[core]["S"][-1, -1]
    deltas[core] = np.sqrt(2.0 / (2 * np.pi * F[-1] * mu * sig))
    check(deltas[core] * 1000.0 < 0.5 * (r_out - r_in),
          f"{core}: 20 kHz penetration depth is smaller than the radial build",
          f"{deltas[core] * 1000:.2f} mm against "
          f"{r_out - r_in:.1f} mm of ribbon stack")

# ---------------------------------------------------- 8. the sigma minimum
head("8. Non-monotonic sigma_eff of the widest-gap core")
S = CORES["gap047"]["S"]
imin = np.argmin(S[1:, :], axis=0) + 1
rise = 100.0 * (S[-1, :] / S.min(axis=0) - 1.0)
check(np.all(F[imin] == F[imin][0]),
      "the minimum sits at one frequency at every temperature",
      f"{F[imin][0]:.0f} Hz")
check(rise.min() > U_EXPANDED,
      f"the rise to 20 kHz exceeds the expanded uncertainty of "
      f"{U_EXPANDED} % (k = 2)",
      f"{rise.min():.1f} % to {rise.max():.1f} %")
old = np.loadtxt(CASE_SUPERSEDED / "res_step_ST.txt")
iold = np.argmin(old, axis=0)
agree = int((F[iold] == F[imin][0]).sum())
rise_old = 100.0 * (old[-1, :] / old.min(axis=0) - 1.0)
check(agree >= 5 and rise_old.min() > U_EXPANDED,
      "the superseded spacer geometry reproduces the same shape, so it is "
      "not an artefact of the spacer-width error",
      f"minimum at {F[imin][0]:.0f} Hz in {agree}/6 temperatures, "
      f"rise {rise_old.min():.1f} % to {rise_old.max():.1f} %")
if agree < len(T):
    note("superseded geometry: at "
         + ", ".join(f"{T[j]:.0f} C" for j in range(len(T))
                     if F[iold[j]] != F[imin][0])
         + f" its minimum sits at {F[iold[np.argmax(F[iold] != F[imin][0])]]:.0f}"
         f" Hz instead")
for core in ("full", "gap017"):
    Sc = CORES[core]["S"]
    r = 100.0 * (Sc[-1, :] / Sc.min(axis=0) - 1.0)
    if r.max() == 0.0:
        note(f"{core}: sigma_eff falls monotonically over the whole band")
    else:
        note(f"{core}: sigma_eff turns up by at most {r.max():.1f} %, inside "
             f"the {U_EXPANDED} % expanded uncertainty, so the turn is not "
             f"resolved")

# ------------------------------------------------------- 9. completeness
head("9. Files the pipeline needs")
for core, path in CASE.items():
    for f in ("res_step_Pc_IMN.txt", "res_step_ST.txt", "res_step_Pc.txt",
              "res_step_HT.txt", "pierscien01.geo", "pierscien01.sif.src",
              "HB"):
        check((path / f).exists(), f"{core}/{f}")
for f in ("README.md", "LICENSE", "CITATION.cff", "Makefile",
          "requirements.txt", "docs/documentation.md", "data/README.md"):
    check((ROOT / f).exists(), f)
prov = FEM / "T13_2026_02" / "Script_analyse_02_26_01.m"
check(prov.exists(), "data/fem/T13_2026_02/Script_analyse_02_26_01.m")
if prov.exists() and "loglog_lin_ident" in prov.read_text(encoding="utf-8") \
        and not list(prov.parent.glob("loglog_lin_ident.m")):
    note("the provenance script calls loglog_lin_ident(), an Octave function "
         "that is not archived, so its fitting section will not run; its "
         "data blocks are the part core_data.py was transcribed from")

# ------------------------------------------------------------------ verdict
print("\n" + "=" * 74)
if failures:
    print(f"{len(failures)} check(s) FAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"All checks passed. {len(notes)} note(s) on the archive.")
