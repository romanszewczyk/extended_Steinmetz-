"""
Measured core losses and identified effective conductivity of a
FINEMET-type nanocrystalline ribbon toroidal core in three states: without
a gap, with two symmetrical 0.17 mm gaps, and with two symmetrical 0.47 mm
gaps.

Provenance
  P_full, P_gap017, P_gap047, S_full, S_gap017
      data/fem/T13_2026_02/Script_analyse_02_26_01.m
  S_gap047
      data/fem/T10_gap047_02_corrected/res_step_ST.txt, the
      complete 42-point re-identification of 2026-07-11 that uses the
      corrected symmetric spacer geometry (+-0.235 mm).  The earlier
      matrices for this configuration were computed with 0.32 mm wide
      spacer blocks and are superseded.

Layout: rows are the 7 frequencies F, columns the 6 temperatures T.

P   measured total core loss, W, Remacomp C1200, IEC 60404-6
S   effective conductivity of the homogenised core material, S/m,
    identified with NETGEN / Elmer / GNU Octave

Peak flux density of the measurements:
    no gap        Bmax = 0.2 T
    2 x 0.17 mm   Bmax = 0.2 T
    2 x 0.47 mm   Bmax = 0.1 T, limited by the magnetising current source

Flux normalisation used by the identification.  The control quantity of
the inner loop is the flux integral over a rectangular window enclosing
the core cross-section, divided by the reference area A0 = 1 cm^2, and
that quantity is driven to Bmax.  Because the geometric cross-section of
the modelled ring is 0.90 cm^2, the corresponding cross-section average
of |B| is 1.10 Bmax (verified on the stored FEM field of the 0.47 mm
case).  A constant offset of the flux normalisation rescales S by a
frequency- and temperature-independent factor and therefore shifts only
the constant term a0 of the extended fit; see docs/documentation.md.

verify_data.py checks these matrices against the files they were
transcribed from.
"""
import numpy as np

F = np.array([50.0, 500.0, 1000.0, 5000.0, 10000.0, 15000.0, 20000.0])  # Hz
T = np.array([20.0, 40.0, 60.0, 80.0, 100.0, 120.0])                    # degC

# ----------------------------------------------------------------------
# Full (ungapped) core, Bmax = 0.2 T
# ----------------------------------------------------------------------
P_full = np.array([
    [0.00069, 0.00065, 0.00053, 0.00043, 0.00037, 0.0003],
    [0.011,   0.011,   0.009,   0.0077,  0.0066,  0.0057],
    [0.028,   0.026,   0.022,   0.019,   0.016,   0.014],
    [0.22,    0.21,    0.18,    0.16,    0.14,    0.12],
    [0.55,    0.52,    0.44,    0.39,    0.34,    0.29],
    [0.92,    0.87,    0.73,    0.66,    0.57,    0.48],
    [1.4,     1.3,     1.1,     0.96,    0.83,    0.7],
])

S_full = np.array([
    [7.74041702e+03, 7.28762423e+03, 5.93415047e+03, 4.80970868e+03, 4.13682005e+03, 3.35250133e+03],
    [1.24325271e+03, 1.24341632e+03, 1.01300687e+03, 8.66446394e+02, 7.40797685e+02, 6.38801405e+02],
    [7.97426559e+02, 7.38448179e+02, 6.21860127e+02, 5.35271602e+02, 4.49543187e+02, 3.92669700e+02],
    [2.58324156e+02, 2.45388598e+02, 2.07916811e+02, 1.83412242e+02, 1.59430264e+02, 1.35936202e+02],
    [1.66187828e+02, 1.56064757e+02, 1.29132206e+02, 1.13201534e+02, 9.79648229e+01, 8.27973198e+01],
    [1.26416065e+02, 1.18072130e+02, 9.63117616e+01, 8.60610661e+01, 7.34913821e+01, 6.11656342e+01],
    [1.11785610e+02, 1.01681318e+02, 8.33256417e+01, 7.12831178e+01, 6.06119963e+01, 5.04255914e+01],
])

# ----------------------------------------------------------------------
# Core with two symmetrical gaps of 0.17 mm, Bmax = 0.2 T
# ----------------------------------------------------------------------
P_gap017 = np.array([
    [0.00052975, 0.00042392, 0.0003475,  0.00031977, 0.00035893, 0.0003391],
    [0.0097768,  0.0091203,  0.0083654,  0.0079045,  0.0075666,  0.0073192],
    [0.024569,   0.022878,   0.02108,    0.021212,   0.021277,   0.020243],
    [0.23069,    0.22277,    0.21087,    0.2102,     0.20896,    0.20892],
    [0.63172,    0.61159,    0.59213,    0.58,       0.57178,    0.57179],
    [1.1035,     1.113,      1.013,      1.0091,     1.012,      1.012],
    [1.805,      1.7046,     1.7065,     1.7137,     1.6054,     1.6054],
])

S_gap017 = np.array([
    [5.99553731e+03, 4.78631644e+03, 3.91794387e+03, 3.60354534e+03, 4.04746482e+03, 3.82267653e+03],
    [1.14476860e+03, 1.05663685e+03, 9.61743158e+02, 9.05819724e+02, 8.66716822e+02, 8.36938112e+02],
    [7.16846517e+02, 6.63885894e+02, 6.28922541e+02, 6.28897747e+02, 6.28175583e+02, 5.95780727e+02],
    [2.99676086e+02, 2.85765435e+02, 2.66240860e+02, 2.65198712e+02, 2.63233505e+02, 2.62889583e+02],
    [2.36448369e+02, 2.24589011e+02, 2.12867795e+02, 2.06941979e+02, 2.02838908e+02, 2.02816735e+02],
    [2.00883586e+02, 2.01444801e+02, 1.74717835e+02, 1.72947378e+02, 1.73080353e+02, 1.73081104e+02],
    [2.05282604e+02, 1.94110690e+02, 1.82362844e+02, 1.86599024e+02, 1.70873201e+02, 1.70873942e+02],
])

# ----------------------------------------------------------------------
# Core with two symmetrical gaps of 0.47 mm, Bmax = 0.1 T
# ----------------------------------------------------------------------
P_gap047 = np.array([
    [0.00024271, 0.00021274, 0.0002401,  0.00014825, 0.00010121, 0.00016389],
    [0.002809,   0.0025195,  0.0023093,  0.0021131,  0.0018795,  0.0018181],
    [0.0075296,  0.0070184,  0.0064867,  0.006159,   0.0058252,  0.0055611],
    [0.094538,   0.090887,   0.087489,   0.085691,   0.082661,   0.082414],
    [0.27127,    0.26109,    0.26216,    0.25204,    0.25099,    0.24134],
    [0.48855,    0.48066,    0.46082,    0.44759,    0.44183,    0.43445],
    [0.8147,     0.79341,    0.77425,    0.75599,    0.7431,     0.72652],
])

S_gap047 = np.array([
    [1.13735859e+04, 9.90907066e+03, 1.12453959e+04, 7.05814091e+03, 4.79577222e+03, 7.57046692e+03],
    [1.37009122e+03, 1.22020411e+03, 1.11375823e+03, 9.81932379e+02, 9.02644917e+02, 8.41733418e+02],
    [9.42434572e+02, 8.70962324e+02, 7.76935122e+02, 7.33760274e+02, 7.17172346e+02, 6.82784980e+02],
    [6.71126451e+02, 6.25340611e+02, 5.85463504e+02, 5.64395386e+02, 5.31113010e+02, 5.27641218e+02],
    [6.96199978e+02, 6.41760624e+02, 6.19397174e+02, 5.82269354e+02, 5.59370951e+02, 5.35439891e+02],
    [7.05004765e+02, 6.64702993e+02, 6.25271905e+02, 5.89266113e+02, 5.69178572e+02, 5.43185532e+02],
    [7.94660728e+02, 7.54806460e+02, 7.33502641e+02, 7.00109526e+02, 6.77363218e+02, 6.48320968e+02],
])

CORES = {
    "full":   {"P": P_full,   "S": S_full,   "B": 0.2, "gap_mm": 0.0,
               "label": "no gap"},
    "gap017": {"P": P_gap017, "S": S_gap017, "B": 0.2, "gap_mm": 0.17,
               "label": "2 gaps, 0.17 mm"},
    "gap047": {"P": P_gap047, "S": S_gap047, "B": 0.1, "gap_mm": 0.47,
               "label": "2 gaps, 0.47 mm"},
}

# Core geometry (source: Wymiary rdzenia.xlsx, not kept here).
# fill_factor is consistent with area_eff_cm2 = fill_factor * area_geom_cm2.
# n_layers is the radial build divided by the ribbon pitch,
# (OD-ID)/2 / (ribbon_um / fill_factor).
GEOM = {
    "OD_mm": 49.0, "ID_mm": 31.0, "H_mm": 10.0,
    "path_len_m": 0.12566, "area_geom_cm2": 0.90, "area_eff_cm2": 0.72,
    "fill_factor": 0.80, "ribbon_um": 22.0,
    "area_ref_cm2": 1.00,      # A0 of the FEM flux normalisation
    "mean_radius_mm": 20.0,    # radius at which H of the FEM source is set
}
GEOM["n_layers"] = int(round(
    0.5 * (GEOM["OD_mm"] - GEOM["ID_mm"]) * 1000.0
    / (GEOM["ribbon_um"] / GEOM["fill_factor"])))
