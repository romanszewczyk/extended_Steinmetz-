# Documentation

`../README.md` gives the short rebuild recipe and `../data/README.md`
describes the data files. This is the reference: what each component
contains, how the numbers were produced, which conventions they carry, and
what limits their reuse.

---

## 1. What was measured and computed

Core losses of one FINEMET-type nanocrystalline ribbon toroidal core were
measured in three states (no gap; two symmetrical 0.17 mm gaps; two
symmetrical 0.47 mm gaps) at 7 frequencies from 50 Hz to 20 kHz and 6
temperatures from 20 to 120 °C, giving 126 operating points. For each point
a finite-element model of the core was tuned until it reproduced the measured
loss; the tuned scalar is the effective conductivity σ_eff of the homogenised
core material.

Both the measured losses and the identified σ_eff are then described, at each
temperature separately, by four models: the classical Steinmetz power law, a
second-order polynomial in log–log coordinates, and two loss-separation
forms. σ_eff of the widest-gap core turns out to be non-monotonic in
frequency, which excludes every loss-separation model structurally and the
power law empirically, and the polynomial describes it to within 9.5 %.

## 2. Reproduction levels

| Level | What is re-run | Requirements | Time |
|---|---|---|---|
| A | All fits, statistics, tables and figures | Python 3 with numpy, scipy, matplotlib | about 1 minute |
| B | The finite-element identification of σ_eff(f,T) | Elmer with `ElmerSolver_mpi`, NETGEN, OpenMPI; GNU Octave for two of the three cases | hours to days |

Level A is self-contained and deterministic. Level B regenerates the
`res_step_*.txt` matrices from the measured targets; because the
identification stops at 2 % tolerances, re-identified values may differ from
the archived ones by up to about 2 % without anything being wrong.

### Level A commands

```bash
make            # everything below, then the self-check
```

or, step by step,

```bash
python3 analysis/run_analysis.py
python3 analysis/make_figures.py
python3 analysis/verify_data.py
```

The three scripts resolve their paths from their own location, so the working
directory does not matter.

## 3. Repository layout

```
.
├── README.md, LICENSE, CITATION.cff, Makefile, requirements.txt
├── analysis/
│   ├── paths.py                          where everything lives (section 4)
│   ├── core_data.py                      the identified data (section 6)
│   ├── loglog_fit.py                     regression library (section 7)
│   ├── loss_models.py                    competing models (section 8)
│   ├── run_analysis.py                   statistics driver (section 9)
│   ├── make_figures.py                   figure builder (section 10)
│   └── verify_data.py                    the self-check (section 11)
├── data/
│   ├── README.md                         the data files, one by one
│   └── fem/                              the finite-element cases (section 5)
├── docs/documentation.md                 this file
├── figures/                              output of make_figures.py
└── results/                              output of run_analysis.py
```

## 4. `analysis/paths.py`

Every path used by the pipeline is resolved from the position of that file,
so the scripts work from any working directory and the repository can be
cloned anywhere. `CASE` maps the three configuration keys used throughout the
code, `full`, `gap017` and `gap047`, onto the case directories under
`data/fem/`. Moving or renaming a directory means editing this one module and
nothing else.

## 5. The finite-element cases

Three production cases plus one superseded set:

| Directory | Configuration | Driver |
|---|---|---|
| `T10_fullT_01` | ungapped, B_max = 0.2 T | `Script_optim03all.m` (Octave) |
| `T10_gap017` | two 0.17 mm gaps, B_max = 0.2 T | `Script_optim03all.m` (Octave) |
| `T10_gap047_02_corrected` | two 0.47 mm gaps, B_max = 0.1 T | `run_sweep.sh` with `run_sweep.py` (Python and MPI) |
| `T10_gap047_01T_superseded` | the same, computed with 0.32 mm spacer blocks | results only, kept for one robustness check |

### 5.1 Files in a case

- `pierscien01.geo`: NETGEN CSG geometry. The core is an annulus of inner
  radius 15.5 mm, outer radius 24.5 mm and height 10 mm, so its geometric
  cross-section is 0.90 cm². The spacers of the gapped cases are
  `orthobrick` blocks on the ±y axis; in the corrected 0.47 mm geometry they
  span x from −0.235 to +0.235 mm, and in the 0.17 mm geometry from −0.085
  to +0.085 mm. The magnetising conductor is a cylinder of 5 mm radius on the
  axis. The air region is a sphere of 1000 mm radius. Element size is capped
  at 0.50 mm inside the core.
- `pierscien01.sif.src`: Elmer solver template. The driver substitutes the
  frequency (line 10), the field amplitude (line 11), the conductivity
  (line 83) and the linear-solver tolerances (lines 168 and 195) **by line
  number**, so editing the template means updating those numbers in
  `Fun_Prepare_Sif.m` in the same directory, or in `write_sif()` of
  `run_sweep.py` for the 0.47 mm case. `pierscien01.sif` is generated per run
  and removed by `do_elmer_clear`.
- `HB`: the measured apparent B(H) curve, B in T against H in A/m, 23 points
  up to 1.05 T at 1500 A/m. Recorded on the ungapped core, and the same curve
  is used for every temperature and every configuration.
- `do_mesh`, `do_elmer`, `do_elmer_clear`: mesh, solve and clean drivers.
  `do_mesh` calls NETGEN in batch mode and then `ElmerGrid`; it needs an X
  display, and `run_sweep.sh` contains the headless equivalent.
- `res_step_HT.txt`, `res_step_ST.txt`, `res_step_Pc.txt`,
  `res_step_Pc_IMN.txt`: converged field amplitude (A/m), effective
  conductivity (S/m), computed loss (W) and measured target loss (W), each a
  7 by 6 matrix with frequency rows and temperature columns. A zero marks a
  point that the file does not record.
- `line000.dat`: the radial sampling line of the last converged run, 601
  nodes at 0.1 mm spacing along the x axis at mid-height, that is in the limb
  perpendicular to the gaps. Column 4 (1-based) is x in metres and columns 16
  to 21 hold the real and imaginary parts of the flux density.

### 5.2 Solver configuration

Elmer 9.0, `MagnetoDynamics` / `WhitneyAVHarmonicSolver`, edge elements with
the Piola transform, BiCGStab(6) at a linear tolerance of 6·10⁻⁵. The
nonlinear B(H) relation enters through an effective reluctivity evaluated
from the local flux-density amplitude. **There is no hysteresis in the
model.** The whole measured loss, hysteretic and excess components included,
is therefore mapped onto Joule dissipation and absorbed into σ_eff. At 50 Hz
that makes σ_eff close to a restatement of the hysteresis loss rather than a
conductivity.

The excitation is a uniform current density in the central conductor, set to
give the field amplitude H at the mean radius of 20 mm. Inside the core this
matches a distributed toroidal winding; outside it does not, and the
difference acts on the fringing region near the gaps. Since σ_eff is tuned to
match the measured loss, that difference is absorbed into σ_eff.

Mesh of the corrected 0.47 mm case: 267 036 nodes, 1 564 209 tetrahedra,
141 224 boundary elements. The 20 kHz, 120 °C point recomputed on a finer
mesh (461 000 nodes, 2.7·10⁶ elements) changes σ_eff by 0.15 % and H by
0.33 %.

### 5.3 The flux normalisation, and why it matters for reuse

This is the one convention that has to be understood before the tabulated
conductivities are reused.

The identification does not control the cross-section average of |B|. It
controls

    B_ctrl = (1/A0) * integral over W of |B| dA,   A0 = 1 cm²

where W is the rectangular window described in section 5.4, averaged over
its 1440 azimuthal angles. This is what `Fun_calc_rot.m` computes, and what
`analyse_B.py` replicates for the Python driver.

Because A0 = 1 cm² while the geometric cross-section is 0.90 cm², driving
B_ctrl to B_max leaves the true area average of |B| in the core at about
1.10 B_max. Recomputed on the stored three-dimensional field of the 0.47 mm
case: B_ctrl = 0.100001 T against a target of 0.1 T, true area mean
0.110187 T, ratio 1.102.

Consequence. Any constant offset between this normalisation and the one the
instrument used (which depends on whether the Remacomp was configured with
the geometric cross-section of 0.90 cm² or the effective one of 0.72 cm²)
rescales σ_eff by a factor that does not depend on frequency or temperature,
because σ_eff ∝ P/B² in the unshielded limit. Such a factor shifts the
constant a₀ of the fits and leaves a₂, a₁, n_eff, Δa₂ and the shape of
σ_eff(f) unchanged. The tabulated σ_eff values reproduce the measured losses
when used with the same convention, and only with it.

### 5.4 Running a case

```bash
cd data/fem/<case>
./do_mesh                        # NETGEN plus ElmerGrid, needs an X display
./do_elmer                       # one solve
./do_elmer_clear                 # remove generated artefacts
octave --eval "Script_optim03all"   # full sweep, ungapped and 0.17 mm cases
```

For the 0.47 mm case, from inside `T10_gap047_02_corrected`:

```bash
screen -S gap047
./run_sweep.sh
```

The sweep is resumable: interrupt it and re-run, and the points already in
`res_step_*.txt` are skipped. `SWEEP_NP` sets the MPI rank count (default
16), `NETGEN_PY` points at a Python interpreter that has the `netgen` module,
and `SWEEP_MIN_RUN_GB` and `SWEEP_MIN_START_GB` drive a memory watchdog that
exists because an earlier attempt on a finer mesh exhausted the machine. The
mesh is NETGEN `moderate`, not `fine`, and that choice is validated at the
reference point; see the case README.

The identification iterates on the pair (H, σ) until the control flux
density is within 2 % of B_max and the computed Joule loss within 2 % of the
measured loss. The Octave driver spends four solves per pass: three trial
amplitudes, inverse-interpolated onto B_max, then one solve with the
conductivity rescaled by the loss ratio. `run_sweep.py` instead updates both
in one proportional step per solve, which is why the archived 0.47 mm sweep
converged 17 of its 42 points on the first solve, 24 on the second and one on
the third. No point in any case failed to converge.

The window over which the control flux density is integrated is a 99 by 97
rectangle, r from 15.05 to 24.85 mm and z from −4.75 to +4.85 mm at 0.1 mm
step, rotated to 1440 azimuthal angles in 0.25° steps. Section 5.3 explains
why its area matters.

## 6. `analysis/core_data.py`

Frequencies, temperatures, the three loss matrices, the three conductivity
matrices, and the core geometry.

Provenance is in the module docstring. Everything except `S_gap047` comes
from `data/fem/T13_2026_02/Script_analyse_02_26_01.m`; `S_gap047` comes from
`data/fem/T10_gap047_02_corrected/res_step_ST.txt`, the complete 42-point
sweep of 2026-07-11 with the corrected spacer geometry.

**These matrices are a hand transcription, not a live link.** Re-running any
finite-element case means editing this file and regenerating everything
downstream. `verify_data.py` compares the two and fails if they have drifted.

`GEOM` carries the dimensions, the geometric and effective cross-sections,
the fill factor, the reference area A0 of the flux normalisation, the mean
radius at which the field amplitude is defined, and the derived layer count
(327).

## 7. `analysis/loglog_fit.py`

Ordinary least squares of ln y on a polynomial in ln f, which is the maximum
likelihood estimator when the relative error of y is constant.

`loglog_polyfit(x, y, degree)` returns coefficients, standard errors, 95 %
confidence intervals from Student's t with n−k degrees of freedom, the
coefficient covariance, R², the residual sum of squares, the hat-matrix
diagonal (leverage), the relative residuals in percent, and AIC together with
AICc.

AICc uses K = k + 1, counting the residual variance as an estimated
parameter, as Burnham and Anderson require for least squares. At n = 7 the
small-sample correction adds 8.0 to a two-parameter model and 20.0 to a
three-parameter one, so every ΔAICc is 12.0 below the corresponding ΔAIC.
This is the single most consequential detail of the statistics: with the
uncorrected AIC the extended model appears to win almost everywhere, and with
AICc it wins decisively only where it should.

`a2_difference_test(row1, row2)` runs a two-sided t test on the difference of
two curvature coefficients with Welch degrees of freedom, which is what
run_analysis.py uses instead of eyeballing whether confidence intervals
overlap.

## 8. `analysis/loss_models.py`

The competing physical descriptions, fitted on the same objective as the
log–log models so that AICc values are comparable.

Loss models: `k_h f + k_e f²` and `k_h f + k_x f^1.5 + k_e f²`.
Conductivity models: their images under P ∝ σ f² B², that is `c_h/f + c_e`
and `c_h/f + c_x f^−0.5 + c_e`.

All coefficients are constrained to be non-negative, which is what makes the
structural argument work: with non-negative coefficients both conductivity
forms are strictly decreasing in frequency and approach a positive constant,
so neither can produce a minimum followed by a rise.

Fitting minimises the sum of squared residuals of ln y, seeded from a
non-negative least-squares solution in linear space and from a
relative-error-weighted variant, then refined by
`scipy.optimize.least_squares` over log-parameters.

## 9. `analysis/run_analysis.py`

The single driver behind every derived number. It prints its report to the
console and writes the same text to `results/analysis_report.txt`:

1. Classical and extended fits of both quantities for all 18 datasets each,
   with b, R², a₂ ± CI, a₁, a₀, p(a₂), ΔAIC, ΔAICc and the 50 Hz leverage.
2. How the small-sample correction changes model selection, plus a Holm
   correction over the 36 simultaneous tests.
3. The practical size of the curvature, as the change of n_eff across the
   band.
4. The consistency relation with the unshielded limit and the curvature
   excess Δa₂.
5. The competing models, as AICc differences and pooled residuals.
6. Non-monotonicity of σ_eff read from the identified values.
7. Leverage of the 50 Hz point and every fit repeated without it.
8. The classical fit restricted to 50 Hz–10 kHz.
9. Differences of a₂ between configurations, and its trend with temperature.
10. The change of the losses between 20 and 120 °C.
11. Local exponents at the band edges and the stationary point of σ_eff.
12. The 0.47 mm minimum against the superseded 0.32 mm geometry.
13. Radial flux profiles.
14. The uncertainty budget.
15. Penetration depth against the observed surface layer.

Outputs into `results/`:

| File | Content |
|---|---|
| `fit_coefficients.csv` | every coefficient of both log–log fits with its standard error, confidence interval, leverage and residual statistics |
| `model_comparison.csv` | pooled relative residuals and AICc differences of the four descriptions, per quantity and configuration |
| `curvature_excess.csv` | a₂ of P, a₂ of σ_eff and their difference, per configuration and temperature |
| `sigma_minimum.csv` | where σ_eff(f) turns and how far it rises again |
| `fit_results.json` | the fits machine readable, plus Δa₂, the σ_eff rises and the uncertainty budget |
| `analysis_report.txt` | the full console report |

## 10. `analysis/make_figures.py`

Five figures as paired vector PDF and 500 dpi PNG. Everything in `figures/`
is generated by this script; nothing is copied in from elsewhere, and each
panel names its configuration and its flux density so that the figures need
no caption to be read.

Colour system: temperature is an ordered quantity and gets a single-hue blue
ramp; the configuration is an identity and gets a fixed colour slot plus its
own marker shape, so colour never carries identity alone and the panels
survive greyscale printing. Multi-panel figures are stacked vertically at
single-column width (90 mm).

Two details of the figures:

- `curvature_excess` draws no error bars on Δa₂. The two coefficients
  entering the difference come from the same measured losses and are strongly
  correlated, so combining their standard errors in quadrature would overstate
  the uncertainty. The spread of each curve across temperature, below 0.006,
  is the honest scale.
- `flux_profile` draws only the nodes inside the core, from r = 15.5 to
  24.45 mm. The SaveLine output holds the elemental field of the core body,
  which is written as a near-zero placeholder in the surrounding air.
  Plotting the full window makes the field look as though it collapses at the
  core surface.

## 11. `analysis/verify_data.py`

`core_data.py` is a hand transcription, and every figure and statistic
descends from it, so the repository needs a way to tell whether the two
halves have drifted apart. This script is it. Nine sections, each printing
PASS, FAIL or NOTE:

1. The grids are strictly increasing and every matrix is 7 by 6, positive and
   complete.
2. `core_data.py` matches the `res_step_*.txt` files it was transcribed from
   wherever those files are populated, and the superseded case was identified
   against the same targets.
3. Every recorded point converged inside its 2 % tolerance.
4. The geometry constants in `GEOM` follow from OD, ID and height, and the
   spacer widths in `pierscien01.geo` match the nominal gaps.
5. The B(H) curve is strictly increasing, identical across the three cases,
   and covers both measurement amplitudes.
6. The flux profiles are uniformly sampled, positive, largest at the inner
   edge; the ungapped profile follows the 1/r law of an unshielded toroid to
   within 1 %, and the peak-to-trough contrast grows with the gap.
7. b_σ exceeds b_P − 2 by an amount that grows with the gap, which is the sign
   and the ordering shielding demands; σ_eff at 120 °C is below its 20 °C
   value at every frequency; the loss grows with frequency everywhere; and
   the penetration depth at 20 kHz is well below the radial build.
8. The minimum of σ_eff of the 0.47 mm core sits at 5 kHz at every
   temperature and the rise to 20 kHz exceeds the expanded uncertainty, and
   the superseded geometry reproduces the same shape.
9. Every file the pipeline reads is present.

```bash
python3 analysis/verify_data.py    # exits 1 if any check fails
```

A NOTE is a property of the archive rather than an error. There are eight,
and they are the caveats of the dataset: the ungapped case records only 17 of
its 42 points, a few steps in temperature run the wrong way at the level of
the measurement scatter, the 0.17 mm core turns up by less than the
uncertainty resolves, the superseded geometry puts its minimum at 10 kHz at
120 °C rather than 5 kHz, and the Octave provenance script calls a function
that is not archived with it.

## 12. Provenance chain

```
measurement (Remacomp C-1200, IEC 60404-6)
  -> res_step_Pc_IMN.txt
  -> FEM identification (Octave or Python drivers)
  -> res_step_ST.txt, res_step_HT.txt
  -> analysis/core_data.py            hand transcription
  -> run_analysis.py  ->  results/
  -> make_figures.py  ->  figures/
```

## 13. Known limits

- The measured losses are the primary data. Everything else is derived.
- σ_eff carries the flux-normalisation convention of section 5.3 and absorbs
  every loss mechanism, hysteresis included. Combining these maps with a
  separate hysteresis model in a finite-element simulation double-counts.
- The three configurations were not all measured at the same flux density,
  and the amplitude dependence of the coefficients is not characterised.
- The ungapped case is archived incompletely: `res_step_*.txt` holds 17 of
  42 points, the record of a run resumed at the fourth temperature, and the
  rest survive only in the Octave analysis script.
- The Octave provenance script `Script_analyse_02_26_01.m` calls
  `loglog_lin_ident()`, which is not archived with it, so its fitting section
  will not run. Its data blocks are what `core_data.py` was transcribed from.
- Several experimental details a reader may want are recorded nowhere here:
  the number of turns and their distribution, the current source range, the
  loss resolution of the instrument, the stability of the thermal chamber,
  the excitation distortion, whether the anneal was field-free, and the core
  mass.
- The environment used for the archived runs: Elmer 9.0 (revision b5cfe83ca,
  built 2026-05-14), NETGEN 6.2.2606, ElmerGrid from the same Elmer build,
  Python 3.12 with numpy, scipy and matplotlib, GNU Octave for the two
  Octave-driven cases. Octave is not installed on the machine that produced
  the corrected 0.47 mm sweep, which is why that case was reimplemented in
  Python.
