# Losses and effective conductivity of a gapped nanocrystalline toroidal core

Measured core losses, the effective conductivities identified from them by a
three-dimensional finite-element model, the finite-element cases themselves,
and the code that fits and compares four descriptions of their frequency
dependence.

## The measurement

One FINEMET-type nanocrystalline ribbon toroidal core, 49 mm outer diameter,
31 mm inner diameter, 10 mm high, was measured on a Remacomp C-1200
according to IEC 60404-6 at seven frequencies from 50 Hz to 20 kHz and six
temperatures from 20 to 120 °C. The same core was measured in three states:
without a gap, with two symmetrical 0.17 mm air gaps, and with two
symmetrical 0.47 mm gaps. That is 126 operating points.

The ungapped core and the 0.17 mm gaps were measured at a peak flux density
of 0.2 T. The 0.47 mm core was measured at 0.1 T, because the magnetising
source could not reach 0.2 T against the wider gaps. Comparisons across the
three configurations therefore mix two amplitudes, and the amplitude
dependence of the fitted coefficients is not characterised here.

## The identification

At each operating point a finite-element model of that geometry was solved
repeatedly, adjusting the magnetising field and the conductivity, until the
modelled flux density and the modelled Joule loss both matched the
measurement within 2 %. The conductivity that comes out is the effective
conductivity σ_eff of the homogenised core material. The model has no
hysteresis, so the whole measured loss, hysteretic and excess components
included, ends up as Joule dissipation and is absorbed into σ_eff. At 50 Hz
that makes σ_eff closer to a restatement of the hysteresis loss than to a
conductivity.

These numbers carry the flux-normalisation convention of section 5.3 of
`docs/documentation.md`, and they reproduce the measured losses only with
that convention. Since σ_eff already contains every loss mechanism, feeding
these maps into a simulation alongside a separate hysteresis model counts
the hysteresis loss twice.

## What the data show

For the widest gaps σ_eff is not monotonic in frequency. It falls to a
minimum at 5 kHz and rises again towards 20 kHz by 18 to 27 %, at all six
temperatures, against an expanded uncertainty of 8.7 % (k = 2). The ungapped
core falls monotonically over the whole band; the 0.17 mm core turns up by
at most 7.9 %, which the uncertainty does not resolve.

No loss-separation model can reproduce a minimum followed by a rise. Under
P ∝ σ f² B² every such model maps onto a sum of non-negative terms in 1/f,
f^−1/2 and a constant, and any such sum decreases monotonically towards its
constant. A power law cannot reproduce it either: fitted to σ_eff of the
0.47 mm core it is in error by up to 113 %, where a second-order polynomial
in log–log coordinates stays within 9.5 % and AICc prefers it at every
temperature by 15 to 29.

`results/model_comparison.csv` carries the same comparison for the other two
configurations and for the losses themselves, where the verdict is less
one-sided. For the ungapped core the power law is already adequate and AICc
prefers it at four of six temperatures.

## Rebuilding the results

The identified data are transcribed into `analysis/core_data.py`, so nothing
has to be re-solved to regenerate every figure and table.

```bash
pip install -r requirements.txt
make            # analysis, figures, then the consistency check
```

Individual steps:

```bash
make analysis   # fits, statistics, result tables, results/analysis_report.txt
make figures    # the five figures
make verify     # rebuild, then check everything against the source data
make help       # every target
```

`analysis/run_analysis.py` prints its whole report to the console and saves
it to `results/analysis_report.txt`. That file holds the residual statistics,
the model comparison, the robustness checks and the uncertainty budget, in
the order they were computed, and it is the place to look for any number
quoted above.

`analysis/verify_data.py` is the self-check. It confirms that the matrices in
`core_data.py` still match the finite-element files they were transcribed
from, that every recorded point converged inside its 2 % tolerance, and that
the geometry constants agree with the CSG source. It then tests the data
against what the physics requires of them: the flux profile of the ungapped
core follows 1/r to within 1 %, the departure of b_σ from b_P − 2 has the
sign shielding demands and grows with the gap, σ_eff falls with temperature
at every frequency, and the penetration depth at 20 kHz stays well below the
radial build. Where the archive is incomplete rather than wrong, the script
says so in a note instead. It exits non-zero if any check fails.

Python 3.9 or later with numpy, scipy and matplotlib is enough. Tested
versions are in `requirements.txt`.

## Layout

| Path | Content |
|---|---|
| `analysis/` | The pipeline: the data module, the fitting library, the competing models, the statistics driver, the figure builder and the self-check |
| `data/fem/` | The finite-element cases: geometries, solver templates, the measured B(H) curve, drivers, identification results and flux profiles |
| `docs/documentation.md` | The reference: what every component contains, how the numbers were produced, which conventions they carry, and what limits their reuse |
| `figures/` | Figures, vector PDF and 500 dpi PNG, all generated by `make_figures.py` |
| `results/` | Generated tables, coefficient CSV, JSON dump and the full numerical report |

`data/README.md` describes the measured and identified data file by file.

## Figures

| File | Content |
|---|---|
| `power_loss.pdf` / `.png` | measured losses and the extended fits |
| `effective_conductivity.pdf` / `.png` | identified effective conductivity |
| `local_exponent.pdf` / `.png` | local exponent n_eff(f) of both quantities |
| `curvature_excess.pdf` / `.png` | curvature of σ_eff and its excess over that of P |
| `flux_profile.pdf` / `.png` | radial flux profile at 20 kHz and 120 °C |

## Re-running the finite-element identification

Rebuilding the conductivities from the measured losses takes hours to days
and needs Elmer with `ElmerSolver_mpi`, NETGEN, OpenMPI, and GNU Octave for
two of the three cases. `docs/documentation.md` describes the cases, the
solver configuration and how to drive each one, and
`data/fem/T10_gap047_02_corrected/README.md` documents the Python and MPI
sweep in detail.

## Provenance

```
measurement (Remacomp C-1200, IEC 60404-6)
  -> data/fem/*/res_step_Pc_IMN.txt      target losses, 7 frequencies by 6 temperatures
  -> finite-element identification        drivers in each case directory
  -> data/fem/*/res_step_ST.txt           effective conductivity
  -> analysis/core_data.py                transcribed by hand
  -> run_analysis.py -> results/,  make_figures.py -> figures/
```

`analysis/core_data.py` is a hand transcription, not a live link. Re-running
any case means editing that file and regenerating everything downstream, and
`make verify` will tell you if the two have drifted apart.

## Citing

Citation metadata is in `CITATION.cff`. The archived release carries

    doi:10.5281/zenodo.XXXXXXX

`XXXXXXX` is a placeholder written so that it cannot be mistaken for a
resolvable identifier. Replace it with the one issued when the archive is
deposited:

```bash
grep -rl '10.5281/zenodo.XXXXXXX' .
```

## Licence

MIT. See `LICENSE`.
