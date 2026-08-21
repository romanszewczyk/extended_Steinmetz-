"""
Locations used by the analysis scripts.

Everything is resolved from the position of this file, so the scripts run
correctly from any working directory and the repository can be cloned
anywhere.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FEM = ROOT / "data" / "fem"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

# the three finite-element cases that produced the tabulated conductivities
CASE = {
    "full": FEM / "T10_fullT_01",
    "gap017": FEM / "T10_gap017",
    "gap047": FEM / "T10_gap047_02_corrected",
}
CASE_SUPERSEDED = FEM / "T10_gap047_01T_superseded"
