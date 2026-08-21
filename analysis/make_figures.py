"""
Every figure of the analysis, written to figures/ as a paired vector PDF
and 500 dpi PNG.

  power_loss                  measured losses and the extended fits
  effective_conductivity      identified effective conductivity
  local_exponent              local exponent n_eff(f) of both quantities
  curvature_excess            a2 of sigma_eff, and the excess over a2 of P
  flux_profile                radial flux profile at 20 kHz, 120 C

Colour system: temperature is an ordered quantity and gets a single-hue
blue ramp; the configuration is an identity and gets a fixed colour slot
plus its own marker, so that the panels survive greyscale printing.

Run from anywhere:  python3 analysis/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core_data import CORES, F, T                      # noqa: E402
from loglog_fit import fit_both                        # noqa: E402
from paths import CASE, FIGURES                       # noqa: E402

FIGDIR = FIGURES
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.45,
    "lines.linewidth": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.axisbelow": True,
    "figure.dpi": 300,
    "savefig.dpi": 500,
})

T_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
T_MARKS = ["o", "s", "^", "D", "v", "P"]
C_COLORS = {"full": "#2a78d6", "gap017": "#1baf7a", "gap047": "#eda100"}
C_MARKS = {"full": "o", "gap017": "s", "gap047": "D"}
C_LABEL = {"full": "no gap", "gap017": "$2{\\times}0.17$ mm",
           "gap047": "$2{\\times}0.47$ mm"}
GRAY = "#52514e"
ORDER = ["full", "gap017", "gap047"]

MM = 1 / 25.4
W1 = 90 * MM
R_IN, R_OUT = 15.5, 24.45


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.png")
    plt.close(fig)
    print("saved", name)


def _fit_panels(qty, ylab, name, ylim=None, legend_loc="lower right"):
    fig, axes = plt.subplots(3, 1, figsize=(W1, 170 * MM), sharex=True,
                             sharey=True)
    fig.subplots_adjust(left=0.145, right=0.965, top=0.965, bottom=0.055,
                        hspace=0.16)
    ff = np.geomspace(F[0], F[-1], 200)
    for ax, core, letter in zip(axes, ORDER, "abc"):
        M = CORES[core][qty]
        for j in range(len(T)):
            _, quad = fit_both(F, M[:, j])
            ax.plot(ff, quad.predict(ff), "-", color=T_COLORS[j], lw=0.9,
                    zorder=2)
            ax.plot(F, M[:, j], T_MARKS[j], color=T_COLORS[j], ms=3.4,
                    mfc=T_COLORS[j], mec="white", mew=0.35, zorder=3,
                    label=f"{int(T[j])} $^\\circ$C")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both")
        ax.set_title(f"({letter}) {C_LABEL[core]}, "
                     f"$B_{{\\max}}={CORES[core]['B']}$ T",
                     loc="left", fontsize=9)
        ax.set_ylabel(ylab)
        if ylim:
            ax.set_ylim(*ylim)
    axes[-1].set_xlabel("$f$ (Hz)")
    axes[0].legend(loc=legend_loc, frameon=False, ncol=2, handletextpad=0.15,
                   borderaxespad=0.2, labelspacing=0.25, columnspacing=0.8)
    save(fig, name)


def power_loss():
    _fit_panels("P", "$P_{\\mathrm{loss}}$ (W)", "power_loss",
                ylim=(6e-5, 4), legend_loc="lower right")


def effective_conductivity():
    _fit_panels("S", "$\\sigma_{\\mathrm{eff}}$ (S/m)",
                "effective_conductivity",
                ylim=(3e1, 3e4), legend_loc="upper right")


def local_exponent():
    fig, axes = plt.subplots(2, 1, figsize=(W1, 125 * MM), sharex=True)
    fig.subplots_adjust(left=0.175, right=0.965, top=0.965, bottom=0.075,
                        hspace=0.16)
    ff = np.geomspace(F[0], F[-1], 200)
    for ax, qty, sym in ((axes[0], "P", "P_{\\mathrm{loss}}"),
                         (axes[1], "S", "\\sigma_{\\mathrm{eff}}")):
        for core in ORDER:
            M = CORES[core][qty]
            curves = np.array([fit_both(F, M[:, j])[1].n_eff(ff)
                               for j in range(len(T))])
            mean = curves.mean(0)
            ax.fill_between(ff, curves.min(0), curves.max(0),
                            color=C_COLORS[core], alpha=0.18, lw=0)
            ax.plot(ff, mean, "-", color=C_COLORS[core], lw=1.4,
                    label=C_LABEL[core])
            ax.plot(ff[::28], mean[::28], C_MARKS[core], color=C_COLORS[core],
                    ms=3.4, mec="white", mew=0.35)
        ax.set_xscale("log")
        ax.grid(True, which="both")
        ax.set_ylabel(f"$n_{{\\mathrm{{eff}}}}$ of ${sym}$")
    axes[-1].set_xlabel("$f$ (Hz)")
    axes[0].axhline(2.0, color=GRAY, lw=0.8, ls=":")
    axes[0].text(18500, 1.955, "eddy-current limit $n=2$", fontsize=6.5,
                 color=GRAY, ha="right", va="top")
    axes[1].axhline(0.0, color=GRAY, lw=0.8, ls=":")
    axes[0].set_title("(a)", loc="left", fontsize=9)
    axes[1].set_title("(b)", loc="left", fontsize=9)
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].legend(frameon=False, loc="upper left")
    save(fig, "local_exponent")


def curvature_excess():
    """Curvature of sigma_eff and the part of it that the unshielded
    mapping P -> sigma_eff cannot produce."""
    fig, axes = plt.subplots(2, 1, figsize=(W1, 118 * MM), sharex=True)
    fig.subplots_adjust(left=0.175, right=0.965, top=0.965, bottom=0.09,
                        hspace=0.14)
    for core in ORDER:
        rowsP = [fit_both(F, CORES[core]["P"][:, j])[1] for j in range(len(T))]
        rowsS = [fit_both(F, CORES[core]["S"][:, j])[1] for j in range(len(T))]
        a2s = np.array([q.beta[0] for q in rowsS])
        cis = np.array([q.ci95[0] for q in rowsS])
        d = a2s - np.array([q.beta[0] for q in rowsP])
        axes[0].errorbar(T, a2s, yerr=cis, fmt=C_MARKS[core], ms=3.6,
                         color=C_COLORS[core], mec="white", mew=0.35,
                         lw=1.0, capsize=2.2, label=C_LABEL[core])
        axes[1].plot(T, d, C_MARKS[core] + "-", ms=3.6, lw=1.2,
                     color=C_COLORS[core], mec="white", mew=0.35,
                     label=C_LABEL[core])
    axes[0].set_ylabel("$a_2$ of $\\sigma_{\\mathrm{eff}}$")
    axes[1].set_ylabel("$\\Delta a_2$")
    axes[1].set_xlabel("$T$ ($^\\circ$C)")
    axes[1].axhline(0.0, color=GRAY, lw=0.8, ls=":")
    for ax, letter in zip(axes, "ab"):
        ax.grid(True)
        ax.set_title(f"({letter})", loc="left", fontsize=9)
        ax.set_xlim(10, 130)
    axes[0].set_ylim(-0.02, 0.20)
    axes[0].legend(frameon=False, loc="upper center", ncol=3,
                   handletextpad=0.2, columnspacing=0.7,
                   borderaxespad=0.15)
    save(fig, "curvature_excess")


def flux_profile():
    """Radial profile of |B| across the core at 20 kHz and 120 C.

    Only the nodes inside the core are drawn: the SaveLine output holds
    the elemental field of the core body, which is not defined in the
    surrounding air and is written there as a near-zero placeholder.
    """
    fig, ax = plt.subplots(figsize=(W1, 68 * MM))
    fig.subplots_adjust(left=0.125, right=0.97, top=0.97, bottom=0.15)
    for core in ORDER:
        path = CASE[core] / "line000.dat"
        if not path.exists():
            print(f"  flux_profile: {path} missing, skipping {core}")
            continue
        d = np.loadtxt(path)
        r = d[:, 3] * 1000.0
        b = np.sqrt((d[:, 15:18] ** 2).sum(1) + (d[:, 18:21] ** 2).sum(1))
        sel = (r >= R_IN) & (r <= R_OUT)
        r, b = r[sel], b[sel] / CORES[core]["B"]
        ax.plot(r, b, "-", color=C_COLORS[core], lw=0.8, zorder=2,
                label=C_LABEL[core]
                + f", $B_{{\\max}}={CORES[core]['B']}$ T")
        ax.plot(r, b, C_MARKS[core], color=C_COLORS[core], ms=2.1,
                mec="white", mew=0.2, zorder=3)
    ax.axvspan(13, R_IN, color="#f2f2ef", zorder=0)
    ax.axvspan(24.5, 27, color="#f2f2ef", zorder=0)
    ax.set_xlim(14, 26)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("radial position $r$ (mm)")
    ax.set_ylabel("$|B|\\,/\\,B_{\\max}$")
    ax.grid(True)
    ax.text(14.35, 0.15, "inner\nair", fontsize=6.5, color=GRAY)
    ax.text(24.85, 0.15, "outer\nair", fontsize=6.5, color=GRAY)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "flux_profile")


if __name__ == "__main__":
    power_loss()
    effective_conductivity()
    local_exponent()
    curvature_excess()
    flux_profile()
    print("All figures written to", FIGDIR)
