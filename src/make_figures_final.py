"""
Figures for the consolidated manuscript (paper/final/main.tex).

Every value below is transcribed from a table in that manuscript and is not recomputed
here: Table 1 (tab:protocol), Table 3 (tab:hargap) and Table 5 (tab:bndial). Keeping the
figures literal rather than re-deriving them means a figure cannot silently disagree with
the table it illustrates. The provenance of the underlying numbers is the script named in
the "% source:" comment above each table.

Style: greyscale fills with hatching and distinct line styles/markers, so every series
remains separable in black and white. Sized for the cas-dc two-column layout --- full
text width for the two-panel figure, single column for the other two.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "final", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# ---- Table 1 (tab:protocol)
SPLITS = [("Random\nover samples", 0.9984), ("Blocked\ninto runs", 0.4431),
          ("Grouped by\nrecording day", 0.5287)]
TARGET_F1 = 0.1626
# ---- Table 3 (tab:hargap)
GAP = [("S$_1\\rightarrow$S$_2$", 0.3497, 0.3434, 0.6),
       ("S$_2\\rightarrow$S$_1$", 0.4897, 0.2767, 21.3)]
# ---- Table 5 (tab:bndial)
LAM = [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
D_FWD = [-5.7, -6.5, -8.6, -23.3, -30.3, -25.9, -33.0]
D_REV = [+4.4, +2.4, +2.1, -12.9, -19.9, -22.8, -20.5]


def fig1():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.45),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    # --- Panel A: two stride-1 windows over 21 timesteps
    n = 21
    ax.set_xlim(-0.8, n + 0.6); ax.set_ylim(-1.35, 1.62)
    # the shared span, drawn behind the boxes
    ax.add_patch(Rectangle((1.09, 0.38), 18.82, 0.54, facecolor="0.62",
                           edgecolor="none", zorder=0))
    for t in range(n):
        ax.add_patch(Rectangle((t + 0.09, 0.45), 0.82, 0.4, facecolor="0.90",
                               edgecolor="0.35", linewidth=0.5, zorder=2))
    ax.text(n / 2, 1.30, "sensor stream, one box per timestep",
            ha="center", va="bottom", fontsize=7.5)
    ax.text(n / 2, 0.97, "shaded: 19 of 20 timesteps shared",
            ha="center", va="bottom", fontsize=7.5)
    # window spans drawn as range bars beneath
    def span(y, x0, x1, ls, lab):
        ax.plot([x0, x1], [y, y], color="black", linewidth=1.2, linestyle=ls,
                solid_capstyle="butt", zorder=3)
        for xe in (x0, x1):
            ax.plot([xe, xe], [y - 0.10, y + 0.10], color="black", linewidth=1.2, zorder=3)
        ax.text((x0 + x1) / 2, y - 0.30, lab, ha="center", va="top", fontsize=7.5)
    span(0.05, 0.09, 19.91, "solid", "window $t$ \u2014 timesteps 1--20")
    span(-0.62, 1.09, 20.91, (0, (3, 1.6)), "window $t+1$ \u2014 timesteps 2--21")
    ax.axis("off")
    ax.set_title("(a) Adjacent stride-1 windows", loc="left", fontsize=8)

    # --- Panel B: source macro-F1 under three splits
    lab = [s for s, _ in SPLITS]; val = [v for _, v in SPLITS]
    hatch = ["", "///", "\\\\\\"]; shade = ["0.30", "0.72", "0.55"]
    xs = np.arange(len(val))
    for i, (x, v) in enumerate(zip(xs, val)):
        bx.bar(x, v, width=0.62, facecolor=shade[i], edgecolor="black",
               linewidth=0.7, hatch=hatch[i])
        bx.text(x, v + 0.022, f"{v:.4f}", ha="center", va="bottom", fontsize=7.5)
    bx.axhline(TARGET_F1, color="black", linestyle=(0, (4, 2)), linewidth=0.9)
    bx.annotate("target macro-F$_1$ = 0.1626\n(unchanged across splits)",
                xy=(2.42, TARGET_F1), xytext=(2.42, 0.86), ha="right", va="top",
                fontsize=7, arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black",
                                            shrinkA=1, shrinkB=1))
    bx.set_xticks(xs); bx.set_xticklabels(lab)
    bx.set_ylabel("source macro-F$_1$"); bx.set_ylim(0, 1.13)
    bx.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bx.set_title("(b) Source fit by validation split", loc="left", fontsize=8)
    fig.savefig(os.path.join(OUT, "fig_leakage.pdf"))
    plt.close(fig); print("  fig_leakage.pdf")


def fig2():
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    xs = np.arange(len(GAP)); w = 0.34
    own = [g[1] for g in GAP]; oth = [g[2] for g in GAP]; pen = [g[3] for g in GAP]
    ax.bar(xs - w / 2, own, w, facecolor="0.32", edgecolor="black", linewidth=0.7,
           label="trained on own days")
    ax.bar(xs + w / 2, oth, w, facecolor="0.80", edgecolor="black", linewidth=0.7,
           hatch="///", label="trained on other building")
    for x, a, b, p in zip(xs, own, oth, pen):
        for xx, v in ((x - w / 2, a), (x + w / 2, b)):
            ax.text(xx, v + 0.009, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
        # guide from the own-days level across to the other-building bar, then a
        # vertical double arrow spanning the drop
        xr = x + w / 2
        ax.plot([x - w / 2, xr + w / 2 + 0.03], [a, a], color="black",
                linewidth=0.6, linestyle=(0, (2, 1.6)), zorder=1)
        xa = xr + w / 2 + 0.10
        ax.annotate("", xy=(xa, a), xytext=(xa, b),
                    arrowprops=dict(arrowstyle="<->", lw=0.7, color="black"))
        ax.text(xa + 0.04, (a + b) / 2, f"{p} pts", ha="left", va="center", fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels([g[0] for g in GAP])
    ax.set_ylabel("target macro-F$_1$"); ax.set_xlim(-0.55, len(GAP) + 0.34)
    ax.set_ylim(0, 0.62)
    ax.legend(frameon=False, loc="upper center", ncol=1, handlelength=1.4,
              borderpad=0.15, bbox_to_anchor=(0.46, 1.02))
    fig.savefig(os.path.join(OUT, "fig_penalty.pdf"))
    plt.close(fig); print("  fig_penalty.pdf")


def fig3():
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    x = np.arange(len(LAM))
    ax.axhline(0, color="0.55", linewidth=0.7)
    ax.plot(x, D_FWD, marker="o", ms=3.6, color="black", linewidth=1.1,
            label="S$_1\\rightarrow$S$_2$ (penalty 0.6 pts)")
    ax.plot(x, D_REV, marker="s", ms=3.4, color="black", linewidth=1.1,
            linestyle=(0, (3.5, 1.8)), markerfacecolor="white",
            label="S$_2\\rightarrow$S$_1$ (penalty 21.3 pts)")
    ipub = LAM.index(0.8)
    ax.axvline(ipub, color="black", linestyle=(0, (1.6, 1.6)), linewidth=0.8)
    ax.text(ipub, 10.3, "published $\\lambda_{BN}=0.8$", fontsize=7,
            ha="center", va="top")
    ax.set_xticks(x); ax.set_xticklabels([f"{v:.1f}" for v in LAM])
    ax.set_xlabel("$\\lambda_{BN}$   (1.0 = source statistics, 0.0 = target statistics)")
    ax.set_ylabel("$\\Delta$ macro-F$_1$ vs source-only (pts)")
    ax.set_ylim(-36, 12)
    ax.legend(frameon=False, loc="lower left", handlelength=2.2, borderpad=0.2)
    fig.savefig(os.path.join(OUT, "fig_bndial.pdf"))
    plt.close(fig); print("  fig_bndial.pdf")


if __name__ == "__main__":
    print("writing figures to paper/final/figures/")
    fig1(); fig2(); fig3()
