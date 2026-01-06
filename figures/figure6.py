import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------------------------------------------------
# Matplotlib params
# ----------------------------------------------------------------------
params = {
    # font
    'font.family': 'serif',
    # 'font.serif': 'Times',
    'font.size': 18,
    # axes
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'axes.linewidth': 0.5,
    # ticks
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'xtick.major.width': 0.3,
    'ytick.major.width': 0.3,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    # legend
    'legend.fontsize': 16,
    # tex
    'text.usetex': True,
}

mpl.rcParams.update(params)

# ----------------------------------------------------------------------
# Load saved simulation-level genus losses
# ----------------------------------------------------------------------
df_boxplot_genus = pd.read_csv(
    "/home/simon/hpc_home/projects/coupling_demography_dist/outputs/"
    "cumulative_biomass_loss_forecast_by_genus.csv"
)

genus_names = ["Spruce", "Other needleleaf", "Broadleaf"]

disturbance_panels = [("Natural Disturbance", "Unplanned"),
                      ("Harvest", "Planned")]  # 2 subplots
period_colors = {"Early": "#66c2a5", "Late": "#fc8d62"}
period_order = ["Early", "Late"]

def cohens_d_paired(a, b):
    """Paired Cohen’s d."""
    diff = b - a
    return diff.mean() / diff.std(ddof=1)


# compute Cohen’s d for each disturbance × genus
dvals = {}

for dist_label, _ in disturbance_panels:
    dvals[dist_label] = {}
    for gname in genus_names:

        sub = df_boxplot_genus.query(
            "disturbance == @dist_label and genus == @gname"
        )

        wide = sub.pivot(index="simulation", columns="period", values="cumulative_loss")

        if {"Early", "Late"}.issubset(wide.columns):
            early = wide["Early"].to_numpy()
            late  = wide["Late"].to_numpy()
            dvals[dist_label][gname] = cohens_d_paired(early, late)
        else:
            dvals[dist_label][gname] = np.nan



# ----------------------------------------------------------------------
# 2) Plot + annotate significance
# ----------------------------------------------------------------------
fig_g, axs_g = plt.subplots(1, 2, figsize=(10.5, 5), constrained_layout=True)

for ax, (dist_label, _) in zip(axs_g, disturbance_panels):
    x_base = np.arange(len(genus_names))
    width = 0.3
    positions = []
    box_data = []
    colors_box = []

    # keep track of max y per genus for placing stars
    genus_ymax = {g: -np.inf for g in genus_names}

    for i, gname in enumerate(genus_names):
        for j, period in enumerate(period_order):
            xpos = x_base[i] + (j - 0.5) * width
            subset = df_boxplot_genus.query(
                "disturbance == @dist_label and period == @period and genus == @gname"
            )["cumulative_loss"].values
            if subset.size == 0:
                continue
            positions.append(xpos)
            box_data.append(subset)
            colors_box.append(period_colors[period])

            # update ymax for this genus
            genus_ymax[gname] = max(genus_ymax[gname], subset.max())

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )

    for patch, c in zip(bp["boxes"], colors_box):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    for whisker in bp['whiskers']:
        whisker.set_color('gray')
    for cap in bp['caps']:
        cap.set_color('gray')
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    ax.set_xticks(x_base)
    ax.set_xticklabels(genus_names, rotation=0)
    ax.set_ylabel("Biomass loss [TgC year$^{-1}$]")
    ax.set_title(dist_label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ---- add significance stars per genus ----
    #for i, gname in enumerate(genus_names):
    #    d = dvals[dist_label].get(gname, np.nan)
    #    if np.isfinite(d):
    
            # choose where to place the text
    #        y = genus_ymax[gname] * 1.05
    
    #        ax.text(
    #            x_base[i],
    #            y,
    #            f"d = {d:.2f}",
    #            ha="center",
    #            va="bottom",
    #            fontsize=10,
    #        )


# legend inside first subplot, bottom-left
handles = [
    mpl.lines.Line2D([0], [0], color=period_colors["Early"], lw=6),
    mpl.lines.Line2D([0], [0], color=period_colors["Late"], lw=6),
]
axs_g[0].legend(
    handles,
    ["Early", "Late"],
    loc="lower left",
    frameon=False,
    fontsize=14,
)
subplot_labels = ['(a)', '(b)']
for idx, ax in enumerate(axs_g.flat):
   ax.text(-0.1, 1.05, subplot_labels[idx], transform=ax.transAxes,
           fontsize=18, fontweight='bold', va='bottom')

plt.savefig(
    "/home/simon/glm1/person/besnard/coupling_demography_dist/figs/fig6.png",
    dpi=300
)

