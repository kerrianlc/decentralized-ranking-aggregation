import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import seaborn as sns

sns.set_style("whitegrid")
colors = sns.color_palette("colorblind")

def plot_convergence(
    results, dataset_name: str, legend: str = "both", **kwargs
):
    from matplotlib.lines import Line2D
 
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 6,
            "legend.title_fontsize": 6.5,
            "lines.linewidth": 1.3,
        }
    )
 
    markers = {"Borda": "o", "Copeland": "s", "Footrule": "^"}
    linestyles = {"Borda": "-", "Copeland": "--", "Footrule": ":"}
 
    colors = sns.color_palette("colorblind")
 
    graph_type_colors = {
        "Complete": colors[0],
        "Watts-Strogatz": colors[1],
        "Geometric": colors[4],
        "2D Grid": colors[2],
        "Cycle": colors[3],
    }
 
    fig_consensus, ax_consensus = plt.subplots(figsize=(2.6, 2.3))
    fig_score_bf, ax_score_bf = plt.subplots(figsize=(2.6, 2.3))
    fig_score_c, ax_score_c = plt.subplots(figsize=(2.6, 2.3))
 
    for method_name, method_results in results.items():
        marker = markers.get(method_name, "o")
        linestyle = linestyles.get(method_name, "-")
 
        for graph_type, result_data in method_results.items():
            color = graph_type_colors[graph_type]
 
            mean_err_scores = result_data[0][0]
            mean_err_consensus = result_data[1][0]
            std_err_consensus = result_data[1][1]
 
            iterations = len(mean_err_scores[:20000])
            style = dict(
                marker=marker,
                markevery=range(iterations // 10, iterations, iterations // 10),
                markersize=4,
                linestyle=linestyle,
                color=color,
            )
 
            ax_score = ax_score_c if method_name == "Copeland" else ax_score_bf
            ax_score.plot(mean_err_scores[:20000], **style)
            ax_consensus.plot(mean_err_consensus[:20000], rasterized=True, **style)
            ax_consensus.fill_between(
                range(iterations),
                mean_err_consensus[:20000] - 2 * std_err_consensus[:20000],
                mean_err_consensus[:20000] + 2 * std_err_consensus[:20000],
                alpha=kwargs.get("alpha", 0.3),
                color=color,
                rasterized=True,
            )
 
    ax_score_bf.set_title("MSE of Scores vs. Iterations (Borda)")
    ax_score_bf.set_xlim(1, iterations)
    ax_score_bf.set_yscale("log")
    ax_score_bf.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
    fig_score_bf.savefig(
        f"fig/{dataset_name}_borda_footrule_score.pdf",
        format="pdf",
        dpi=150,
        bbox_inches="tight",
    )
 
    ax_score_c.set_title("MSE of Scores vs. Iterations (Copeland)")
    ax_score_c.set_xlim(1, iterations)
    ax_score_c.set_yscale("log")
    ax_score_c.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
    fig_score_c.savefig(
        f"fig/{dataset_name}_copeland_score.pdf",
        format="pdf",
        dpi=150,
        bbox_inches="tight",
    )
 
    ax_consensus.set_title("Kendall-$\\tau$ Error vs. Iterations")
    ax_consensus.set_xlim(1, iterations)
    ax_consensus.ticklabel_format(style="scientific", axis="both", scilimits=(0, 0))
 
    color_handles = [
        Line2D([0], [0], color=graph_type_colors[gt], lw=1.3, label=gt)
        for gt in sorted(graph_type_colors.keys())
        if gt in results[list(results.keys())[0]]
    ]

    method_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            lw=1.3,
            marker=markers[method],
            linestyle=linestyles[method],
            markersize=4,
            label=method,
        )
        for method in results.keys()
    ]


    if legend == "both":
        first_legend = ax_consensus.legend(
            handles=color_handles, loc="upper right", title="Graph Type"
        )
        ax_consensus.add_artist(first_legend)
        ax_consensus.legend(handles=method_handles, loc="upper left", title="Method")
    ax_consensus.yaxis.get_major_formatter().get_offset = lambda: "1e0"
    plt.tight_layout()
 
    method_names = "_vs_".join([m.lower().replace(" ", "_") for m in results.keys()])
    fig_consensus.savefig(
        f"fig/{dataset_name}_{method_names}_consensus.pdf",
        format="pdf",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def plot_breakdown(result):
    p = result["params"]
    n, m, t = p["n"], p["m"], p["iterations"]
    n_trials, epsilons = p["n_trials"], p["epsilons"]
    deltas = result["delta_range"]
    styles = {"Borda": ("o", colors[0]), "Copeland": ("s", colors[1])}

    with plt.rc_context({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 11}):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        for name, errs in result["excess_errors"].items():
            marker, color = styles[name]
            for eps, e in zip(epsilons, errs):
                ax.scatter(e, [eps] * n_trials, color=color, alpha=0.25, s=12,
                           linewidths=0, rasterized=True)
            breakdown = [next((eps for eps, e in zip(epsilons, errs) if max(e) >= d), np.nan)
                         for d in deltas]
            ax.plot(deltas, breakdown, color=color, markevery=10,
                    markersize=6, lw=2.5, label=name)

        ax.set(title=f"Kendall-$\\tau$ Error vs. Contamination\n($t={t}$)",
               ylabel="Contamination $\\varepsilon$",
               xlabel="Kendall-$\\tau$ error $\\delta$", xlim=(0, None))
        ax.title.set_fontsize(11)
        ax.set_box_aspect(1)
        ax.margins(y=0.03)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(f"fig/breakdown_n{n}_m{m}_iter{t}.pdf", dpi=300, bbox_inches="tight")
        plt.show()