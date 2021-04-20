from scipy.stats import norm
from scipy.stats import lognorm
from matplotlib.patches import ConnectionPatch
import seaborn as sns
import matplotlib.patches as mpatches
import pandas as pd

import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np

plt.rc("text", usetex=True)

matplotlib.rc("text", usetex=True)
matplotlib.rc("font", **{"family": "sans-serif"})
params = {"text.latex.preamble": [r"\usepackage{amsmath}"]}
pyplot.rcParams.update(params)


SIGMA, MU = 0.75, 3
A = -3


def model_nonlinear(x):
    return np.exp(x)


def model_linear(x):
    return A * x + B


B = model_nonlinear(MU) - A * MU

OUTPUT_DIST_NONLINEAR = lognorm(s=SIGMA, scale=model_nonlinear(MU))
OUTPUT_DIST_LINEAR = norm(A * MU + B, A ** 2 * SIGMA)
INPUT_DIST = norm(MU, SIGMA)


def plot_transformation(palette):

    if palette == "tab10":
        color = "tab:blue"
    else:
        color = "k"

    grid_norm = np.linspace(-50, 50, 10000)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.axis("off")

    # Distribution of input parameter
    ax3.plot(INPUT_DIST.pdf(grid_norm), grid_norm)
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_ylim([-0, 6])
    ax3.set_xlim([0, 0.7])
    ax3.set_xlabel(r"Density")
    ax3.set_ylabel(r"$\theta$", rotation=0)
    ax3.text(0.35, 1.8, r"$f_{\hat{\theta}}$", color=color)
    ax3.invert_yaxis()

    xyA = (0, MU)
    xyB = (model_nonlinear(MU), MU)
    con = ConnectionPatch(
        xyA=xyA,
        xyB=xyB,
        coordsA="data",
        axesA=ax3,
        axesB=ax4,
        ls="--",
        lw=4,
    )
    ax3.annotate(
        r"$\hat{\theta}$",
        xy=(0.0, MU),
        xytext=(0.05, MU - 0.6),
        arrowprops=dict(arrowstyle="->"),
    )
    ax4.add_artist(con)

    # Model
    ax4.plot(model_nonlinear(grid_norm), grid_norm)
    ax4.set_ylim([-0, 6])
    ax4.invert_yaxis()
    ax4.set_xlim([0, 80])
    ax4.legend()
    ax4.text(60, 3.6, r"$\mathcal{M}_1(\theta)$", color=color)
    ax4.set_xlabel(r"$y_1$")
    ax4.set_yticks([])
    ax4.set_xticks([])

    xyA = (model_nonlinear(MU), MU)
    xyB = (model_nonlinear(MU), 0)

    con = ConnectionPatch(
        xyA=xyA,
        xyB=xyB,
        coordsA="data",
        axesA=ax4,
        axesB=ax2,
        ls="--",
        lw=4,
    )
    ax4.add_artist(con)

    # Distribution of quantitiy of interest
    ax2.plot(
        model_nonlinear(grid_norm),
        OUTPUT_DIST_NONLINEAR.pdf(model_nonlinear(grid_norm)),
    )
    ax2.set_xlim([0, 80])
    ax2.set_ylim([0, None])
    ax2.set_ylabel(r"Density")
    ax2.text(30, 0.018, r"$f_{\hat{y}_1}$", color=color)
    ax2.set_yticks([])
    ax2.set_xticks([])

    ax2.annotate(
        r"$\hat{y}_1$",
        xy=(model_nonlinear(MU), 0.00),
        xytext=(model_nonlinear(MU) - 2.5, 0.0075),
        arrowprops=dict(arrowstyle="->"),
    )

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    fname = "fig-illustration-transformation"
    if palette == "binary_r":
        fname += "-bw"

    fig.savefig(fname)


def plot_distribution_comparison(palette):

    plot_lower, plot_upper = 0, 80

    x = np.linspace(plot_lower, plot_upper, 1000)
    fig, (ax1, ax2) = plt.subplots(2)

    # Nonlinear
    ax1.plot(x, OUTPUT_DIST_NONLINEAR.pdf(x), label=r"$f_{\hat{y}_1}$")

    confi_lower, confi_upper = (
        OUTPUT_DIST_NONLINEAR.ppf(0.05),
        OUTPUT_DIST_NONLINEAR.ppf(0.95),
    )

    kwargs = {
        "ec": "black",
        "color": "grey",
        "alpha": 0.2,
        "label": r"$\mathcal{U}_{y_1}(0.1)$",
    }
    rect = mpatches.Rectangle(
        [confi_lower, 0], confi_upper - confi_lower, 0.01, **kwargs
    )
    ax1.add_patch(rect)

    ax1.set_yticklabels([]), ax1.set_ylabel(r"Density")
    ax1.set_ylim([0, 0.040])

    xlabels = ["", r"$\underline{y}_1$", "", "", "", "", "", "", ""]
    xticks = [0, confi_lower, model_nonlinear(MU), 30, 40, 50, 60, 70, 80]
    ax1.set_xticks(xticks), ax1.set_xticklabels(xlabels)

    # Linear
    ax2.plot(x, OUTPUT_DIST_LINEAR.pdf(x), label=r"$f_{\hat{y}_2}$")

    confi_lower, confi_upper = OUTPUT_DIST_LINEAR.ppf(0.05), OUTPUT_DIST_LINEAR.ppf(
        0.95
    )

    xlabels = [
        0,
        r"$\underline{y}_2$",
        r"$\hat{y}_1, \hat{y}_2$",
        30,
        40,
        50,
        60,
        70,
        80,
    ]
    xticks = [0, confi_lower, model_nonlinear(MU), 30, 40, 50, 60, 70, 80]
    ax2.set_xticks(xticks), ax2.set_xticklabels(xlabels)

    ax2.set_xlabel(r"$y_g$")
    xyA = (model_nonlinear(MU), OUTPUT_DIST_NONLINEAR.pdf(model_nonlinear(MU)))
    xyB = (model_linear(MU), 0)

    kwargs = {"coordsA": "data", "ls": "--", "lw": 5}
    con = ConnectionPatch(xyA=xyA, xyB=xyB, axesA=ax1, axesB=ax2, **kwargs)
    ax2.add_artist(con)

    kwargs = {
        "ec": "black",
        "color": "grey",
        "alpha": 0.2,
        "label": r"$\mathcal{U}_{y_2}(0.1)$",
    }
    rect = mpatches.Rectangle(
        [confi_lower, 0], confi_upper - confi_lower, 0.02, **kwargs
    )
    ax2.add_patch(rect)

    ax2.set_yticklabels([]), ax2.set_ylabel(r"Density")
    ax2.set_yticks([])
    ax2.set_ylim([0, 0.08])

    ax1.set_xlim([0, plot_upper]), ax2.set_xlim([0, plot_upper])
    ax1.set_yticks([])
    ax1.legend(), ax2.legend()

    fname = "fig-illustration-comparison"
    if palette == "binary_r":
        fname += "-bw"

    fig.savefig(fname)


def plot_comparison_regret(palette):
    plot_lower, plot_upper = 1, 5

    confi_lower, confi_upper = INPUT_DIST.ppf(0.05), INPUT_DIST.ppf(0.95)

    kwargs = {"height_ratios": [3, 1]}
    fig, ax1 = plt.subplots()

    xvals = np.linspace(plot_lower, plot_upper, 1000)

    if palette == "binary_r":
        kwargs_1 = {"alpha": 0.2, "label": r"regret choosing $g_1$", "hatch": "--"}
        kwargs_2 = {"alpha": 0.2, "label": r"regret choosing $g_2$", "hatch": ".."}
        ls = "--"

    else:
        kwargs_1 = {"alpha": 0.2, "label": r"regret choosing $g_1$"}
        kwargs_2 = {"alpha": 0.2, "label": r"regret choosing $g_2$"}
        ls = "-"

    ax1.plot(xvals, model_nonlinear(xvals), label=r"$\mathcal{M}_{1}(\theta)$")
    ax1.plot(xvals, model_linear(xvals), label=r"$\mathcal{M}_{2}(\theta)$", ls=ls)
    ax1.legend()

    regret_x = np.linspace(confi_lower, MU, 1000)

    ax1.fill_between(
        regret_x, model_linear(regret_x), model_nonlinear(regret_x), **kwargs_1
    )

    regret_x = np.linspace(MU, confi_upper, 1000)
    ax1.fill_between(
        regret_x, model_linear(regret_x), model_nonlinear(regret_x), **kwargs_2
    )

    xticks = [1, 2, 3, 4, 5]

    ax1.set_ylim([-30, None])

    ax1.set_xlim([plot_lower, plot_upper])

    kwargs = {
        "ec": "black",
        "color": "grey",
        "alpha": 0.2,
        "label": r"$\mathcal{U}(0.1)$",
    }

    rect = mpatches.Rectangle(
        [confi_lower, -30], confi_upper - confi_lower, 30, **kwargs
    )
    ax1.add_patch(rect)

    xlabels = [1, 2, r"$\hat{\theta}$", 4, 5]
    ax1.set_xticks(xticks), ax1.set_xticklabels(xlabels)
    ax1.set_yticks([]), ax1.set_yticklabels([])
    ax1.legend(ncol=2)
    ax1.set_xlabel(r"$\theta$")

    fname = "fig-illustration-comparison-regret"
    if palette == "binary_r":
        fname += "-bw"

    fig.savefig(fname)


def plot_comparison_models(palette):

    if palette != "binary_r":
        colors = ["tab:blue", "tab:orange"]
    else:
        colors = ["dimgray", "darkgray"]
    plot_lower, plot_upper = 1, 5

    for version in range(3):

        fig, ax = plt.subplots()

        xvals = np.linspace(plot_lower, plot_upper, 1000)

        ax.plot(
            xvals,
            model_nonlinear(xvals),
            label=r"$\mathcal{M}_{1}(\theta)$",
            color=colors[0],
        )

        yval = model_nonlinear([3])[0]
        if version > 0:
            ax.vlines(
                x=3,
                ymin=0.001,
                ymax=yval,
                linestyles="dashed",
                label="Point estimate",
                color="k",
            )
            ax.hlines(y=yval, xmin=0, xmax=3, linestyles="dashed", color="k")
        if version > 1:
            ax.plot(
                xvals,
                model_linear(xvals),
                label=r"$\mathcal{M}_{2}(\theta)$",
                color=colors[1],
            )

        ax.legend()

        xticks = [1, 2, 3, 4, 5]
        xlabels = [1, 2, r"$\hat{\theta}$", 4, 5]
        yticks = [10, yval, 30, 40, 50, 60]
        ylabels = [10, r"$\hat{y}_g$", 30, 40, 50, 60]

        ax.set_xticks(xticks), ax.set_xticklabels(xlabels)
        ax.set_yticks(yticks), ax.set_yticklabels(ylabels)

        ax.set_ylim([0.001, 60])
        ax.set_xlim([plot_lower, plot_upper])
        ax.legend(loc="upper left")

        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$y_g$")

        fname = f"fig-illustration-comparison-model-{version + 1}"
        if palette == "binary_r":
            fname += "-bw"

        fig.savefig(fname)


def plot_illustration_complexity(palette):
    def increasing(x, a):
        return a * x ** 2

    def decreasing(x, a, n):
        a = 1
        n = 0.4
        return 1 - (x / a) ** n

    x = np.linspace(0, 1, 1000)

    y_incr = np.apply_along_axis(increasing, 0, x, *[0.5])
    y_decr = np.apply_along_axis(decreasing, 0, x, *[1, 0.4])

    for ext_ in ["teaser-1", "teaser-2", "full"]:

        fig, ax = plt.subplots()

        ax.plot(x, y_decr, label="Model features")

        if ext_ == "teaser-2":
            ax.plot(x, y_incr, label="Uncertainty propagation")

        if ext_ == "full":
            ax.plot(x, y_incr, label="Uncertainty propagation")
            ax.plot(x, y_decr + y_incr, label="Model error", linestyle="--")

        ax.set_ylim(-0.1, 1)
        if ext_ == "teaser-1":
            ax.legend(loc="upper right", bbox_to_anchor=(0.908, 1))
        else:
            ax.legend()

        ax.set_xlabel("Model complexity")
        ax.set_ylabel("Model error")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        fname = f"fig-illustration-complexity-{ext_}"
        if palette == "binary_r":
            fname += "-bw"

        fig.savefig(fname)


def plot_ranks_illustration(palette):

    """Create rank plot illustration."""
    df = pd.DataFrame(
        index=["As-if", "Subjective Bayes", "Minimax regret", "Maximin"],
        columns=["Model 1", "Model 2"],
    )
    df["Model 1"] = [0, 1, 0, 0]
    df["Model 2"] = [0, 0, 1, 1]

    if palette != "binary_r":
        colors = ["tab:blue", "tab:orange"]
        linestyle = ["-", "-", "-", "-"]
    else:
        colors = ["gray", "black"]
        linestyle = ["--", "-"]

    fig, ax = plt.subplots(1)
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    for i, col in enumerate(df.columns):
        kwargs = {
            "marker": "o",
            "linestyle": linestyle[i],
            "linewidth": 5,
            "markersize": 45,
            "color": colors[i],
        }

        df[col].plot(**kwargs)
        # Flip y-axis.
        plt.axis([-0.4, 3.5, 1.25, -0.25])

        plt.yticks(
            [0, 1],
            labels=["Rank 1", "Rank 2"],
        )
        plt.xticks(
            [0, 1, 2, 3],
            labels=["As-if", "Subjective \n Bayes", "Minimax \n regret", "Maximin"],
        )
        plt.xlabel("")
        plt.tick_params(axis="both", color="white", pad=20)
        plt.legend(
            markerscale=0.3,
            labelspacing=0.8,
            handlelength=3,
            bbox_to_anchor=(0.45, 1.2),
            loc="upper center",
            ncol=2,
            labels=[r"$g_{1}$", r"$g_{2}$"],
        )

    fname = "fig-illustration-criterion-policy-ranks"
    if palette == "binary_r":
        fname += "-bw"

    fig.savefig(fname)


for palette in ["tab10", "binary_r"]:

    if palette == "binary_r":

        matplotlib.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
        matplotlib.rcParams["axes.labelsize"] = 30
        matplotlib.rcParams["xtick.labelsize"] = 26
        matplotlib.rcParams["ytick.labelsize"] = 26
        matplotlib.rcParams["legend.fontsize"] = 22

    sns.set_palette(palette)

    plot_distribution_comparison(palette)
    plot_comparison_regret(palette)
    plot_comparison_models(palette)
    plot_transformation(palette)
    plot_illustration_complexity(palette)
    plot_ranks_illustration(palette)
