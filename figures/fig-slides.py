import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import pandas as pd
import seaborn as sns



def plot_as_if_optimization(df_quantities_mean, uncertainty=True):

    fig, ax = plt.subplots()
    df_ = df_quantities_mean["subsidy_effect_on_years_by_type"]["difference"]

    df_.plot(kind="bar", rot=0, ylim=(0, 2.2), alpha=0.7, label="Point estimate", ax=ax)

    if uncertainty:
        errors = [
            df_.values - [0.18, 0.1, 0, 0],
            [0.75, 1.35, 1.237, 1.1748] - df_.values,
        ]
        plt.errorbar(
            x=df_.index,
            y=df_.values,
            yerr=errors,
            fmt="none",
            ecolor="k",
            capsize=8,
            elinewidth=2.5,
            label="Confidence interval",
        )
        suffix = "-uq"
    else:
        suffix = ""

    ax.set_xticklabels(["Type 1", "Type 2", "Type 3", "Type 4"])

    ax.set_ylabel(r"$\Delta$ Schooling")
    ax.set_xlabel("")
    ax.set_ylim([0, 2])
    ax.yaxis.set_ticks([0.5, 1.0, 1.5, 2.0])
    ax.legend()

    plt.savefig(f"fig-as-if-optimization{suffix}")


def plot_illustration_complexity():
    def increasing(x, a):
        return a * x ** 2

    def decreasing(x, a, n):
        a = 1
        n = 0.4
        return 1 - (x / a) ** n

    x = np.linspace(0, 1, 1000)

    y_incr = np.apply_along_axis(increasing, 0, x, *[0.5])
    y_decr = np.apply_along_axis(decreasing, 0, x, *[1, 0.4])

    for ext_ in ["teaser", "full"]:

        fig, ax = plt.subplots()

        ax.plot(x, y_decr, label="Model features")
        if ext_ == "full":
            ax.plot(x, y_incr, label="Uncertainty propagation")
            ax.plot(x, y_decr + y_incr, label="Model error", linestyle="--")

        ax.set_ylim(-0.1, 1)
        ax.legend()
        ax.set_xlabel("Model complexity")
        ax.set_ylabel("Model error")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        fig.savefig(f"fig-illustration-complexity-{ext_}")


def clean_results(df):
    """Rename indices and prepare data for plotting."""

    df = df.reset_index()

    df.Policy = df.Policy.replace(
        {
            "subsidy_effect_on_years_for_type_0": "Type 1",
            "subsidy_effect_on_years_for_type_1": "Type 2",
            "subsidy_effect_on_years_for_type_2": "Type 3",
            "subsidy_effect_on_years_for_type_3": "Type 4",
        }
    )

    df.Criterion = df.Criterion.replace(
        {
            "subjective_bayes": "Subjective Bayes",
            "minimax_regret": "Minimax regret",
            "maximin": "Maximin",
            "as_if": "As-if",
        }
    )

    df = df.set_index(["Criterion", "Policy"])
    df.Rank = np.where(df.Value == 0, 0, df.Rank)
    df = df.reindex(["As-if", "Subjective Bayes", "Minimax regret", "Maximin"], level=0)

    return df.Rank.unstack()


def plot_ranks(df):

    """Create rank plot."""

    colors = ["tab:cyan", "tab:olive", "tab:red", "tab:purple"]

    fig, ax = plt.subplots(1)
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    for i, col in enumerate(df.columns):
        kwargs = {
            "marker": "o",
            "linewidth": 3,
            "markersize": 35,
            "color": colors[i],
        }

        df[col].plot(**kwargs)
        # Flip y-axis.
        plt.axis([-0.1, 3.1, 3.2, -0.2])

        plt.yticks(
            [0, 1, 2, 3],
            labels=["Rank 1", "Rank 2", "Rank 3", "Rank 4"],
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
            ncol=4,
        )

    plt.savefig("fig-criterion-policy-ranks")


if __name__ == "__main__":

    df_quantities_mean = pkl.load(open("qoi-at-mean.pickle", "rb"))
    df_decisions = pd.read_pickle("df-decisions-0.100.pkl")
    df_decisions = clean_results(df_decisions)

    sns.set_palette("tab10")
    plot_illustration_complexity()
    plot_ranks(df_decisions)
    for boolean in False, True:
        plot_as_if_optimization(df_quantities_mean, uncertainty=boolean)
