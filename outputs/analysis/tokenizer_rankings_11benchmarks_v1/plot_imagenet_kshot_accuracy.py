#!/usr/bin/env python3
"""Plot the five ImageNet k-shot linear-probing series."""

from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT = Path(__file__).with_name("imagenet_kshot_accuracy.png")
SHOT_LABELS = ["1", "2", "4", "8", "16"]
SERIES = {
    "UniTok": [40.102, 52.192, 62.258, 67.864, 71.584],
    "VILA-U": [39.104, 53.342, 64.234, 70.138, 73.614],
    "TokLIP-S": [47.014, 59.154, 67.722, 72.392, 74.970],
    "TokLIP-L": [48.530, 62.488, 70.998, 75.856, 78.466],
    "MetaCLIP": [33.172, 45.564, 56.470, 63.524, 68.428],
}
COLORS = {
    "UniTok": "#2563EB",
    "VILA-U": "#F97316",
    "TokLIP-S": "#16A34A",
    "TokLIP-L": "#DC2626",
    "MetaCLIP": "#9333EA",
}
MARKERS = {"UniTok": "o", "VILA-U": "s", "TokLIP-S": "^", "TokLIP-L": "D", "MetaCLIP": "P"}


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )
    fig, ax = plt.subplots(figsize=(11.2, 6.5), facecolor="#F8FAFC")
    ax.set_facecolor("#FFFFFF")
    x = list(range(len(SHOT_LABELS)))

    for model, values in SERIES.items():
        ax.plot(
            x,
            values,
            label=model,
            color=COLORS[model],
            marker=MARKERS[model],
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.3,
            linewidth=2.8,
            zorder=3,
        )

    endpoint_y = {
        "UniTok": 71.3,
        "VILA-U": 73.3,
        "TokLIP-S": 75.2,
        "TokLIP-L": 78.5,
        "MetaCLIP": 68.2,
    }
    for model, values in SERIES.items():
        ax.annotate(
            f"{values[-1]:.2f}",
            xy=(x[-1], values[-1]),
            xytext=(4.10, endpoint_y[model]),
            textcoords="data",
            color=COLORS[model],
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
        )

    ax.set_title("ImageNet K-shot Linear Probing", loc="left", fontsize=19, color="#0F172A", pad=18)
    ax.set_xlabel("Shots per class", fontweight="bold", labelpad=10)
    ax.set_ylabel("Top-1 accuracy (%)", fontweight="bold")
    ax.set_xticks(x, SHOT_LABELS)
    ax.set_xlim(-0.18, 4.42)
    ax.set_ylim(30, 81)
    ax.set_yticks(range(30, 81, 10))
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.9, alpha=0.72)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.4,
    )
    fig.subplots_adjust(left=0.10, right=0.93, top=0.84, bottom=0.15)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
