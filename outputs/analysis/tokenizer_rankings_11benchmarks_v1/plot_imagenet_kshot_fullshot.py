#!/usr/bin/env python3
"""Plot ImageNet k-shot and DINOv2 10-epoch full-shot scores."""

from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT = Path(__file__).with_name("imagenet_kshot_fullshot_accuracy_10epoch.png")

SHOT_LABELS = ["1-shot", "2-shot", "4-shot", "8-shot", "16-shot", "Full-shot"]
SERIES = {
    "UniTok": [40.102, 52.192, 62.258, 67.864, 71.584, 78.336],
    "VILA-U": [39.104, 53.342, 64.234, 70.138, 73.614, 77.844],
    "TokLIP-S": [47.014, 59.154, 67.722, 72.392, 74.970, 77.720],
    "TokLIP-L": [48.530, 62.488, 70.998, 75.856, 78.466, 80.754],
    "MetaCLIP": [33.172, 45.564, 56.470, 63.524, 68.428, 77.686],
}

COLORS = {
    "UniTok": "#2563EB",
    "VILA-U": "#F97316",
    "TokLIP-S": "#16A34A",
    "TokLIP-L": "#DC2626",
    "MetaCLIP": "#9333EA",
}
MARKERS = {
    "UniTok": "o",
    "VILA-U": "s",
    "TokLIP-S": "^",
    "TokLIP-L": "D",
    "MetaCLIP": "P",
}


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

    fig, ax = plt.subplots(figsize=(11.5, 6.7), facecolor="#F8FAFC")
    ax.set_facecolor("#FFFFFF")
    x = list(range(len(SHOT_LABELS)))

    # Visually separate the full-data regime from the few-shot settings.
    ax.axvspan(4.55, 5.45, color="#E2E8F0", alpha=0.72, zorder=0)
    ax.text(
        5,
        32.1,
        "FULL DATA",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#64748B",
    )

    for model, values in SERIES.items():
        color = COLORS[model]
        marker = MARKERS[model]
        # Solid lines compare k-shot settings; the dashed final segment marks
        # the protocol/regime change from k-shot to full-shot probing.
        ax.plot(
            x[:5],
            values[:5],
            label=model,
            color=color,
            marker=marker,
            markersize=7.5,
            markeredgecolor="white",
            markeredgewidth=1.2,
            linewidth=2.6,
            zorder=3,
        )
        ax.plot(
            x[4:],
            values[4:],
            color=color,
            marker=marker,
            markersize=7.5,
            markeredgecolor="white",
            markeredgewidth=1.2,
            linewidth=2.2,
            linestyle=(0, (4, 2)),
            zorder=3,
        )

    # Label the final scores without crowding the tightly clustered endpoints.
    label_positions = {
        "UniTok": (5.22, 79.4),
        "VILA-U": (5.22, 78.1),
        "TokLIP-S": (5.22, 76.8),
        "TokLIP-L": (5.22, 81.0),
        "MetaCLIP": (5.22, 75.5),
    }
    for model, values in SERIES.items():
        ax.annotate(
            f"{values[-1]:.2f}",
            xy=(x[-1], values[-1]),
            xytext=label_positions[model],
            textcoords="data",
            color=COLORS[model],
            fontsize=9.5,
            fontweight="bold",
            ha="left",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS[model],
                "linewidth": 1.0,
                "alpha": 0.75,
                "shrinkA": 1,
                "shrinkB": 4,
            },
        )

    ax.set_title(
        "ImageNet Linear Probing Accuracy",
        loc="left",
        fontsize=19,
        color="#0F172A",
        pad=18,
    )
    ax.set_ylabel("Top-1 accuracy (%)", fontweight="bold")
    ax.set_xlabel("Training examples per class", fontweight="bold", labelpad=10)
    ax.set_xticks(x, SHOT_LABELS)
    ax.set_xlim(-0.25, 5.58)
    ax.set_ylim(30, 83)
    ax.set_yticks(range(30, 81, 10))
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.9, alpha=0.7)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.4,
    )
    for text in legend.get_texts():
        text.set_color("#334155")

    fig.subplots_adjust(left=0.095, right=0.91, top=0.84, bottom=0.15)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
