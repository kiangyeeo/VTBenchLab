#!/usr/bin/env python
"""Plot per-epoch best Top-1 accuracy for matched BN and no-BN runs."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_BN_RUN = (
    WORKSPACE
    / "outputs"
    / "vae_linear_probing_dinov2_single_paperlr_bn"
    / "webssl_mae300m_full2b_224"
)
DEFAULT_NO_BN_RUN = (
    WORKSPACE
    / "outputs"
    / "vae_linear_probing_dinov2_single_paperlr"
    / "webssl_mae300m_full2b_224"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bn-run", type=Path, default=DEFAULT_BN_RUN)
    parser.add_argument("--no-bn-run", type=Path, default=DEFAULT_NO_BN_RUN)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BN_RUN / "bn_vs_no_bn_accuracy_by_epoch.png",
    )
    return parser.parse_args()


def load_accuracy_curve(run_dir):
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    epoch_length = int(protocol["epoch_length_updates"])
    records = [
        json.loads(line)
        for line in (run_dir / "metrics_history.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    records.sort(key=lambda record: int(record["iteration"]))
    epochs = [int(record["iteration"]) / epoch_length for record in records]
    accuracies = [100.0 * float(record["best_classifier"]["accuracy"]) for record in records]
    return epochs, accuracies


def load_font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / filename,
        Path("/usr/share/fonts/dejavu") / filename,
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def centered_text(draw, xy, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def main():
    args = parse_args()
    bn_epochs, bn_accuracy = load_accuracy_curve(args.bn_run)
    no_bn_epochs, no_bn_accuracy = load_accuracy_curve(args.no_bn_run)
    if bn_epochs != no_bn_epochs:
        raise RuntimeError("BN and no-BN runs do not share the same evaluated epochs")

    width, height = 1800, 1100
    image = Image.new("RGB", (width, height), "#FAFBFC")
    draw = ImageDraw.Draw(image)
    colors = {
        "text": "#202631",
        "muted": "#667085",
        "grid": "#DDE3EA",
        "minor_grid": "#EEF1F5",
        "axis": "#8993A4",
        "bn": "#2563EB",
        "no_bn": "#F97316",
    }
    fonts = {
        "title": load_font(44, bold=True),
        "subtitle": load_font(25),
        "axis": load_font(29, bold=True),
        "tick": load_font(25),
        "legend": load_font(28, bold=True),
        "annotation": load_font(28, bold=True),
    }

    plot_left, plot_top, plot_right, plot_bottom = 165, 190, 1660, 900
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    x_min, x_max = min(bn_epochs) - 0.35, max(bn_epochs) + 0.8
    y_min, y_max = 0.0, 65.0

    def x_position(epoch):
        return plot_left + (epoch - x_min) / (x_max - x_min) * plot_width

    def y_position(accuracy):
        return plot_bottom - (accuracy - y_min) / (y_max - y_min) * plot_height

    draw.rounded_rectangle(
        (plot_left, plot_top, plot_right, plot_bottom),
        radius=8,
        fill="#FFFFFF",
        outline="#C7CED8",
        width=2,
    )
    for value in range(5, 66, 5):
        y = y_position(value)
        major = value % 10 == 0
        draw.line(
            (plot_left, y, plot_right, y),
            fill=colors["grid"] if major else colors["minor_grid"],
            width=2 if major else 1,
        )
        if major:
            text = f"{value}%"
            box = draw.textbbox((0, 0), text, font=fonts["tick"])
            draw.text(
                (plot_left - (box[2] - box[0]) - 18, y - (box[3] - box[1]) / 2),
                text,
                font=fonts["tick"],
                fill=colors["muted"],
            )
    for epoch in bn_epochs:
        x = x_position(epoch)
        draw.line((x, plot_top, x, plot_bottom), fill="#F3F5F8", width=1)
        centered_text(
            draw,
            (x, plot_bottom + 42),
            f"{epoch:g}",
            fonts["tick"],
            colors["muted"],
        )

    title = "WebSSL MAE-300M Linear Probe: BatchNorm vs. No BatchNorm"
    centered_text(draw, (width / 2, 60), title, fonts["title"], colors["text"])
    centered_text(
        draw,
        (width / 2, 112),
        "Best validation Top-1 classifier at each epoch · ImageNet-1K",
        fonts["subtitle"],
        colors["muted"],
    )
    centered_text(
        draw,
        ((plot_left + plot_right) / 2, 1025),
        "Epoch",
        fonts["axis"],
        colors["text"],
    )

    y_label = Image.new("RGBA", (300, 60), (0, 0, 0, 0))
    y_draw = ImageDraw.Draw(y_label)
    centered_text(y_draw, (150, 30), "Top-1 Accuracy", fonts["axis"], colors["text"])
    y_label = y_label.rotate(90, expand=True)
    image.paste(y_label, (27, int((plot_top + plot_bottom - y_label.height) / 2)), y_label)

    def draw_curve(epochs, accuracies, color, marker):
        points = [(x_position(epoch), y_position(value)) for epoch, value in zip(epochs, accuracies)]
        draw.line(points, fill=color, width=7, joint="curve")
        for x, y in points:
            radius = 9
            if marker == "circle":
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=color,
                    outline="#FFFFFF",
                    width=3,
                )
            else:
                draw.rounded_rectangle(
                    (x - radius, y - radius, x + radius, y + radius),
                    radius=3,
                    fill=color,
                    outline="#FFFFFF",
                    width=3,
                )

    draw_curve(bn_epochs, bn_accuracy, colors["bn"], "circle")
    draw_curve(no_bn_epochs, no_bn_accuracy, colors["no_bn"], "square")

    legend_left, legend_top, legend_right, legend_bottom = 1130, 720, 1615, 865
    draw.rounded_rectangle(
        (legend_left, legend_top, legend_right, legend_bottom),
        radius=10,
        fill="#FFFFFF",
        outline="#D3D9E2",
        width=2,
    )
    for index, (label, color, marker) in enumerate(
        (("With BatchNorm", colors["bn"], "circle"), ("Without BatchNorm", colors["no_bn"], "square"))
    ):
        y = legend_top + 43 + index * 62
        draw.line((legend_left + 27, y, legend_left + 87, y), fill=color, width=7)
        if marker == "circle":
            draw.ellipse((legend_left + 48, y - 9, legend_left + 66, y + 9), fill=color)
        else:
            draw.rectangle((legend_left + 48, y - 9, legend_left + 66, y + 9), fill=color)
        draw.text(
            (legend_left + 108, y - 17),
            label,
            font=fonts["legend"],
            fill=colors["text"],
        )

    for accuracy, color, y_offset in (
        (bn_accuracy[-1], colors["bn"], -42),
        (no_bn_accuracy[-1], colors["no_bn"], 17),
    ):
        draw.text(
            (x_position(bn_epochs[-1]) + 20, y_position(accuracy) + y_offset),
            f"{accuracy:.2f}%",
            font=fonts["annotation"],
            fill=color,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, quality=95)
    print(args.output)


if __name__ == "__main__":
    main()
