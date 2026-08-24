#!/usr/bin/env python
"""Render side-by-side BN/no-BN epoch curves for WebSSL MAE and DINOv1."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from plot_bn_vs_no_bn_epochs import centered_text, load_accuracy_curve, load_font


WORKSPACE = Path(__file__).resolve().parents[2]
BN_ROOT = WORKSPACE / "outputs" / "vae_linear_probing_dinov2_single_paperlr_bn"
NO_BN_ROOT = WORKSPACE / "outputs" / "vae_linear_probing_dinov2_single_paperlr"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=BN_ROOT / "bn_vs_no_bn_webssl_dinov1_by_epoch.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runs = (
        (
            "WebSSL MAE-300M",
            BN_ROOT / "webssl_mae300m_full2b_224",
            NO_BN_ROOT / "webssl_mae300m_full2b_224",
        ),
        (
            "DINOv1 ViT-S/16",
            BN_ROOT / "dinov1_vits16",
            NO_BN_ROOT / "dinov1_vits16",
        ),
    )
    curves = []
    for title, bn_run, no_bn_run in runs:
        bn_epochs, bn_accuracy = load_accuracy_curve(bn_run)
        no_bn_epochs, no_bn_accuracy = load_accuracy_curve(no_bn_run)
        if bn_epochs != no_bn_epochs:
            raise RuntimeError(f"{title}: BN and no-BN evaluated epochs differ")
        curves.append((title, bn_epochs, bn_accuracy, no_bn_accuracy))

    width, height = 2400, 1180
    image = Image.new("RGB", (width, height), "#FAFBFC")
    draw = ImageDraw.Draw(image)
    colors = {
        "text": "#202631",
        "muted": "#667085",
        "grid": "#DDE3EA",
        "minor_grid": "#EEF1F5",
        "border": "#C7CED8",
        "bn": "#2563EB",
        "no_bn": "#F97316",
    }
    fonts = {
        "title": load_font(46, bold=True),
        "subtitle": load_font(25),
        "panel": load_font(34, bold=True),
        "axis": load_font(28, bold=True),
        "tick": load_font(23),
        "legend": load_font(27, bold=True),
        "annotation": load_font(24, bold=True),
        "gain": load_font(25, bold=True),
    }

    centered_text(
        draw,
        (width / 2, 55),
        "Effect of BatchNorm on Linear-Probe Accuracy",
        fonts["title"],
        colors["text"],
    )
    centered_text(
        draw,
        (width / 2, 107),
        "WebSSL MAE-300M vs. DINOv1 ViT-S/16 · best validation Top-1 classifier at each epoch",
        fonts["subtitle"],
        colors["muted"],
    )

    legend_y = 157
    legend_items = (("With BatchNorm", colors["bn"], "circle"), ("Without BatchNorm", colors["no_bn"], "square"))
    legend_starts = (850, 1250)
    for start, (label, color, marker) in zip(legend_starts, legend_items):
        draw.line((start, legend_y, start + 62, legend_y), fill=color, width=7)
        if marker == "circle":
            draw.ellipse((start + 22, legend_y - 9, start + 40, legend_y + 9), fill=color)
        else:
            draw.rectangle((start + 22, legend_y - 9, start + 40, legend_y + 9), fill=color)
        draw.text((start + 80, legend_y - 17), label, font=fonts["legend"], fill=colors["text"])

    panels = ((125, 245, 1145, 945), (1275, 245, 2295, 945))
    y_min, y_max = 0.0, 80.0

    for panel, (title, epochs, bn_accuracy, no_bn_accuracy) in zip(panels, curves):
        left, top, right, bottom = panel
        panel_width = right - left
        panel_height = bottom - top
        x_min, x_max = min(epochs) - 0.35, max(epochs) + 0.75

        def x_position(epoch):
            return left + (epoch - x_min) / (x_max - x_min) * panel_width

        def y_position(accuracy):
            return bottom - (accuracy - y_min) / (y_max - y_min) * panel_height

        draw.rounded_rectangle(
            panel,
            radius=8,
            fill="#FFFFFF",
            outline=colors["border"],
            width=2,
        )
        centered_text(draw, ((left + right) / 2, top - 38), title, fonts["panel"], colors["text"])

        for value in range(5, 81, 5):
            y = y_position(value)
            major = value % 10 == 0
            draw.line(
                (left, y, right, y),
                fill=colors["grid"] if major else colors["minor_grid"],
                width=2 if major else 1,
            )
            if major:
                tick = f"{value}%"
                box = draw.textbbox((0, 0), tick, font=fonts["tick"])
                draw.text(
                    (left - (box[2] - box[0]) - 15, y - (box[3] - box[1]) / 2),
                    tick,
                    font=fonts["tick"],
                    fill=colors["muted"],
                )
        for epoch in epochs:
            x = x_position(epoch)
            draw.line((x, top, x, bottom), fill="#F3F5F8", width=1)
            centered_text(draw, (x, bottom + 37), f"{epoch:g}", fonts["tick"], colors["muted"])

        def draw_curve(accuracies, color, marker):
            points = [(x_position(epoch), y_position(value)) for epoch, value in zip(epochs, accuracies)]
            draw.line(points, fill=color, width=7, joint="curve")
            for x, y in points:
                radius = 8
                bounds = (x - radius, y - radius, x + radius, y + radius)
                if marker == "circle":
                    draw.ellipse(bounds, fill=color, outline="#FFFFFF", width=3)
                else:
                    draw.rounded_rectangle(bounds, radius=2, fill=color, outline="#FFFFFF", width=3)

        draw_curve(bn_accuracy, colors["bn"], "circle")
        draw_curve(no_bn_accuracy, colors["no_bn"], "square")

        for accuracy, color, offset in (
            (bn_accuracy[-1], colors["bn"], -37),
            (no_bn_accuracy[-1], colors["no_bn"], 13),
        ):
            draw.text(
                (x_position(epochs[-1]) + 12, y_position(accuracy) + offset),
                f"{accuracy:.2f}%",
                font=fonts["annotation"],
                fill=color,
            )

        gain = bn_accuracy[-1] - no_bn_accuracy[-1]
        gain_text = f"Epoch 10 BN gain: {gain:+.2f} pp"
        gain_box = draw.textbbox((0, 0), gain_text, font=fonts["gain"])
        gain_width = gain_box[2] - gain_box[0]
        gain_left = left + 34
        gain_top = bottom - 70
        draw.rounded_rectangle(
            (gain_left, gain_top, gain_left + gain_width + 34, gain_top + 46),
            radius=10,
            fill="#F6F8FB",
            outline="#D9DFE8",
            width=2,
        )
        draw.text(
            (gain_left + 17, gain_top + 8),
            gain_text,
            font=fonts["gain"],
            fill=colors["text"],
        )
        centered_text(draw, ((left + right) / 2, 1045), "Epoch", fonts["axis"], colors["text"])

    y_label = Image.new("RGBA", (320, 60), (0, 0, 0, 0))
    y_draw = ImageDraw.Draw(y_label)
    centered_text(y_draw, (160, 30), "Top-1 Accuracy", fonts["axis"], colors["text"])
    y_label = y_label.rotate(90, expand=True)
    image.paste(y_label, (22, int((panels[0][1] + panels[0][3] - y_label.height) / 2)), y_label)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, quality=95)
    print(args.output)


if __name__ == "__main__":
    main()
