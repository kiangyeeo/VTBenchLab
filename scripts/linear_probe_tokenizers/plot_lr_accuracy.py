#!/usr/bin/env python
"""Render final ImageNet linear-probe accuracy against base LR as SVG."""

import argparse
import html
import json
import math
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = WORKSPACE / "outputs" / "vae_linear_probing_dinov2_single_paperlr"
RUNS = (
    ("MetaCLIP", "metaclip_b16_2pt5b"),
    ("TokLIP-S", "toklip_s_semantic_256"),
    ("TokLIP-L", "toklip_l_semantic_384"),
    ("UniTok", "unitok"),
    ("VILA-U", "vilau_7b_256_semantic_penultimate"),
)
COLORS = {
    "MetaCLIP": "#4C78A8",
    "TokLIP-S": "#F58518",
    "TokLIP-L": "#E45756",
    "UniTok": "#54A24B",
    "VILA-U": "#B279A2",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "lr_accuracy_curves_step12500.svg",
    )
    return parser.parse_args()


def load_record(history_path, requested_iteration):
    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"Empty metrics history: {history_path}")
    iteration = requested_iteration if requested_iteration is not None else max(
        record["iteration"] for record in records
    )
    matches = [record for record in records if record["iteration"] == iteration]
    if not matches:
        raise RuntimeError(f"Iteration {iteration} not found in {history_path}")
    return matches[-1]


def star_points(cx, cy, outer_radius=9.0, inner_radius=4.0):
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        radius = outer_radius if index % 2 == 0 else inner_radius
        points.append(f"{cx + radius * math.cos(angle):.2f},{cy + radius * math.sin(angle):.2f}")
    return " ".join(points)


def nice_bounds(values, step):
    lower = math.floor((min(values) - 0.6) / step) * step
    upper = math.ceil((max(values) + 0.6) / step) * step
    return lower, upper


def render_panel(parts, panel, series, x_limits, y_limits, title, x_ticks):
    left, top, width, height = panel
    x_min, x_max = x_limits
    y_min, y_max = y_limits

    def x_position(value):
        fraction = (math.log10(value) - math.log10(x_min)) / (
            math.log10(x_max) - math.log10(x_min)
        )
        return left + fraction * width

    def y_position(value):
        return top + height - (value - y_min) / (y_max - y_min) * height

    parts.append(
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" '
        'fill="#FFFFFF" stroke="#B8C0CC" stroke-width="1" rx="3"/>'
    )
    y_step = 5 if y_max - y_min > 20 else 2
    y_tick = math.ceil(y_min / y_step) * y_step
    while y_tick <= y_max + 1e-9:
        y = y_position(y_tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + width}" y2="{y:.2f}" '
            'stroke="#E5E9EF" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" '
            f'class="tick">{y_tick:g}</text>'
        )
        y_tick += y_step

    for lr in x_ticks:
        if lr < x_min or lr > x_max:
            continue
        x = x_position(lr)
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + height}" '
            'stroke="#F0F2F5" stroke-width="1"/>'
        )
        label_y = top + height + 22
        parts.append(
            f'<text x="{x:.2f}" y="{label_y}" text-anchor="end" '
            f'transform="rotate(-42 {x:.2f} {label_y})" class="x-tick">{lr:g}</text>'
        )

    parts.append(
        f'<text x="{left + width / 2:.2f}" y="{top - 18}" text-anchor="middle" '
        f'class="panel-title">{html.escape(title)}</text>'
    )
    for label, lrs, accuracies, best_index in series:
        visible = [
            (lr, accuracy)
            for lr, accuracy in zip(lrs, accuracies)
            if x_min <= lr <= x_max
        ]
        points = " ".join(
            f"{x_position(lr):.2f},{y_position(accuracy):.2f}" for lr, accuracy in visible
        )
        color = COLORS[label]
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for lr, accuracy in visible:
            x, y = x_position(lr), y_position(accuracy)
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{color}" '
                'stroke="#FFFFFF" stroke-width="1.2">'
                f'<title>{html.escape(label)}: base LR {lr:g}, Top-1 {accuracy:.3f}%</title>'
                '</circle>'
            )
        best_lr = lrs[best_index]
        if x_min <= best_lr <= x_max:
            best_accuracy = accuracies[best_index]
            x, y = x_position(best_lr), y_position(best_accuracy)
            parts.append(
                f'<polygon points="{star_points(x, y)}" fill="{color}" '
                'stroke="#FFFFFF" stroke-width="1.4">'
                f'<title>{html.escape(label)} best: base LR {best_lr:g}, Top-1 {best_accuracy:.3f}%</title>'
                '</polygon>'
            )


def main():
    args = parse_args()
    series = []
    iterations = set()
    for label, directory in RUNS:
        record = load_record(
            args.output_root / directory / "metrics_history.jsonl",
            args.iteration,
        )
        iterations.add(int(record["iteration"]))
        classifiers = sorted(record["classifiers"], key=lambda item: item["base_lr"])
        lrs = [float(item["base_lr"]) for item in classifiers]
        accuracies = [100.0 * float(item["metrics"]["top-1"]) for item in classifiers]
        best_index = max(range(len(accuracies)), key=accuracies.__getitem__)
        series.append((label, lrs, accuracies, best_index))

    if len(iterations) != 1:
        raise RuntimeError(f"Runs use different final iterations: {sorted(iterations)}")
    iteration = iterations.pop()
    reference_lrs = series[0][1]
    if any(lrs != reference_lrs for _label, lrs, _values, _best in series[1:]):
        raise RuntimeError("The five runs do not share one base-LR grid")

    all_values = [value for _label, _lrs, values, _best in series for value in values]
    high_values = [
        value
        for _label, lrs, values, _best in series
        for lr, value in zip(lrs, values)
        if lr >= 0.005
    ]
    full_y = nice_bounds(all_values, 5)
    zoom_y = nice_bounds(high_values, 2)

    width, height = 1800, 860
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<style>'
        'text{font-family:Inter,Arial,sans-serif;fill:#202631}'
        '.title{font-size:28px;font-weight:700}'
        '.subtitle{font-size:16px;fill:#5B6573}'
        '.panel-title{font-size:18px;font-weight:650}'
        '.tick{font-size:14px;fill:#4E5968}'
        '.x-tick{font-size:12px;fill:#4E5968}'
        '.axis-label{font-size:16px;font-weight:600}'
        '.legend{font-size:14px;font-weight:600}'
        '</style>',
        f'<rect width="{width}" height="{height}" fill="#FAFBFC"/>',
        f'<text x="{width / 2}" y="42" text-anchor="middle" class="title">'
        'ImageNet-1K linear probing: Top-1 accuracy vs. learning rate</text>',
        f'<text x="{width / 2}" y="70" text-anchor="middle" class="subtitle">'
        f'Final validation at update {iteration:,} · x-axis is base LR · effective LR = base LR × 4 · stars mark maxima</text>',
    ]

    legend_width = 330
    legend_start = (width - legend_width * len(series)) / 2
    for index, (label, lrs, values, best_index) in enumerate(series):
        x = legend_start + index * legend_width
        y = 108
        color = COLORS[label]
        parts.extend(
            [
                f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                f'<circle cx="{x + 14}" cy="{y}" r="4" fill="{color}" stroke="#FFFFFF" stroke-width="1"/>',
                f'<text x="{x + 38}" y="{y + 5}" class="legend">{html.escape(label)} '
                f'(best {lrs[best_index]:g}, {values[best_index]:.2f}%)</text>',
            ]
        )

    full_panel = (90, 165, 820, 545)
    zoom_panel = (1010, 165, 730, 545)
    render_panel(
        parts,
        full_panel,
        series,
        (reference_lrs[0], reference_lrs[-1]),
        full_y,
        "Full 13-LR grid",
        reference_lrs,
    )
    render_panel(
        parts,
        zoom_panel,
        series,
        (0.005, 0.5),
        zoom_y,
        "High-LR region (zoomed)",
        reference_lrs,
    )
    parts.extend(
        [
            '<text x="900" y="834" text-anchor="middle" class="axis-label">Base learning rate (log scale)</text>',
            '<text x="25" y="438" text-anchor="middle" transform="rotate(-90 25 438)" class="axis-label">'
            'ImageNet-1K validation Top-1 accuracy (%)</text>',
            '</svg>',
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
