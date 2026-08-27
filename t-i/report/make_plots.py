"""Generate the vector T-I result figure from preserved experiment measurements."""

from itertools import pairwise
from pathlib import Path
from statistics import mean, stdev

from reportlab.lib.colors import Color, HexColor, black
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

BLUE = HexColor("#357DBD")
ORANGE = HexColor("#D97706")
GRID = HexColor("#D5D9DE")

# Trailing means over the ten loss values logged in each 1,000-step interval.
LOSS_ITERATIONS = list(range(11, 51))
LOSS_MEAN_1K = [
    0.0532650,
    0.0517434,
    0.0524987,
    0.0516732,
    0.0458485,
    0.0435100,
    0.0423725,
    0.0426608,
    0.0425845,
    0.0379287,
    0.0350711,
    0.0373819,
    0.0361832,
    0.0366789,
    0.0332652,
    0.0300183,
    0.0320367,
    0.0280474,
    0.0269283,
    0.0251427,
    0.0258872,
    0.0253465,
    0.0231366,
    0.0265326,
    0.0273657,
    0.0219461,
    0.0232065,
    0.0205942,
    0.0205329,
    0.0203387,
    0.0188992,
    0.0207438,
    0.0186096,
    0.0188802,
    0.0155505,
    0.0212304,
    0.0173138,
    0.0179537,
    0.0190073,
    0.0182632,
]

DIAGNOSTIC_ITERATIONS = [10, 15, 20, 25, 30, 35, 40, 45, 50]
DIAGNOSTIC_SUCCESS = [10, 30, 20, 10, 15, 20, 30, 25, 20]

# Updated from evaluation-final/summary.json after all three seeds complete.
FINAL_SEEDS = [0, 1, 2]
FINAL_SUCCESS = [12, 13, 8]


def _text(pdf: canvas.Canvas, x: float, y: float, value: str, size: float = 7) -> None:
    pdf.setFillColor(black)
    pdf.setFont("Times-Roman", size)
    pdf.drawCentredString(x, y, value)


def _axes(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
    y_ticks: list[float],
    title: str,
    x_label: str,
    y_label: str,
    tick_format: str = "g",
) -> tuple[float, float, float, float]:
    left, bottom = x + 37, y + 31
    plot_width, plot_height = width - 45, height - 52
    pdf.setFillColor(black)
    pdf.setFont("Times-Bold", 8)
    pdf.drawCentredString(x + width / 2, y + height - 9, title)
    pdf.setFont("Times-Roman", 6.7)
    pdf.drawCentredString(left + plot_width / 2, y + 8, x_label)
    pdf.saveState()
    pdf.translate(x + 7, bottom + plot_height / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, y_label)
    pdf.restoreState()

    for tick in y_ticks:
        tick_y = bottom + (tick - y_min) / (y_max - y_min) * plot_height
        pdf.setStrokeColor(GRID)
        pdf.setLineWidth(0.35)
        pdf.line(left, tick_y, left + plot_width, tick_y)
        pdf.setFont("Times-Roman", 6.3)
        pdf.setFillColor(black)
        pdf.drawRightString(left - 3, tick_y - 2, format(tick, tick_format))
    pdf.setStrokeColor(black)
    pdf.setLineWidth(0.65)
    pdf.line(left, bottom, left, bottom + plot_height)
    pdf.line(left, bottom, left + plot_width, bottom)
    return left, bottom, plot_width, plot_height


def _draw_line(pdf: canvas.Canvas, points: list[tuple[float, float]]) -> None:
    pdf.setStrokeColor(BLUE)
    pdf.setLineWidth(1.5)
    for start, end in pairwise(points):
        pdf.line(start[0], start[1], end[0], end[1])
    pdf.setFillColor(BLUE)
    for px, py in points:
        pdf.circle(px, py, 1.7, fill=1, stroke=0)


def main() -> None:
    output_dir = Path(__file__).parent / "fig"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pusht_ti_results.pdf"

    page_width, page_height = 7.05 * inch, 2.25 * inch
    pdf = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))
    panel_width = page_width / 3

    left, bottom, width, height = _axes(
        pdf,
        0,
        0,
        panel_width,
        page_height,
        0.0,
        0.06,
        [0.00, 0.02, 0.04, 0.06],
        "Training objective",
        "Training iteration (thousands)",
        "1k-step mean loss",
        ".2f",
    )
    loss_points = [
        (
            left + (iteration - 10) / 40 * width,
            bottom + loss / 0.06 * height,
        )
        for iteration, loss in zip(LOSS_ITERATIONS, LOSS_MEAN_1K, strict=True)
    ]
    _draw_line(pdf, loss_points)
    for tick in [10, 20, 30, 40, 50]:
        _text(pdf, left + (tick - 10) / 40 * width, bottom - 9, str(tick), 6.2)

    left, bottom, width, height = _axes(
        pdf,
        panel_width,
        0,
        panel_width,
        page_height,
        0,
        40,
        [0, 10, 20, 30, 40],
        "Checkpoint selection",
        "Training iteration (thousands)",
        "Success over 20 episodes (%)",
    )
    diagnostic_points = [
        (
            left + (iteration - 10) / 40 * width,
            bottom + success / 40 * height,
        )
        for iteration, success in zip(DIAGNOSTIC_ITERATIONS, DIAGNOSTIC_SUCCESS, strict=True)
    ]
    selected_x = left + (15 - 10) / 40 * width
    pdf.setStrokeColor(ORANGE)
    pdf.setDash(3, 2)
    pdf.setLineWidth(1.1)
    pdf.line(selected_x, bottom, selected_x, bottom + height)
    pdf.setDash()
    _draw_line(pdf, diagnostic_points)
    for tick in [10, 20, 30, 40, 50]:
        _text(pdf, left + (tick - 10) / 40 * width, bottom - 9, str(tick), 6.2)
    pdf.setFillColor(ORANGE)
    pdf.setFont("Times-Roman", 6.2)
    pdf.drawString(left + 3, bottom + height - 8, "selected: 15k")

    final_mean = mean(FINAL_SUCCESS)
    final_std = stdev(FINAL_SUCCESS)
    left, bottom, width, height = _axes(
        pdf,
        2 * panel_width,
        0,
        panel_width,
        page_height,
        0,
        30,
        [0, 10, 20, 30],
        "Final fixed-checkpoint evaluation",
        "Environment seed (100 episodes)",
        "Success (%)",
    )
    band_bottom = bottom + max(0, final_mean - final_std) / 30 * height
    band_top = bottom + min(30, final_mean + final_std) / 30 * height
    pdf.setFillColor(Color(BLUE.red, BLUE.green, BLUE.blue, alpha=0.12))
    pdf.rect(left, band_bottom, width, band_top - band_bottom, fill=1, stroke=0)
    mean_y = bottom + final_mean / 30 * height
    pdf.setStrokeColor(ORANGE)
    pdf.setDash(3, 2)
    pdf.setLineWidth(1.1)
    pdf.line(left, mean_y, left + width, mean_y)
    pdf.setDash()
    bar_width = width * 0.16
    for index, (seed, success) in enumerate(zip(FINAL_SEEDS, FINAL_SUCCESS, strict=True)):
        center = left + width * (0.2 + index * 0.3)
        bar_top = bottom + success / 30 * height
        pdf.setFillColor(BLUE)
        pdf.rect(center - bar_width / 2, bottom, bar_width, bar_top - bottom, fill=1, stroke=0)
        _text(pdf, center, bottom - 9, str(seed), 6.2)
        _text(pdf, center, bar_top + 3, f"{success}%", 6.2)
    pdf.setFillColor(ORANGE)
    pdf.setFont("Times-Roman", 6.2)
    pdf.drawString(left + 3, bottom + height - 8, f"mean = {final_mean:.1f}%")

    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    main()
