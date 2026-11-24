"""Make the paper's two quantitative figures from the anonymous reproduction package."""

import argparse
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree as ET

from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


INK = "#28313C"
GRAY = "#A5B1BC"
SKY = "#478BCA"
AMETHYST = "#6E5BA3"
SIGNAL_RED = "#EB2124"
WHITE = "#FFFFFF"
WIDTH = 246


class Chart:
    """Keep SVG and PDF labels on the same typographic baseline, measured in points."""

    def __init__(self, out, name, height):
        self.path = out / name
        self.height = height
        self.svg = ET.Element("svg", {"xmlns": "http://www.w3.org/2000/svg", "width": f"{WIDTH}pt",
                                      "height": f"{height}pt", "viewBox": f"0 0 {WIDTH} {height}"})
        self.pdf = canvas.Canvas(str(self.path.with_suffix(".pdf")), pagesize=(WIDTH, height), invariant=1)
        self.pdf.setTitle(name.replace("_", " "))
        self.pdf.setAuthor("")
        self.rect(0, 0, WIDTH, height, WHITE, "background")

    def rect(self, x, top, width, height, fill, label):
        ET.SubElement(self.svg, "rect", {"id": label, "x": str(x), "y": str(top), "width": str(width),
                                        "height": str(height), "fill": fill})
        self.pdf.setFillColor(HexColor(fill))
        self.pdf.rect(x, self.height - top - height, width, height, fill=1, stroke=0)

    def line(self, x1, y1, x2, y2, colour, weight, label):
        ET.SubElement(self.svg, "line", {"id": label, "x1": str(x1), "y1": str(y1), "x2": str(x2),
                                        "y2": str(y2), "stroke": colour, "stroke-width": str(weight)})
        self.pdf.setStrokeColor(HexColor(colour))
        self.pdf.setLineWidth(weight)
        self.pdf.line(x1, self.height - y1, x2, self.height - y2)

    def dot(self, x, y, r, fill, edge, weight, label):
        ET.SubElement(self.svg, "circle", {"id": label, "cx": str(x), "cy": str(y), "r": str(r),
                                          "fill": fill, "stroke": edge, "stroke-width": str(weight)})
        self.pdf.setStrokeColor(HexColor(edge))
        self.pdf.setFillColor(HexColor(fill))
        self.pdf.setLineWidth(weight)
        self.pdf.circle(x, self.height - y, r, stroke=1, fill=1)

    def text(self, x, y, text, label, size=7.6, anchor="start", colour=INK, bold=False):
        weight = "bold" if bold else "normal"
        node = ET.SubElement(self.svg, "text", {"id": label, "x": str(x), "y": str(y), "fill": colour,
                                               "font-family": "Arial", "font-size": str(size),
                                               "font-weight": weight, "text-anchor": anchor})
        node.text = text
        self.pdf.setFont("PlotArialBold" if bold else "PlotArial", size)
        self.pdf.setFillColor(HexColor(colour))
        draw = {"start": self.pdf.drawString, "middle": self.pdf.drawCentredString, "end": self.pdf.drawRightString}
        draw[anchor](x, self.height - y, text)

    def axis(self, left, right, y, ticks, title):
        self.line(left, y, right, y, GRAY, 0.65, "axis")
        for n in ticks:
            x = left + (right - left) * (n - ticks[0]) / (ticks[-1] - ticks[0])
            self.line(x, y, x, y + 3, GRAY, 0.65, f"tick-{n}")
            self.text(x, y + 11.5, str(n), f"tick-label-{n}", size=7.25, anchor="middle")
        self.text(3, y + 11.5, title, "axis-unit", size=7.5)

    def save(self):
        ET.indent(self.svg)
        ET.ElementTree(self.svg).write(self.path.with_suffix(".svg"), encoding="utf-8", xml_declaration=True)
        self.pdf.showPage()
        self.pdf.save()


def sampling_chart(out, data):
    chart = Chart(out, "sampling_control", 104)
    left, right, axis_y = 79, 235, 85.5
    chart.dot(81.5, 9, 2.5, WHITE, SKY, 0.95, "legend-best-symbol")
    chart.text(87.6, 11.5, "Best single call", "legend-best", size=7.4)
    chart.dot(151.5, 9, 2.5, AMETHYST, AMETHYST, 0.85, "legend-four-symbol")
    chart.text(157.6, 11.5, "Four-report estimate", "legend-four", size=7.4)
    for group, name, y in (("primary", "3 uniform + change", 42), ("random", "4 random subsets", 70)):
        x = lambda v: left + (right - left) * (v - 60) / 20
        first, last = data[group + "_best"], data[group + "_aggregate"]
        chart.text(3, y + 2.7, name, group + "-label", size=7.75)
        chart.line(x(first), y, x(last), y, GRAY, 1.25, group + "-line")
        chart.dot(x(first), y, 2.75, WHITE, SKY, 1, group + "-best")
        chart.dot(x(last), y, 2.85, AMETHYST, AMETHYST, 0.8, group + "-four")
        chart.text(x(first), y - 6.6, f"{first:.1f}", group + "-best-value", anchor="middle", size=7.8)
        chart.text(x(last), y - 6.6, f"{last:.1f}", group + "-four-value", anchor="middle", size=7.8, colour=INK, bold=True)
    chart.axis(left, right, axis_y, [60, 65, 70, 75, 80], "Accuracy (%)")
    chart.save()


def errors_chart(out, counts):
    chart = Chart(out, "count_errors", 116)
    left, right, axis_y = 79, 235, 100
    for tag, heading, y in (("a", "Changes in correctness", 10), ("b", "Final error direction", 59)):
        chart.text(3, y, tag, tag + "-letter", size=8.2, bold=True)
        chart.text(13, y, heading, tag + "-label", size=7.7, bold=True)
    bars = (("corrected", "Corrected", 23), ("introduced", "New errors", 38),
            ("full_under", "Undercount", 73), ("full_over", "Overcount", 88))
    bar_colours = {"corrected": AMETHYST, "introduced": SIGNAL_RED, "full_under": SKY, "full_over": SKY}
    for key, label, y in bars:
        n = counts[key]
        x = left + (right - left) * n / 20
        chart.text(13, y + 2.65, label, key + "-label", size=7.75)
        chart.rect(left, y - 2.1, x - left, 4.2, bar_colours[key], key + "-bar")
        chart.text(x + 4.0, y + 2.65, str(n), key + "-count", size=7.8)
    chart.axis(left, right, axis_y, [0, 5, 10, 15, 20], "Questions")
    chart.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    fonts = Path(__file__).resolve().parent / "fonts"
    parser.add_argument("--arial", type=Path, default=fonts / "Arimo-Regular.ttf",
                        help="regular Arial-compatible font used for the direct PDF preview")
    parser.add_argument("--arial-bold", type=Path, default=fonts / "Arimo-Bold.ttf",
                        help="bold Arial-compatible font used for the direct PDF preview")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont("PlotArial", str(args.arial)))
    pdfmetrics.registerFont(TTFont("PlotArialBold", str(args.arial_bold)))
    spec = importlib.util.spec_from_file_location("archived_replay", args.package / "reproduce.py")
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    sampling = archived.reproduce()["figure_data"]["sampling_control"]
    errors = json.loads((args.package / "held_out_errors/results.json").read_text())["figure_counts"]
    sampling_chart(args.out, sampling)
    errors_chart(args.out, errors)
    metadata = {
        "sampling_control": {"size_pt": [246, 104], "source": "reproduce.py:figure_data.sampling_control",
                             "accuracy_pct": sampling, "matched_development_questions": 200,
                             "markers": {"open": "best single call", "filled": "four-report estimate"}},
        "count_errors": {"size_pt": [246, 116], "source": "held_out_errors/results.json:figure_counts",
                         "counts": errors, "unit": "held-out development questions"},
        "palette": {"ink": INK, "axis": GRAY, "comparison": SKY, "reference_or_correction": AMETHYST,
                    "introduced_error": SIGNAL_RED, "background": WHITE},
        "fonts": {"native_powerpoint": "Arial", "exported_pdf": "embedded Arial",
                  "portable_svg": "embedded Arimo (Arial-compatible)"},
        "input_sha256": {p: sha256((args.package / p).read_bytes()).hexdigest()
                         for p in ("data/action_count.csv", "reproduce.py", "held_out_errors/results.json")},
    }
    (args.out / "chart_data.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
