"""Crop a three-page vector PDF export to the original figure sizes recorded by merge_figures.py."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import fitz


def split_pdf(source, ledger, destination):
    with fitz.open(source) as document:
        records = ledger["pages"]
        if len(document) != len(records):
            raise ValueError("The slide export and crop ledger have different page counts")
        for page_index, (page, record) in enumerate(zip(document, records)):
            width, height = ledger["canvas_pt"]
            if abs(page.rect.width - width) > 0.2 or abs(page.rect.height - height) > 0.2:
                raise ValueError("Export the deck at its native slide dimensions")
            left, top, right, bottom = record["rect_pt"]
            figure_width, figure_height = record["source_size_pt"]
            if (
                abs(right - left - figure_width) > 0.001
                or abs(bottom - top - figure_height) > 0.001
            ):
                raise ValueError(f"Crop dimensions disagree for {record['figure']}")
            if record["page"] != page_index + 1:
                raise ValueError("The crop ledger is out of order")
            # Some PDF exporters round the physical page box by a few hundredths of a point.
            page.set_mediabox(fitz.Rect(0, 0, width, height))
            with fitz.open() as cropped:
                target = cropped.new_page(width=figure_width, height=figure_height)
                target.show_pdf_page(
                    target.rect, document, page_index, clip=fitz.Rect(left, top, right, bottom)
                )
                if not target.get_text().strip():
                    raise ValueError(f"{record['figure']} has no selectable text")
                cropped.set_metadata({"title": record["figure"]})
                cropped.save(destination / (record["figure"] + ".pdf"), garbage=4, deflate=True)


def main():
    tools = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export", type=Path, help="Existing PowerPoint or Impress PDF export, one slide per page"
    )
    parser.add_argument("--deck", type=Path, default=tools / "manuscript_figures.pptx")
    parser.add_argument("--crops", type=Path, default=tools / "figure_crops.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--soffice",
        default="soffice",
        help="Impress command for headless export when --export is absent",
    )
    args = parser.parse_args()
    ledger = json.loads(args.crops.read_text(encoding="utf-8"))
    args.out.mkdir(exist_ok=True, parents=True)
    if args.export is not None:
        split_pdf(args.export, ledger, args.out)
    else:
        with tempfile.TemporaryDirectory() as scratch:
            staging = Path(scratch).resolve()
            profile = (staging / "profile").as_uri()
            subprocess.run(
                [
                    args.soffice,
                    f"-env:UserInstallation={profile}",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(staging),
                    str(args.deck.resolve()),
                ],
                check=True,
            )
            split_pdf(staging / (args.deck.stem + ".pdf"), ledger, args.out)


if __name__ == "__main__":
    main()
