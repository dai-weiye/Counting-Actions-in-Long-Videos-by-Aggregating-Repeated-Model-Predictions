"""Remove template metadata and default Office effects from an editable FigGenie slide."""

import argparse
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from lxml import etree


P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
THEME_FONT_ALIASES = {
    "\u5b8b\u4f53": "SimSun",
    "\u65b0\u7d30\u660e\u9ad4": "PMingLiU",
}


def english_theme_font_names(data):
    """Use the English family name of the same Office fallback font."""
    for localized, english in THEME_FONT_ALIASES.items():
        data = data.replace(
            f'typeface="{localized}"'.encode("utf-8"),
            f'typeface="{english}"'.encode("ascii"),
        )
    return data


def encode(node):
    return etree.tostring(node, encoding="UTF-8", xml_declaration=True, standalone=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Choose separate input and output files")
    with ZipFile(args.source) as source, ZipFile(args.output, "w") as target:
        for filename in sorted(source.namelist()):
            if filename == "docProps/thumbnail.jpeg":
                continue
            data = source.read(filename)
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", filename):
                root = etree.fromstring(data)
                for props in list(root.iter(f"{{{P}}}spPr")) + list(root.iter(f"{{{P}}}grpSpPr")):
                    for effect in list(props):
                        if effect.tag in (f"{{{A}}}effectLst", f"{{{A}}}effectDag"):
                            props.remove(effect)
                    etree.SubElement(props, f"{{{A}}}effectLst")
                for effect in root.iter(f"{{{A}}}effectRef"):
                    effect.set("idx", "0")
                for shape in root.iter(f"{{{P}}}cNvPr"):
                    shape_id = shape.get("id", "")
                    name = re.sub(r"[^\x20-\x7e]+", "", shape.get("name", "")).strip()
                    shape.set("name", name or f"Element {shape_id}")
                data = encode(root)
            elif re.fullmatch(r"ppt/theme/theme\d+\.xml", filename):
                data = english_theme_font_names(data)
            elif filename == "docProps/core.xml":
                root = etree.fromstring(data)
                for child in list(root):
                    root.remove(child)
                data = encode(root)
            elif filename == "_rels/.rels":
                root = etree.fromstring(data)
                for rel in list(root):
                    if rel.get("Target") == "docProps/thumbnail.jpeg":
                        root.remove(rel)
                data = encode(root)
            member = ZipInfo(filename, date_time=(2000, 1, 1, 0, 0, 0))
            member.compress_type = ZIP_DEFLATED
            target.writestr(member, data)


if __name__ == "__main__":
    main()
