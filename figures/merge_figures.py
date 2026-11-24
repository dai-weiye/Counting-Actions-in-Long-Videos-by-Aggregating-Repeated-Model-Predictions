"""Center independent editable figure slides on a shared PowerPoint canvas at their native print size."""

import argparse
import hashlib
import json
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from lxml import etree


P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
APP = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
NS = {"p": P, "a": A, "r": R}
SCALE = Decimal(12700)
SLIDE_PATH = "ppt/slides/slide1.xml"
SLIDE_RELS = "ppt/slides/_rels/slide1.xml.rels"
STYLE_FILES = (
    "ppt/theme/theme1.xml",
    "ppt/slideMasters/slideMaster1.xml",
    "ppt/slideLayouts/slideLayout7.xml",
)


def xml(data):
    return etree.fromstring(data, parser=etree.XMLParser(resolve_entities=False, no_network=True))


def encode(node):
    return etree.tostring(node, encoding="UTF-8", xml_declaration=True, standalone=True)


def pt(emu):
    return float(Decimal(emu) / SCALE)


def merge(paths, output, crop_path):
    payloads = [p.read_bytes() for p in paths]
    sizes = []
    for data in payloads:
        with ZipFile(BytesIO(data)) as source:
            box = xml(source.read("ppt/presentation.xml")).find("p:sldSz", NS)
            sizes.append((int(box.get("cx")), int(box.get("cy"))))
    canvas_width = max(width for width, _ in sizes)
    canvas_height = max(height for _, height in sizes)
    with ZipFile(BytesIO(payloads[0])) as template:
        parts = {name: template.read(name) for name in template.namelist()}
    pres = xml(parts["ppt/presentation.xml"])
    slide_list = pres.find("p:sldIdLst", NS)
    if len(slide_list) != 1:
        raise ValueError("The template must contain one slide")
    slide_size = pres.find("p:sldSz", NS)
    slide_size.attrib.update({"cx": str(canvas_width), "cy": str(canvas_height), "type": "custom"})
    master_rels = xml(parts["ppt/_rels/presentation.xml.rels"])
    types = xml(parts["[Content_Types].xml"])
    next_rel = 1 + max(int(r.get("Id")[3:]) for r in master_rels)
    first_id = int(slide_list[0].get("id"))
    crop = {
        "canvas_pt": [pt(canvas_width), pt(canvas_height)],
        "coordinate_origin": "top left",
        "rect_format": "left, top, right, bottom in points",
        "pages": [],
    }

    for index, (path, data) in enumerate(zip(paths, payloads), 1):
        with ZipFile(BytesIO(data)) as src:
            current = xml(src.read("ppt/presentation.xml"))
            if len(current.find("p:sldIdLst", NS)) != 1:
                raise ValueError(f"{path.name} must contain a single figure")
            if any(src.read(name) != parts[name] for name in STYLE_FILES):
                raise ValueError("The sources need identical layouts, masters and themes")
            layout_rel = xml(src.read(SLIDE_RELS))
            if (
                len(layout_rel) != 1
                or layout_rel[0].get("Type") != R + "/slideLayout"
                or layout_rel[0].get("Target") != "../slideLayouts/slideLayout7.xml"
            ):
                raise ValueError(f"{path.name} contains slide-specific assets or references")
            size = current.find("p:sldSz", NS)
            width, height = int(size.get("cx")), int(size.get("cy"))
            x, y = (canvas_width - width) // 2, (canvas_height - height) // 2
            page = xml(src.read(SLIDE_PATH))
            tree = page.find("p:cSld/p:spTree", NS)
            for shape in tree:
                kind = etree.QName(shape).localname
                if kind in {"nvGrpSpPr", "grpSpPr"}:
                    continue
                if kind not in {"grpSp", "sp", "cxnSp"}:
                    raise ValueError(f"Expected native geometry or live text, found {kind}")
                properties = "p:grpSpPr" if kind == "grpSp" else "p:spPr"
                origin = shape.find(properties + "/a:xfrm/a:off", NS)
                if origin is None:
                    raise ValueError("A top-level shape has no explicit origin")
                origin.set("x", str(int(origin.get("x")) + x))
                origin.set("y", str(int(origin.get("y")) + y))
            for effect in page.iter(f"{{{A}}}effectRef"):
                effect.set("idx", "0")
            if page.xpath(
                "//a:outerShdw | //a:innerShdw | //a:prstShdw | //p:pic | //p:graphicFrame",
                namespaces=NS,
            ):
                raise ValueError("The slide must contain unshadowed native figure elements only")
            parts[f"ppt/slides/slide{index}.xml"] = encode(page)
            parts[f"ppt/slides/_rels/slide{index}.xml.rels"] = encode(layout_rel)
            crop["pages"].append(
                {
                    "page": index,
                    "figure": path.stem,
                    "source_pptx_sha256": hashlib.sha256(data).hexdigest(),
                    "source_size_pt": [pt(width), pt(height)],
                    "rect_pt": [pt(x), pt(y), pt(x + width), pt(y + height)],
                    "native_shapes": len(page.findall(".//p:sp", NS)),
                    "native_connectors": len(page.findall(".//p:cxnSp", NS)),
                    "native_groups": len(page.findall(".//p:grpSp", NS)),
                    "text_runs": len(page.findall(".//a:t", NS)),
                }
            )
            if index > 1:
                rid = f"rId{next_rel}"
                next_rel += 1
                etree.SubElement(
                    slide_list,
                    f"{{{P}}}sldId",
                    {"id": str(first_id + index - 1), f"{{{R}}}id": rid},
                )
                etree.SubElement(
                    master_rels,
                    f"{{{REL}}}Relationship",
                    Id=rid,
                    Type=R + "/slide",
                    Target=f"slides/slide{index}.xml",
                )
                etree.SubElement(
                    types,
                    f"{{{CT}}}Override",
                    PartName=f"/ppt/slides/slide{index}.xml",
                    ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
                )

    parts["ppt/presentation.xml"] = encode(pres)
    parts["ppt/_rels/presentation.xml.rels"] = encode(master_rels)
    parts["[Content_Types].xml"] = encode(types)
    core = xml(parts["docProps/core.xml"])
    for child in list(core):
        core.remove(child)
    parts["docProps/core.xml"] = encode(core)
    app = etree.Element(f"{{{APP}}}Properties", nsmap={None: APP})
    etree.SubElement(app, f"{{{APP}}}Slides").text = str(len(paths))
    parts["docProps/app.xml"] = encode(app)
    if "docProps/thumbnail.jpeg" in parts:
        del parts["docProps/thumbnail.jpeg"]
        root_rels = xml(parts["_rels/.rels"])
        for rel in list(root_rels):
            if rel.get("Target") == "docProps/thumbnail.jpeg":
                root_rels.remove(rel)
        parts["_rels/.rels"] = encode(root_rels)
    with ZipFile(output, "w") as archive:
        for name, content in sorted(parts.items()):
            member = ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
            member.compress_type = ZIP_DEFLATED
            archive.writestr(member, content)
    crop_path.write_text(json.dumps(crop, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", type=Path, nargs=3, metavar="FIGURE.pptx")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--crop-json", required=True, type=Path)
    args = parser.parse_args()
    merge(args.figures, args.output, args.crop_json)
