"""

@geli-gel, @amanpreet692, @soldni

"""
import json
import os
import re
import warnings
import xml.etree.ElementTree as et
from collections import defaultdict
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from grobid_client.grobid_client import GrobidClient

from papermage.magelib import (
    Document,
    Entity,
    Metadata,
    PagesFieldName,
    RowsFieldName,
    Span,
    TokensFieldName,
)
from papermage.magelib.box import Box
from papermage.parsers.parser import Parser
from papermage.utils.merge import cluster_and_merge_neighbor_spans

REQUIRED_DOCUMENT_FIELDS = [PagesFieldName, RowsFieldName, TokensFieldName]
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


GROBID_VILA_MAP = {
    # title has no coordinates, not recoverable
    "title": ("Title",),
    # author has no coordinates, but they can be recovered from combining 'persName'
    "author": ("Author",),
    # abstract has no coordinates, but they can be recovered from combining 's'
    "abstract": ("Abstract",),
    # keywords has no coordinates, not recoverable
    "keywords": ("Keywords",),
    # sections are easily available. compared to VILA, section number is extracted and
    # reported as metadata.
    "head": ("Section",),
    # paragraph has no coordinates, but they can be recovered from combining 's'
    "p": ("Paragraph",),
    # list has no coordinates, but they can be recovered from combining 'item'
    "list": ("List",),
    "biblStruct": ("Bibliography",),
    "formula": ("Equation",),
    # tables are enclosed into figures, need to check
    "figure": ("Figure",),
    "table": ("Table",),
    # figDesc is caption for both tables and figures
    "figDesc": ("Caption",),
    # header and footers get merged into notes fields, but these are kinda tricky to
    # recover because things like author affiliations are also in the notes fields.
    # also, no coordinates.
    # for footnotes, need to check if this is of type "foot"
    "note": (
        "Header",
        "Footer",
        "Footnote",
    ),
}


def find_contiguous_ones(array):
    # Add a sentinel value at the beginning/end
    array = np.concatenate([[0], array, [0]])

    # Find the indexes where the array changes
    diff_indices = np.where(np.diff(array) != 0)[0] + 1
    # Find the start and end indexes of zero spans
    zero_start_indices = diff_indices[:-1:2]
    zero_end_indices = diff_indices[1::2] - 1

    spans = list(zip(zero_start_indices, zero_end_indices))

    # Exclude the spans with no element
    spans = [(start - 1, end) for start, end in spans if end - start >= 0]

    return spans


class GrobidFullParser(Parser):
    """Grobid parser that uses Grobid python client to hit a running
    Grobid server and convert resulting grobid XML TEI coordinates into
    PaperMage Entities to annotate an existing Document.

    Run a Grobid server (from https://grobid.readthedocs.io/en/latest/Grobid-docker/):
    > docker pull lfoppiano/grobid:0.7.2
    > docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.2
    """

    def __init__(self, check_server: bool = True, **grobid_config: Any):
        self.grobid_config = {
            "grobid_server": "http://localhost:8070",
            "batch_size": 1000,
            "sleep_time": 5,
            "timeout": 60,
            "coordinates": sorted(set((*GROBID_VILA_MAP.keys(), "s", "ref", "body", "item", "persName"))),
            **grobid_config,
        }
        assert "coordinates" in self.grobid_config, "Grobid config must contain 'coordinates' key"

        with NamedTemporaryFile(mode="w", delete=False) as f:
            json.dump(self.grobid_config, f)
            config_path = f.name

        self.client = GrobidClient(config_path=config_path, check_server=check_server)

        os.remove(config_path)

    def parse(  # type: ignore
        self, input_pdf_path: str, doc: Document, xml_out_dir: Optional[str] = None
    ) -> Document:
        assert doc.symbols != ""
        for field in REQUIRED_DOCUMENT_FIELDS:
            assert field in doc.layers

        (_, _, xml) = self.client.process_pdf(
            service="processFulltextDocument",
            pdf_file=input_pdf_path,
            generateIDs=False,
            consolidate_header=False,
            consolidate_citations=False,
            include_raw_citations=False,
            include_raw_affiliations=False,
            tei_coordinates=True,
            segment_sentences=True,
        )
        assert xml is not None, "Grobid returned no XML"

        if xml_out_dir:
            os.makedirs(xml_out_dir, exist_ok=True)
            xmlfile = os.path.join(xml_out_dir, os.path.basename(input_pdf_path).replace(".pdf", ".xml"))
            with open(xmlfile, "w") as f_out:
                f_out.write(xml)

        self._parse_xml_onto_doc(xml=xml, doc=doc)

        for p in getattr(doc, "p", []):
            grobid_text_elems = [s.metadata["grobid_text"] for s in p.s]
            grobid_text = " ".join(filter(lambda text: isinstance(text, str), grobid_text_elems))
            p.metadata["grobid_text"] = grobid_text

        # add vila-like entities
        vila_entities = self._make_vila_groups(doc)
        doc.annotate_layer(entities=vila_entities, name="vila_entities")
        return doc

    def _make_spans_from_boxes(self, doc: Document, entity: Entity) -> List[Span]:
        tokens = [cast(Entity, t) for match in doc.intersect_by_box(entity, "tokens") for t in match.tokens]
        results = cluster_and_merge_neighbor_spans(
            spans=sorted(set(s for t in tokens for s in t.spans), key=lambda x: x.start)
        )
        merged_spans = results.merged
        return merged_spans

    def _make_spans_from_boxes_if_not_found(self, doc: Document, entity: Entity) -> List[Span]:
        spans = [Span(start=s.start, end=s.end) for s in entity.spans]
        if not spans:
            spans = self._make_spans_from_boxes(doc, entity)
        return spans

    def _make_entities_of_type(
        self, doc: Document, entities: List[Entity], entity_type: str, id_offset: int = 0
    ) -> List[Entity]:
        entities = [
            Entity(
                spans=self._make_spans_from_boxes_if_not_found(doc=doc, entity=ent),
                boxes=[Box(l=b.l, t=b.t, w=b.w, h=b.h, page=b.page) for b in ent.boxes],
                metadata=Metadata(**ent.metadata.to_json(), label=entity_type, id=id_offset + i),
            )
            for i, ent in enumerate(entities)
        ]
        return entities

    def _update_reserved_positions(
        self, reserved_positions: np.ndarray, entities: List[Entity]
    ) -> Tuple[List[Entity], np.ndarray]:
        new_entities: List[Entity] = []
        for ent in entities:
            new_spans = []
            for span in ent.spans:
                already_reserved = reserved_positions[span.start : span.end]
                for start, end in find_contiguous_ones(~already_reserved):
                    new_spans.append(Span(start=start + span.start, end=end + span.start))
                reserved_positions[span.start : span.end] = True
            if new_spans:
                new_entities.append(Entity(spans=new_spans, boxes=ent.boxes, metadata=ent.metadata))
        return new_entities, reserved_positions

    def _make_vila_groups(self, doc: Document) -> List[Entity]:
        ents: List[Entity] = []
        reserved_positions = np.zeros(len(doc.symbols), dtype=bool)

        if _ := getattr(doc, "title", []):
            # title has no coordinates, so we can't recover its position!
            pass

        if h := getattr(doc, "author", []):
            h_ = self._make_entities_of_type(doc=doc, entities=h, entity_type="Author", id_offset=len(ents))
            h__, reserved_positions = self._update_reserved_positions(reserved_positions, h_)
            ents.extend(h__)

        if a := getattr(doc, "abstract", []):
            a_ = self._make_entities_of_type(doc=doc, entities=a, entity_type="Abstract", id_offset=len(ents))
            a__, reserved_positions = self._update_reserved_positions(reserved_positions, a_)
            ents.extend(a__)

        if _ := getattr(doc, "keywords", []):
            # keywords has no coordinates, so we can't recover their positions!
            pass

        if s := getattr(doc, "head", []):
            s_ = self._make_entities_of_type(doc=doc, entities=s, entity_type="Section", id_offset=len(ents))
            s__, reserved_positions = self._update_reserved_positions(reserved_positions, s_)
            ents.extend(s__)

        if t := getattr(doc, "list", []):
            t_ = self._make_entities_of_type(doc=doc, entities=t, entity_type="List", id_offset=len(ents))
            t__, reserved_positions = self._update_reserved_positions(reserved_positions, t_)
            ents.extend(t__)

        if b := getattr(doc, "biblStruct", []):
            b_ = self._make_entities_of_type(doc=doc, entities=b, entity_type="Bibliography", id_offset=len(ents))
            b__, reserved_positions = self._update_reserved_positions(reserved_positions, b_)
            ents.extend(b__)

        if e := getattr(doc, "formula", []):
            e_ = self._make_entities_of_type(doc=doc, entities=e, entity_type="Equation", id_offset=len(ents))
            e__, reserved_positions = self._update_reserved_positions(reserved_positions, e_)
            ents.extend(e__)

        if figs := getattr(doc, "figure", []):
            for fig in figs:
                current_boxes = [Box(l=b.l, t=b.t, w=b.w, h=b.h, page=b.page) for b in fig.boxes]

                if "figDesc" in doc.layers:
                    caption_boxes = [b for d in doc.intersect_by_box(fig, "figDesc") for b in d.boxes]
                    current_boxes = [b for b in current_boxes if b not in caption_boxes]

                if "table" in doc.layers:
                    table_boxes = [b for d in doc.intersect_by_box(fig, "table") for b in d.boxes]
                    current_boxes = [b for b in current_boxes if b not in table_boxes]

                if not current_boxes:
                    continue

                new_fig = Entity(
                    spans=self._make_spans_from_boxes(doc, Entity(boxes=current_boxes)),
                    boxes=current_boxes,
                    metadata=Metadata(**fig.metadata.to_json(), label="Figure", id=len(ents)),
                )
                new_figs, reserved_positions = self._update_reserved_positions(reserved_positions, [new_fig])
                ents.extend(new_figs)

        if t := getattr(doc, "table", []):
            t_ = self._make_entities_of_type(doc=doc, entities=t, entity_type="Table", id_offset=len(ents))
            t__, reserved_positions = self._update_reserved_positions(reserved_positions, t_)
            ents.extend(t__)

        if c := getattr(doc, "figDesc", []):
            c_ = self._make_entities_of_type(doc=doc, entities=c, entity_type="Caption", id_offset=len(ents))
            c__, reserved_positions = self._update_reserved_positions(reserved_positions, c_)
            ents.extend(c__)

        if _ := getattr(doc, "note", []):
            # notes have no coordinates, so we can't recover their positions!
            pass

        if p := getattr(doc, "p", []):
            p_ = self._make_entities_of_type(doc=doc, entities=p, entity_type="Paragraph", id_offset=len(ents))
            p__, reserved_positions = self._update_reserved_positions(reserved_positions, p_)
            ents.extend(p__)

        return ents

    def _parse_xml_onto_doc(self, xml: str, doc: Document) -> Document:
        try:
            xml_root = et.fromstring(xml)
        except Exception as e:
            if xml == "[GENERAL] An exception occurred while running Grobid.":
                warnings.warn("Grobid returned an error; check server logs")
                return doc
            raise e

        all_box_groups = self._get_box_groups(xml_root)
        for field, box_groups in all_box_groups.items():
            doc.annotate_layer(name=field, entities=box_groups)

        return doc

    def _xml_coords_to_boxes(self, coords_attribute: str, page_sizes: dict) -> List[Box]:
        coords_list = coords_attribute.split(";")
        boxes = []
        for coords in coords_list:
            if coords == "":
                # this page has no coordinates
                continue
            pg, x, y, w, h = coords.split(",")
            proper_page = int(pg) - 1
            boxes.append(
                Box(l=float(x), t=float(y), w=float(w), h=float(h), page=proper_page).to_relative(
                    *page_sizes[proper_page]
                )
            )
        return boxes

    def _get_box_groups(self, root: et.Element) -> Dict[str, List[Entity]]:
        page_size_root = root.find(".//tei:facsimile", NS)
        assert page_size_root is not None, "No facsimile found in Grobid XML"

        page_size_data = page_size_root.findall(".//tei:surface", NS)
        page_sizes = dict()
        for data in page_size_data:
            page_sizes[int(data.attrib["n"]) - 1] = [float(data.attrib["lrx"]), float(data.attrib["lry"])]

        all_boxes: Dict[str, List[Entity]] = defaultdict(list)

        for field in self.grobid_config["coordinates"]:
            structs = root.findall(f".//tei:{field}", NS)
            for i, struct in enumerate(structs):
                if (coords_str := struct.attrib.get("coords", None)) is None:
                    all_coords = struct.findall(".//*[@coords]")
                    coords_str = ";".join([c.attrib["coords"] for c in all_coords if "coords" in c.attrib])

                if coords_str == "":
                    continue

                if not (boxes := self._xml_coords_to_boxes(coords_str, page_sizes)):
                    # we check if the boxes are empty because sometimes users monkey-patch
                    # _xml_coords_to_boxes to filter by page
                    continue

                metadata_dict: Dict[str, Any] = {
                    f"grobid_{re.sub(r'[^a-zA-Z0-9_]+', '_', k)}": v
                    for k, v in struct.attrib.items()
                    if k != "coords"
                }
                metadata_dict["grobid_order"] = i
                metadata_dict["grobid_text"] = struct.text
                metadata = Metadata.from_json(metadata_dict)
                box_group = Entity(boxes=boxes, metadata=metadata)
                all_boxes[field].append(box_group)

        return all_boxes


if __name__ == "__main__":
    from argparse import ArgumentParser

    from papermage.parsers import PDFPlumberParser

    ap = ArgumentParser()
    ap.add_argument("pdf_path", type=str)
    opts = ap.parse_args()

    doc = PDFPlumberParser().parse(opts.pdf_path)
    doc = GrobidFullParser().parse(opts.pdf_path, doc)

    for p in getattr(doc, "p", []):
        for s in p.s:
            print(s.metadata.grobid_text)
