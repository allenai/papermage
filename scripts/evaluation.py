import json
from argparse import ArgumentParser
from functools import partial
from typing import Any, Dict, List

import torch
import tqdm
from necessary import necessary
from sklearn.metrics import classification_report

from papermage.magelib import Box, Document, Entity, Image, Metadata, Span
from papermage.parsers.grobid_parser import GrobidClient, GrobidFullParser
from papermage.predictors import (
    BasePredictor,
    HFBIOTaggerPredictor,
    IVILATokenClassificationPredictor,
)

with necessary("datasets"):
    import datasets


FULL_PATH = (
    "/net/nfs2.s2-research/shannons/projects/2108_S2VL_Sampling/s2-paper-sampling-and-annotation/"
    "combined_pipelines/combined-pdfs"
)


def patch_entity_from_json():
    def from_json(cls, entity_json: dict) -> "Entity":
        # the .get(..., None) or [] pattern is to handle the case where the key is present but the value is None
        return cls(
            spans=[Span.from_json(span_json=span_json) for span_json in entity_json.get("spans", None) or []],
            boxes=[Box.from_json(box_json=box_json) for box_json in entity_json.get("boxes", None) or []],
            metadata=Metadata.from_json(entity_json.get("metadata", None) or {}),
        )

    Entity.from_json = classmethod(from_json)


def patch_grobid_parser_xml_coords_to_boxes():
    old_xml_coords_to_boxes = GrobidFullParser._xml_coords_to_boxes

    def _xml_coords_to_boxes(self, coords_attribute: str, page_sizes: dict) -> List[Box]:
        current_page = getattr(self, "__current_page__", None)
        boxes = old_xml_coords_to_boxes(self=self, coords_attribute=coords_attribute, page_sizes=page_sizes)
        if current_page is not None:
            boxes = [box for box in boxes if box.page == current_page]
            for box in boxes:
                # when evaluating on a single page, the page number is always 0
                box.page = 0
        return boxes

    GrobidFullParser._xml_coords_to_boxes = _xml_coords_to_boxes


def patch_grobid_client_process_pdf():
    old_process_pdf = GrobidClient.process_pdf
    cache: Dict[str, Any] = {}

    def process_pdf(self, *args, pdf_file: str, **kwargs):
        if pdf_file not in cache:
            cache[pdf_file] = old_process_pdf(self=self, *args, pdf_file=pdf_file, **kwargs)
        return cache[pdf_file]

    GrobidClient.process_pdf = process_pdf


def run_vila(doc: Document, vila_predictor: BasePredictor) -> Document:
    entities = vila_predictor.predict(doc=doc)
    doc.annotate_layer(entities=entities, name="vila_entities")
    return doc


def run_grobid(doc: Document, grobid_parser: GrobidFullParser) -> Document:
    pdf_path = f"{FULL_PATH}/{doc.metadata.sha}.pdf"
    setattr(grobid_parser, "__current_page__", doc.metadata.page)
    grobid_doc = grobid_parser.parse(doc=doc, input_pdf_path=pdf_path)
    delattr(grobid_parser, "__current_page__")
    return grobid_doc


patch_entity_from_json()
patch_grobid_parser_xml_coords_to_boxes()
patch_grobid_client_process_pdf()

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("mode", choices=["new", "old", "grobid-fast", "grobid-full"])
    args = ap.parse_args()

    dt = datasets.load_dataset("allenai/s2-vl", split="test")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "new":
        vila_predictor = HFBIOTaggerPredictor.from_pretrained(
            model_name_or_path="allenai/vila-roberta-large-s2vl-internal",
            entity_name="tokens",
            context_name="pages",
            device=device,
        )
        run_fn = partial(run_vila, vila_predictor=vila_predictor)
    elif args.mode == "old":
        vila_predictor = IVILATokenClassificationPredictor.from_pretrained(
            "allenai/ivila-row-layoutlm-finetuned-s2vl-v2", device=device
        )
        run_fn = partial(run_vila, vila_predictor=vila_predictor)
    elif args.mode.startswith("grobid"):
        if args.mode == "grobid-full":
            grobid_parser = GrobidFullParser(grobid_server="http://s2-elanding-24.reviz.ai2.in:32771")
        else:
            grobid_parser = GrobidFullParser(grobid_server="http://s2-elanding-24.reviz.ai2.in:32772")
        run_fn = partial(run_grobid, grobid_parser=grobid_parser)
    else:
        raise ValueError(f"Invalid value for `mode`: {args.mode}")

    docs = []
    gold_tokens = []
    pred_tokens = []

    for row in tqdm.tqdm(dt, desc="Predicting", unit="doc"):
        doc = Document.from_json(row["doc"])
        images = [Image.from_base64(image) for image in row["images"]]
        doc.annotate_images(images=images)
        docs.append(doc)
        doc = run_fn(doc=doc)

        gold_tokens.extend(
            e[0].metadata.type if len(e := token._vila_entities) else "null" for token in doc.tokens
        )
        pred_tokens.extend(
            e[0].metadata.label if len(e := token.vila_entities) else "null" for token in doc.tokens
        )

    print(classification_report(y_true=gold_tokens, y_pred=pred_tokens, digits=4))
