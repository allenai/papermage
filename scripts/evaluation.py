from argparse import ArgumentParser
from functools import partial
import json
from typing import Any, Optional

import torch
import tqdm
from necessary import necessary
from sklearn.metrics import classification_report

from papermage.magelib import Document, Image, Entity, Span, Box, Metadata
from papermage.predictors.base_predictor import BasePredictor
from papermage.predictors import HFBIOTaggerPredictor, IVILATokenClassificationPredictor
from papermage.parsers.grobid_parser import GrobidFullParser

with necessary("datasets"):
    import datasets


def from_json(cls, entity_json: dict) -> "Entity":
    # the .get(..., None) or [] pattern is to handle the case where the key is present but the value is None
    return cls(
        spans=[Span.from_json(span_json=span_json) for span_json in entity_json.get("spans", None) or []],
        boxes=[Box.from_json(box_json=box_json) for box_json in entity_json.get("boxes", None) or []],
        metadata=Metadata.from_json(entity_json.get("metadata", None) or {}),
    )


def run_vila(doc: Document, vila_predictor: BasePredictor, **kwargs: Any) -> Document:
    entities = vila_predictor.predict(doc=doc)
    doc.annotate_entity(entities=entities, field_name="vila_entities")
    return doc


def run_grobid(doc: Document, pdf_path: str, grobid_parser: GrobidFullParser, **kwargs: Any) -> Document:
    grobid_doc = grobid_parser.parse(doc=doc, input_pdf_path=pdf_path)
    return grobid_doc


Entity.from_json = classmethod(from_json)   # type: ignore


ap = ArgumentParser()
ap.add_argument("mode", choices=["new", "old", "grobid-fast", "grobid-full"])
args = ap.parse_args()

dt = datasets.load_dataset('allenai/s2-vl', split='test')
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
        grobid_parser = GrobidFullParser(grobid_server='http://s2-elanding-24.reviz.ai2.in:32771')
    else:
        grobid_parser = GrobidFullParser(grobid_server='http://s2-elanding-24.reviz.ai2.in:32772')
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

    pdf_path = f"/net/nfs2.s2-research/lucas/s2-vl/raw/pdfs/{doc.metadata.sha}-{doc.metadata.page:02d}.pdf"
    doc = run_fn(doc=doc, pdf_path=pdf_path)

    gold_tokens.extend(e[0].metadata.type if len(e := token._vila_entities) else "null" for token in doc.tokens)
    pred_tokens.extend(e[0].metadata.label if len(e := token.vila_entities) else "null" for token in doc.tokens)

print(classification_report(y_true=gold_tokens, y_pred=pred_tokens, digits=4))
