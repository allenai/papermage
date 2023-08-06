from argparse import ArgumentParser

import torch
import tqdm
from necessary import necessary
from sklearn.metrics import classification_report

from papermage.magelib import Document, Image
from papermage.predictors import HFBIOTaggerPredictor, IVILATokenClassificationPredictor

with necessary("datasets"):
    import datasets

ap = ArgumentParser()
ap.add_argument("vila", choices=["new", "old"])
args = ap.parse_args()

dt = datasets.load_dataset('allenai/s2-vl', split='test')
# dt = datasets.load_dataset("json", data_dir="/net/nfs2.s2-research/lucas/s2-vl/data/", split="test")
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.vila == "new":
    vila_predictor = HFBIOTaggerPredictor.from_pretrained(
        model_name_or_path="allenai/vila-roberta-large-s2vl-internal",
        entity_name="tokens",
        context_name="pages",
        device=device,
    )
elif args.vila == "old":
    vila_predictor = IVILATokenClassificationPredictor.from_pretrained(
        "allenai/ivila-row-layoutlm-finetuned-s2vl-v2", device=device
    )
else:
    raise ValueError(f"Invalid value for `vila`: {args.vila}")

docs = []

gold_tokens = []
pred_tokens = []

for row in tqdm.tqdm(dt, desc="Predicting", unit="doc"):
    doc = Document.from_json(row["doc"])
    images = [Image.from_base64(image) for image in row["images"]]
    doc.annotate_images(images=images)
    docs.append(doc)

    entities = vila_predictor.predict(doc=doc)
    doc.annotate_entity(entities=entities, field_name="vila_entities")

    gold_tokens.extend(e[0].metadata.type if len(e := token._vila_entities) else "null" for token in doc.tokens)
    pred_tokens.extend(e[0].metadata.label if len(e := token.vila_entities) else "null" for token in doc.tokens)

print(classification_report(y_true=gold_tokens, y_pred=pred_tokens, digits=4))
