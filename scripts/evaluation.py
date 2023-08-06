from necessary import necessary
import torch
import tqdm
from papermage.magelib import Document, Image
from papermage.predictors import HFBIOTaggerPredictor, LPBlockPredictor
from sklearn.metrics import precision_recall_fscore_support

with necessary("datasets"):
    import datasets


dt = datasets.load_dataset('allenai/s2-vl', split='test')
vila_predictor = HFBIOTaggerPredictor.from_pretrained(
    model_name_or_path="allenai/vila-roberta-large-s2vl-internal",
    entity_name="tokens",
    context_name="pages",
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

docs = []

p_, r_, f1_ = [], [], []

for row in tqdm.tqdm(dt):
    doc = Document.from_json(row['doc'])
    images = [Image.from_base64(image) for image in row['images']]
    doc.annotate_images(images=images)
    docs.append(doc)

    entities = vila_predictor.predict(doc=doc)
    doc.annotate_entity(entities=entities, field_name='vila_entities')

    gold_tokens = [e[0].metadata.type if len(e := token._vila_entities) else 'null' for token in doc.tokens]
    pred_tokens = [e[0].metadata.label if len(e := token.vila_entities) else 'null' for token in doc.tokens]

    p, r, f1, _ = precision_recall_fscore_support(gold_tokens, pred_tokens, average='macro', zero_division=0)
    p_.append(p)
    r_.append(r)
    f1_.append(f1)


print(f'p={sum(p_) / len(p_):.4f}; r={sum(r_) / len(r_):.4f}; f1={sum(f1_) / len(f1_):.4f}')
# print(squad_metric.compute())
