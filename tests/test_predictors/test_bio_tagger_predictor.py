"""

@kylel, benjaminn

"""

import json
import pathlib
import unittest

import transformers

from papermage.magelib import Document, Entity, Span
from papermage.predictors.base_predictors.hf_predictors import (
    BIOBatch,
    BIOPrediction,
    HFBIOTaggerPredictor,
)

TEST_SCIBERT_WEIGHTS = "allenai/scibert_scivocab_uncased"


class TestBioTaggerPredictor(unittest.TestCase):
    def setUp(self):
        transformers.set_seed(407)
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"

        # setup document
        with open(self.fixture_path / "entity_classification_predictor_test_doc_papermage.json", "r") as f:
            test_doc_json = json.load(f)
        self.doc = Document.from_json(doc_json=test_doc_json)
        ent1 = Entity(spans=[Span(start=86, end=456)])
        ent2 = Entity(spans=[Span(start=457, end=641)])
        self.doc.annotate_layer(name="bibs", entities=[ent1, ent2])

        # setup predictor
        self.id2label = {0: "O", 1: "B_Label", 2: "I_Label"}
        self.label2id = {label: id_ for id_, label in self.id2label.items()}
        self.predictor = HFBIOTaggerPredictor.from_pretrained(
            model_name_or_path=TEST_SCIBERT_WEIGHTS,
            entity_name="tokens",
            context_name="pages",
            **{"num_labels": len(self.id2label), "id2label": self.id2label, "label2id": self.label2id},
        )

    def test_preprocess(self):
        doc = Document(symbols="This is a test document.")
        tokens = [
            Entity(spans=[Span(start=0, end=4)]),
            Entity(spans=[Span(start=5, end=7)]),
            Entity(spans=[Span(start=8, end=9)]),
            Entity(spans=[Span(start=10, end=14)]),
            Entity(spans=[Span(start=15, end=23)]),
            Entity(spans=[Span(start=23, end=24)]),
        ]
        doc.annotate_layer(name="tokens", entities=tokens)
        sents = [Entity(spans=[Span(start=0, end=24)])]
        doc.annotate_layer(name="sents", entities=sents)

        batches = self.predictor.preprocess(doc=doc, context_name="sents")
        self.assertIsInstance(batches[0], BIOBatch)
        decoded_batch = self.predictor.tokenizer.batch_decode(batches[0].input_ids)
        self.assertListEqual(decoded_batch, ["[CLS] this is a test document. [SEP]"])

    def test_predict_pages_tokens(self):
        predictor = HFBIOTaggerPredictor.from_pretrained(
            model_name_or_path=TEST_SCIBERT_WEIGHTS,
            entity_name="tokens",
            context_name="pages",
            **{"num_labels": len(self.id2label), "id2label": self.id2label, "label2id": self.label2id},
        )
        token_tags = predictor.predict(doc=self.doc)
        assert len(token_tags) == 340

        self.doc.annotate_layer(name="token_tags", entities=token_tags)
        for token_tag in token_tags:
            assert isinstance(token_tag.metadata.label, str)
            assert isinstance(token_tag.metadata.score, float)

    def test_predict_bibs_tokens(self):
        self.predictor.context_name = "bibs"
        token_tags = self.predictor.predict(doc=self.doc)
        assert len(token_tags) == 38

    def test_missing_fields(self):
        self.predictor.entity_name = "OHNO"
        with self.assertRaises(AssertionError) as e:
            self.predictor.predict(doc=self.doc)
            assert "OHNO" in e.exception

        self.predictor.entity_name = "tokens"
        self.predictor.context_name = "BLABLA"
        with self.assertRaises(AssertionError) as e:
            self.predictor.predict(doc=self.doc)
            assert "BLABLA" in e.exception

        self.predictor.context_name = "pages"

    def test_predict_pages_tokens_roberta(self):
        predictor = HFBIOTaggerPredictor.from_pretrained(
            model_name_or_path="roberta-base",
            entity_name="tokens",
            context_name="pages",
            add_prefix_space=True,  # Needed for roberta
            **{"num_labels": len(self.id2label), "id2label": self.id2label, "label2id": self.label2id},
        )
        token_tags = predictor.predict(doc=self.doc)
        assert len(token_tags) == 924

        self.doc.annotate_layer(name="token_tags", entities=token_tags)
        for token_tag in token_tags:
            assert isinstance(token_tag.metadata.label, str)
            assert isinstance(token_tag.metadata.score, float)

    # def test_postprocess(self):
    #     self.predictor.postprocess(
    #         doc=self.doc,
    #         context_name="pages",
    #         preds=[
    #             BIOPrediction(context_id=0, entity_id=0, label="B-Label", score=0.4),
    #             BIOPrediction(context_id=0, entity_id=1, label="I-Label", score=0.2),
    #             BIOPrediction(context_id=0, entity_id=2, label="O", score=0.3),
    #             BIOPrediction(context_id=0, entity_id=3, label=None, score=None),
    #             BIOPrediction(context_id=0, entity_id=4, label="B-Label", score=0.4),
    #             BIOPrediction(context_id=0, entity_id=5, label=None, score=None),
    #             BIOPrediction(context_id=0, entity_id=6, label="I-Label", score=0.2),
    #         ],
    #     )
