"""

@kylel, benjaminn

"""

import json
import pathlib
import unittest

from papermage.parsers import PDFPlumberParser
from papermage.predictors import EntityClassificationPredictor
from papermage.types import Document, Entity, Span

TEST_SCIBERT_WEIGHTS = "allenai/scibert_scivocab_uncased"
# TEST_BIB_PARSER_WEIGHTS = '/Users/kylel/ai2/mmda/stefans/'


class TestEntityClassificationPredictor(unittest.TestCase):
    def setUp(self):
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        with open(self.fixture_path / "entity_classification_predictor_test_doc_papermage.json", "r") as f:
            test_doc_json = json.load(f)
        self.doc = Document.from_json(doc_json=test_doc_json)
        ent1 = Entity(spans=[Span(start=86, end=456)])
        ent2 = Entity(spans=[Span(start=457, end=641)])
        self.doc.annotate_entity(field_name="bibs", entities=[ent1, ent2])
        ent1.id = 0
        ent2.id = 1

        self.predictor = EntityClassificationPredictor.from_pretrained(
            model_name_or_path=TEST_SCIBERT_WEIGHTS, entity_name="tokens", context_name="pages"
        )

    def test_predict_pages_tokens(self):
        predictor = EntityClassificationPredictor.from_pretrained(
            model_name_or_path=TEST_SCIBERT_WEIGHTS, entity_name="tokens", context_name="pages"
        )
        token_tags = predictor.predict(doc=self.doc)
        assert len(token_tags) == len([token for page in self.doc.pages for token in page.tokens])

        self.doc.annotate_entity(field_name="token_tags", entities=token_tags)
        for token_tag in token_tags:
            assert isinstance(token_tag.metadata.label, str)
            assert isinstance(token_tag.metadata.score, float)

    def test_predict_bibs_tokens(self):
        self.predictor.context_name = "bibs"
        token_tags = self.predictor.predict(doc=self.doc)
        assert len(token_tags) == len([token for bib in self.doc.bibs for token in bib.tokens])

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
        predictor = EntityClassificationPredictor.from_pretrained(
            model_name_or_path="roberta-base",
            entity_name="tokens",
            context_name="pages",
            add_prefix_space=True,  # Needed for roberta
        )
        token_tags = predictor.predict(doc=self.doc)
        assert len(token_tags) == len([token for page in self.doc.pages for token in page.tokens])

        self.doc.annotate_entity(field_name="token_tags", entities=token_tags)
        for token_tag in token_tags:
            assert isinstance(token_tag.metadata.label, str)
            assert isinstance(token_tag.metadata.score, float)
