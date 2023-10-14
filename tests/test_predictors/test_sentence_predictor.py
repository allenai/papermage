"""

Test sentence predictor

@kylel

"""

import json
import pathlib
import unittest

from papermage.magelib import Document, Entity, Span
from papermage.predictors import PysbdSentencePredictor, SVMWordPredictor


class TestPysbdSentencePredictor(unittest.TestCase):
    def setUp(self):
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        with open(self.fixture_path / "2304.02623v1.json") as f_in:
            self.doc = Document.from_json(json.load(f_in))
        # TODO: seems like Words causes SentencePredictor to behave better
        #       may be worth testing.
        # word_predictor = SVMWordPredictor.from_path(
        #     tar_path=str(self.fixture_path / "svm_word_predictor/svm_word_predictor.tar.gz")
        # )
        # words = word_predictor.predict(self.doc)
        # self.doc.annotate_entity(field_name="words", entities=words)

    def test_predict(self):
        predictor = PysbdSentencePredictor()
        doc = Document(symbols="This is a test. This is another test.")
        tokens = [
            Entity(spans=[Span(0, 4)]),
            Entity(spans=[Span(5, 7)]),
            Entity(spans=[Span(8, 9)]),
            Entity(spans=[Span(10, 14)]),
            Entity(spans=[Span(14, 15)]),
            Entity(spans=[Span(16, 20)]),
            Entity(spans=[Span(21, 23)]),
            Entity(spans=[Span(24, 31)]),
            Entity(spans=[Span(32, 36)]),
            Entity(spans=[Span(36, 37)]),
        ]
        doc.annotate_layer(name="tokens", entities=tokens)
        sents = predictor.predict(doc)
        doc.annotate_layer(name="sentences", entities=sents)
        self.assertEqual(len(doc.sentences), 2)
        self.assertEqual(doc.sentences[0].text, "This is a test.")
        self.assertEqual(doc.sentences[1].text, "This is another test.")

    def test_predict_paper(self):
        predictor = PysbdSentencePredictor()
        sents = predictor.predict(self.doc)
        self.doc.annotate_layer(name="sentences", entities=sents)
        self.assertEqual(len(self.doc.sentences), 310)

        valid_sent_1 = """Large language models have introduced exciting new opportunities and challenges in designing and developing new AI-assisted writing support tools."""
        self.assertTrue(any([valid_sent_1 in s.text.replace("\n", " ") for s in self.doc.sentences[:10]]))

        valid_sent_2 = """Recent work has shown that leveraging this new tech- nology can transform writing in many scenarios such as ideation during creative writing, editing support, and summarization."""
        self.assertTrue(any([valid_sent_2 in s.text.replace("\n", " ") for s in self.doc.sentences[:10]]))

        valid_sent_3 = """How- ever, AI-supported expository writing —including real-world tasks like scholars writing literature reviews or doctors writing progress notes—is relatively understudied."""
        self.assertTrue(any([valid_sent_3 in s.text.replace("\n", " ") for s in self.doc.sentences[:10]]))

        valid_sent_4 = """In this position paper, we argue that developing AI supports for expository writing has unique and exciting research challenges and can lead to high real-world impacts."""
        self.assertTrue(any([valid_sent_4 in s.text.replace("\n", " ") for s in self.doc.sentences[:10]]))
