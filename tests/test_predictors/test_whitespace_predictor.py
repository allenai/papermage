"""


@kylel

"""

import unittest

from papermage.magelib import Document, Entity, Span
from papermage.predictors import HFWhitspaceTokenPredictor


class TestWhitespacePredictor(unittest.TestCase):
    def test_predict(self):
        # fmt:off
        #          0         10        20        30        40        50        60        70
        #          01234567890123456789012345678901234567890123456789012345678901234567890123456789
        symbols = "The goal of meta-learning is to train a model on a vari&ety of learning tasks"
        # fmt:on

        spans = [
            Span(start=0, end=3),  # The
            Span(start=4, end=8),  # goal
            Span(start=9, end=11),  # of
            Span(start=12, end=16),  # meta
            Span(start=16, end=17),  # -
            Span(start=17, end=25),  # learning
            Span(start=26, end=28),  # is
            Span(start=29, end=31),  # to
            Span(start=32, end=37),  # train
            Span(start=38, end=39),  # a
            Span(start=40, end=45),  # model
            Span(start=46, end=48),  # on
            Span(start=49, end=50),  # a
            Span(start=51, end=56),  # vari&
            Span(start=56, end=59),  # ety
            Span(start=60, end=62),  # of
            Span(start=63, end=71),  # learning
            Span(start=72, end=77),  # tasks
        ]

        doc = Document(symbols=symbols)
        doc.annotate_layer(name="tokens", entities=[Entity(spans=[span]) for i, span in enumerate(spans)])

        predictor = HFWhitspaceTokenPredictor()
        ws_chunks = predictor.predict(doc)

        doc.annotate_layer(name="ws_chunks", entities=ws_chunks)
        self.assertEqual(
            [c.text for c in doc.ws_chunks],
            [
                "The",
                "goal",
                "of",
                "meta-learning",
                "is",
                "to",
                "train",
                "a",
                "model",
                "on",
                "a",
                "vari&ety",
                "of",
                "learning",
                "tasks",
            ],
        )
        self.assertEqual(" ".join([c.text for c in doc.ws_chunks]), doc.symbols)
