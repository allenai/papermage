"""

@kylel

"""


import json
import os
import pathlib
import unittest

from papermage.magelib import Document, Entity, Metadata, Span
from papermage.predictors import PaperQaPredictor


class TestPaperQAPredictor(unittest.TestCase):
    def setUp(self):
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        with open(self.fixture_path / "2304.02623v1.json", "r") as f:
            test_doc_json = json.load(f)
        self.doc = Document.from_json(doc_json=test_doc_json)

        self.paper_qa_predictor = PaperQaPredictor()

        self.using_github_actions = (
            "USING_GITHUB_ACTIONS" in os.environ and os.environ["USING_GITHUB_ACTIONS"] == "true"
        )

    def test_merge_adjacent_sentences(self):
        locs = [0, 1, 2, 3]
        result = self.paper_qa_predictor.merge_adjacent_sentences(locs=locs, slack=1, max_len=3)
        self.assertListEqual(result, [[0, 1, 2], [], [], [3]])

        locs = [0, 1, 3, 5, 8]
        result = self.paper_qa_predictor.merge_adjacent_sentences(locs=locs, slack=2, max_len=3)
        self.assertListEqual(result, [[0, 1, 2], [], [], [3]])
