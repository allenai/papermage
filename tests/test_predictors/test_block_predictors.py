"""

Block predictors.

@kylel

"""

import pathlib
import unittest

from papermage.predictors import LPEffDetPubLayNetBlockPredictor


class TestLayoutParserBoxPubLayNetBlockPredictor(unittest.TestCase):
    def setUp(self):
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        self.predictor = LPEffDetPubLayNetBlockPredictor.from_pretrained()

    def test_predict(self):
        pass
