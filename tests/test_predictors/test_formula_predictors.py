"""

Formula predictors.

@kylel

"""

import pathlib
import unittest

from papermage.predictors import LPEffDetFormulaPredictor


class TestLayoutParserBoxMathFormulaBlockPredictor(unittest.TestCase):
    def setUp(self):
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        self.predictor = LPEffDetFormulaPredictor.from_pretrained()

    def test_predict(self):
        pass
