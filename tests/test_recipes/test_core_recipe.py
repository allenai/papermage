"""

@kylel

"""

import pathlib
import unittest

from papermage.recipes import CoreRecipe


class TestCoreRecipe(unittest.TestCase):
    def setUp(self):
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        self.recipe = CoreRecipe(
            svm_word_predictor_path=str(self.fixture_path / "svm_word_predictor/svm_word_predictor.tar.gz")
        )

    def test_stability(self):
        self.recipe.run(self.fixture_path / "1903.10676.pdf")
        # beyond summarization
        self.recipe.run(self.fixture_path / "2304.02623v1.pdf")
        # semantic reader
        # self.recipe.run(self.fixture_path / "2303.14334v2.pdf")
        # papermage
        # self.recipe.run(self.fixture_path / "papermage.pdf")
        # s2orc
        self.recipe.run(self.fixture_path / "2020.acl-main.447.pdf")
        self.recipe.run(self.fixture_path / "4be952924cd565488b4a239dc6549095029ee578.pdf")
        # longeval
        self.recipe.run(self.fixture_path / "2023.eacl-main.121.pdf")
        # citesee
        self.recipe.run(self.fixture_path / "2302.07302v1.pdf")
