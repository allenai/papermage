"""

@kylel

"""

import pathlib
import unittest

from papermage.recipes import CoreRecipe


class TestCoreRecipe(unittest.TestCase):
    def setUp(self):
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        self.recipe = CoreRecipe()

    def test_stability(self):
        doc = self.recipe.from_path(pdfpath=self.fixture_path / "1903.10676.pdf")
        doc = self.recipe.from_path(pdfpath=self.fixture_path / "2304.02623v1.pdf")
        doc = self.recipe.from_path(pdfpath=self.fixture_path / "2020.acl-main.447.pdf")
        doc = self.recipe.from_path(pdfpath=self.fixture_path / "4be952924cd565488b4a239dc6549095029ee578.pdf")
        doc = self.recipe.from_path(pdfpath=self.fixture_path / "2023.eacl-main.121.pdf")
