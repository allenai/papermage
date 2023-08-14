"""

@kylel

"""

import os
import pathlib
import unittest

from papermage.magelib import Document, Entity, Image
from papermage.recipes import CoreRecipe
from tests.test_recipes.core_recipe_fixtures import (  # SEGMENT_OF_WORD_JSONS,
    BASE64_PAGE_IMAGE,
    FIRST_3_BLOCKS_JSON,
    FIRST_5_ROWS_JSON,
    FIRST_10_TOKENS_JSON,
    FIRST_10_VILA_JSONS,
    FIRST_1000_SYMBOLS,
    PAGE_JSON,
)


def round_all_floats(d: dict):
    import numbers

    def formatfloat(x):
        return "%.4g" % float(x)

    def pformat(dictionary, function):
        if isinstance(dictionary, dict):
            return {key: pformat(value, function) for key, value in dictionary.items()}
        if isinstance(dictionary, list):
            return [pformat(element, function) for element in dictionary]
        if isinstance(dictionary, numbers.Number):
            return function(dictionary)
        return dictionary

    return pformat(d, formatfloat)


class TestCoreRecipe(unittest.TestCase):
    def setUp(self):
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        self.recipe = CoreRecipe()

    def test_stability(self):
        self.recipe.from_path(pdfpath=self.fixture_path / "1903.10676.pdf")
        self.recipe.from_path(pdfpath=self.fixture_path / "2304.02623v1.pdf")
        self.recipe.from_path(pdfpath=self.fixture_path / "2020.acl-main.447.pdf")
        self.recipe.from_path(pdfpath=self.fixture_path / "4be952924cd565488b4a239dc6549095029ee578.pdf")
        self.recipe.from_path(pdfpath=self.fixture_path / "2023.eacl-main.121.pdf")
