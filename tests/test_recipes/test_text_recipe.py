"""

@kylel

"""

import pathlib
import unittest

from papermage.recipes import TextRecipe


class TestCoreRecipe(unittest.TestCase):
    def setUp(self):
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        self.recipe = TextRecipe()

    def test_stability(self):
        doc = self.recipe.run("This is a test document. This is a test document. This is a test document.")
        self.assertEqual(doc.symbols, "This is a test document. This is a test document. This is a test document.")
        self.assertEqual(len(doc.tokens), 15)
        self.assertEqual(len(doc.sentences), 3)
        self.assertEqual(doc.tokens[0].text, "This")
        self.assertEqual(doc.sentences[0].text, "This is a test document.")
