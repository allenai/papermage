"""

@kylel

"""

import pathlib
import unittest

from papermage.recipes import MinimalTextImageRecipe, MinimalTextOnlyRecipe


class TestMinimalTextOnlyRecipe(unittest.TestCase):
    def setUp(self):
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        self.recipe = MinimalTextOnlyRecipe()

    def test_stability(self):
        doc = self.recipe.run(self.fixture_path / "1903.10676.pdf")
        self.assertTrue(doc.symbols.startswith("Field\nTask\nDataset\nSOTA\nB ERT -Base\nS CI B ERT"))
        self.assertListEqual([t.text for t in doc.tokens[:5]], ["Field", "Task", "Dataset", "SOTA", "B"])


class TestMinimalTextImageRecipe(unittest.TestCase):
    def setUp(self):
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        self.recipe_high_quality = MinimalTextImageRecipe(dpi=300)
        self.recipe_low_quality = MinimalTextImageRecipe(dpi=72)

    def test_high_quality(self):
        doc = self.recipe_high_quality.run(self.fixture_path / "1903.10676.pdf")
        self.assertTrue(doc.symbols.startswith("Field\nTask\nDataset\nSOTA\nB ERT -Base\nS CI B ERT"))
        self.assertListEqual([t.text for t in doc.tokens[:5]], ["Field", "Task", "Dataset", "SOTA", "B"])
        self.assertEqual(len(doc.images), 1)
        self.assertEqual(doc.images[0].pilimage.height, 3509)
        self.assertEqual(doc.images[0].pilimage.width, 2480)

    def test_low_quality(self):
        doc = self.recipe_low_quality.run(self.fixture_path / "1903.10676.pdf")
        self.assertTrue(doc.symbols.startswith("Field\nTask\nDataset\nSOTA\nB ERT -Base\nS CI B ERT"))
        self.assertListEqual([t.text for t in doc.tokens[:5]], ["Field", "Task", "Dataset", "SOTA", "B"])
        self.assertEqual(len(doc.images), 1)
        self.assertEqual(doc.images[0].pilimage.height, 842)
        self.assertEqual(doc.images[0].pilimage.width, 595)
