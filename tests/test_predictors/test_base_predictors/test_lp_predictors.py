import json
import pathlib
import unittest

from papermage.magelib import Entity
from papermage.parsers import PDFPlumberParser
from papermage.predictors.base_predictors.lp_predictors import LPPredictor
from papermage.rasterizers import PDF2ImageRasterizer


class TestLayoutParserBoxPredictor(unittest.TestCase):
    def setUp(self):
        self.fixture_path = pathlib.Path(__file__).parent.parent.parent / "fixtures"
        self.parser = PDFPlumberParser()
        self.rasterizer = PDF2ImageRasterizer()
        self.layout_predictor = LPPredictor.from_pretrained("lp://efficientdet/PubLayNet")

    def test_predict(self):
        input_pdf_path = self.fixture_path / "4be952924cd565488b4a239dc6549095029ee578.pdf"

        gold_blocks_path = self.fixture_path / "4be952924cd565488b4a239dc6549095029ee578_lp_blocks.json"
        with open(gold_blocks_path, "r") as f:
            gold_blocks_json = json.load(f)
            gold_blocks_list = [Entity.from_json(entity_json=block) for block in gold_blocks_json]

        doc = self.parser.parse(input_pdf_path=input_pdf_path)
        images = self.rasterizer.rasterize(input_pdf_path=input_pdf_path, dpi=72)
        # PDF2ImageRasterizer.attach_images(images=images, doc=doc)
        doc.annotate_images(images=images)
        pred_blocks_list = self.layout_predictor.predict(doc=doc)

        for pred_block_ents, gold_block_ents in zip(pred_blocks_list, gold_blocks_list):
            assert pred_block_ents.metadata["type"] == gold_block_ents.metadata["type"]
            for pred_box, gold_box in zip(pred_block_ents.boxes, gold_block_ents.boxes):
                for field in ["page", "w", "h", "l", "t"]:
                    self.assertAlmostEqual(getattr(pred_box, field), getattr(gold_box, field), places=3)
