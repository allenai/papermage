"""

@shannons, @kylel

"""

import json
import os
import pathlib
import unittest

from PIL import Image

from papermage.magelib import Document
from papermage.parsers.pdfplumber_parser import PDFPlumberParser
from papermage.predictors import (
    IVILATokenClassificationPredictor,
    LPEffDetPubLayNetBlockPredictor,
)
from papermage.rasterizers.rasterizer import PDF2ImageRasterizer


class TestFigureVilaPredictors(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"

        cls.DOCBANK_LABEL_MAP = {
            "0": "paragraph",
            "1": "title",
            "2": "equation",
            "3": "reference",
            "4": "section",
            "5": "list",
            "6": "table",
            "7": "caption",
            "8": "author",
            "9": "abstract",
            "10": "footer",
            "11": "date",
            "12": "figure",
        }
        cls.DOCBANK_LABEL_MAP = {int(key): val for key, val in cls.DOCBANK_LABEL_MAP.items()}

        cls.S2VL_LABEL_MAP = {
            "0": "Title",
            "1": "Author",
            "2": "Abstract",
            "3": "Keywords",
            "4": "Section",
            "5": "Paragraph",
            "6": "List",
            "7": "Bibliography",
            "8": "Equation",
            "9": "Algorithm",
            "10": "Figure",
            "11": "Table",
            "12": "Caption",
            "13": "Header",
            "14": "Footer",
            "15": "Footnote",
        }

        cls.S2VL_LABEL_MAP = {int(key): val for key, val in cls.S2VL_LABEL_MAP.items()}

    def test_vila_predictors(self):
        layout_predictor = LPEffDetPubLayNetBlockPredictor.from_pretrained()

        pdfplumber_parser = PDFPlumberParser()
        rasterizer = PDF2ImageRasterizer()

        doc = pdfplumber_parser.parse(input_pdf_path=self.fixture_path / "1903.10676.pdf")
        images = rasterizer.rasterize(input_pdf_path=self.fixture_path / "1903.10676.pdf", dpi=72)
        doc.annotate_images(images)

        layout_regions = layout_predictor.predict(doc)
        doc.annotate_layer(name="blocks", entities=layout_regions)

        predictor_with_blocks = IVILATokenClassificationPredictor.from_pretrained(
            "allenai/ivila-block-layoutlm-finetuned-docbank"
        )
        results_with_blocks = predictor_with_blocks.predict(doc=doc)

        predictor_with_rows = IVILATokenClassificationPredictor.from_pretrained(
            "allenai/ivila-row-layoutlm-finetuned-s2vl-v2"
        )
        results_with_rows = predictor_with_rows.predict(doc=doc)
