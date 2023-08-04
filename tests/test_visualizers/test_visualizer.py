"""

@kylel

"""

import json
import os
import pathlib
import re
import unittest

import numpy as np

from papermage.magelib import Box, Document, Entity, Span
from papermage.parsers import PDFPlumberParser
from papermage.rasterizers import PDF2ImageRasterizer


class TestVisualizer(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        with open(self.fixture_path / "2304.02623v1.json") as f_in:
            doc_dict = json.load(f_in)
            self.doc = Document.from_json(doc_dict)

        rasterizer = PDF2ImageRasterizer()
        images = rasterizer.rasterize(input_pdf_path=str(self.fixture_path / "2304.02623v1.pdf"), dpi=72)
        rasterizer.attach_images(images=images, doc=self.doc)
        page = self.doc.pages[0]
        page_image = page.images[0]
        entities_on_page = page.tokens
