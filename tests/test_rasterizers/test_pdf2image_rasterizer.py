"""

@benjaminn, @kylel

"""

import json
import os
import pathlib
import re
import unittest

from papermage.magelib import Document, Entity, Image, Span
from papermage.rasterizers import PDF2ImageRasterizer


class TestPDF2ImageRasterizer(unittest.TestCase):
    def setUp(cls) -> None:
        cls.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"

    def test_raseterize(self):
        rasterizer = PDF2ImageRasterizer()
        images = rasterizer.rasterize(input_pdf_path=self.fixture_path / "2304.02623v1.pdf", dpi=72)
        assert len(images) == 4
        assert images[0].pilimage.size == (612, 792)
        assert isinstance(images[0], Image)

    def test_attach(self):
        rasterizer = PDF2ImageRasterizer()
        image = Image.create_rgb_random()
        doc = Document(symbols="This is a doc.")
        page = Entity(spans=[Span(start=0, end=14)])
        doc.annotate_layer(name="pages", entities=[page])
        rasterizer.attach_images(images=[image], doc=doc)
        assert doc.pages[0].images[0] == image
