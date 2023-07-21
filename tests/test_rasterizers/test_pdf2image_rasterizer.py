"""
@benjaminn
"""

import json
import os
import pathlib
import re
import unittest

from papermage.rasterizers import PDF2ImageRasterizer
from papermage.types import Image


class TestPDF2ImageRasterizer(unittest.TestCase):
    def setUp(cls) -> None:
        cls.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"

    def test_raseterize(self):
        rasterizer = PDF2ImageRasterizer()
        images = rasterizer.rasterize(input_pdf_path=self.fixture_path / "2304.02623v1.pdf", dpi=72)
        assert len(images) == 4
        assert images[0].pilimage.size == (612, 792)
        assert isinstance(images[0], Image)
