"""

Minimal recipe that handles PDF binaries.

@kylel

"""

import logging
from pathlib import Path
from typing import Dict, List, Union

from papermage.magelib import Document, Entity
from papermage.parsers.pdfplumber_parser import PDFPlumberParser
from papermage.rasterizers.rasterizer import PDF2ImageRasterizer
from papermage.recipes.recipe import Recipe
from papermage.utils.annotate import group_by


class MinimalTextImageRecipe(Recipe):
    def __init__(
        self,
        dpi: int = 300,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dpi = dpi
        self.logger.info("Instantiating recipe...")
        self.parser = PDFPlumberParser()
        self.rasterizer = PDF2ImageRasterizer()
        self.logger.info("Finished instantiating recipe")

    def from_pdf(self, pdf: Path) -> Document:
        self.logger.info("Parsing document...")
        doc = self.parser.parse(input_pdf_path=pdf)

        self.logger.info("Rasterizing document...")
        images = self.rasterizer.rasterize(input_pdf_path=pdf, dpi=self.dpi)
        doc.annotate_images(images=list(images))
        self.rasterizer.attach_images(images=images, doc=doc)
        return doc


class MinimalTextOnlyRecipe(Recipe):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Instantiating recipe...")
        self.parser = PDFPlumberParser()
        self.logger.info("Finished instantiating recipe")

    def from_pdf(self, pdf: Path) -> Document:
        self.logger.info("Parsing document...")
        doc = self.parser.parse(input_pdf_path=pdf)

        self.logger.info("Rasterizing document...")
        return doc


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, type=str, help="Path to PDF file.")
    parser.add_argument("--output", type=str, help="Path to output JSON file.")
    parser.add_argument("--recipe", type=str, default="text_only", help="Recipe to use.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rasterization.")
    args = parser.parse_args()

    if args.recipe == "text_image":
        recipe = MinimalTextImageRecipe(dpi=args.dpi)
        doc = recipe.from_pdf(pdf=args.pdf)
        with open(args.output, "w") as f:
            json.dump(doc.to_json(), f, indent=2)
    elif args.recipe == "text_only":
        recipe = MinimalTextOnlyRecipe()
        doc = recipe.from_pdf(pdf=args.pdf)
        with open(args.output, "w") as f:
            json.dump(doc.to_json(), f, indent=2)
    else:
        raise ValueError(f"Invalid recipe: {args.recipe}")
