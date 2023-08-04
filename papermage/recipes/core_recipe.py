"""

@kylel

"""

import logging

logger = logging.getLogger(__name__)

from papermage.parsers.pdfplumber_parser import PDFPlumberParser
from papermage.predictors import HFEntityClassificationPredictor, LPBlockPredictor
from papermage.rasterizers.rasterizer import PDF2ImageRasterizer
from papermage.recipes.recipe import Recipe
from papermage.types import Document, Entity


class CoreRecipe(Recipe):
    def __init__(
        self,
        effdet_publaynet_predictor_path: str = "lp://efficientdet/PubLayNet",
        effdet_mfd_predictor_path: str = "lp://efficientdet/MFD",
        vila_predictor_path: str = "allenai/vila-roberta-large-s2vl-internal",
    ):
        logger.info("Instantiating recipe...")
        self.parser = PDFPlumberParser()
        self.rasterizer = PDF2ImageRasterizer()

        self.effdet_publaynet_predictor = LPBlockPredictor.from_pretrained(effdet_publaynet_predictor_path)
        self.effdet_mfd_predictor = LPBlockPredictor.from_pretrained(effdet_mfd_predictor_path)
        self.vila_predictor = HFEntityClassificationPredictor.from_pretrained(
            vila_predictor_path,
            entity_name="tokens",
            context_name="pages",
        )
        logger.info("Finished instantiating recipe")

    def from_path(self, pdfpath: str) -> Document:
        logger.info("Parsing document...")
        doc = self.parser.parse(input_pdf_path=pdfpath)

        logger.info("Rasterizing document...")
        images = self.rasterizer.rasterize(input_pdf_path=pdfpath, dpi=72)
        doc.annotate_images(images=list(images))

        logger.info("Predicting blocks...")

        layout = self.effdet_publaynet_predictor.predict(doc=doc)
        equations = self.effdet_mfd_predictor.predict(doc=doc)

        # we annotate layout info in the document
        doc.annotate_entity(field_name="layout", entities=layout)

        # list annotations separately
        doc.annotate_entity(field_name="equations", entities=equations)

        # blocks are used by IVILA, so we need to annotate them as well
        # copy the entities because they already have docs attached
        blocks = [Entity.from_json(ent.to_json()) for ent in layout + equations]
        doc.annotate_entity(field_name="blocks", entities=blocks)

        logger.info("Predicting vila...")
        vila_entities = self.vila_predictor.predict(doc=doc)
        doc.annotate_entity(field_name="vila_entities", entities=vila_entities)

        return doc
