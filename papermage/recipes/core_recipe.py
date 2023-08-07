"""

@kylel

"""

import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

from papermage.magelib import (
    BlocksFieldName,
    Document,
    EntitiesFieldName,
    Entity,
    ImagesFieldName,
    MetadataFieldName,
    PagesFieldName,
    ParagraphsFieldName,
    RelationsFieldName,
    RowsFieldName,
    SentencesFieldName,
    SymbolsFieldName,
    TokensFieldName,
    WordsFieldName,
)
from papermage.parsers.pdfplumber_parser import PDFPlumberParser
from papermage.predictors import (
    HFBIOTaggerPredictor,
    IVILATokenClassificationPredictor,
    LPBlockPredictor,
    PysbdSentencePredictor,
    SVMWordPredictor,
)
from papermage.rasterizers.rasterizer import PDF2ImageRasterizer
from papermage.recipes.recipe import Recipe


class CoreRecipe(Recipe):
    def __init__(
        self,
        effdet_publaynet_predictor_path: str = "lp://efficientdet/PubLayNet",
        effdet_mfd_predictor_path: str = "lp://efficientdet/MFD",
        ivila_predictor_path: str = "allenai/ivila-row-layoutlm-finetuned-s2vl-v2",
        bio_roberta_predictor_path: str = "allenai/vila-roberta-large-s2vl-internal",
        svm_word_predictor_path: str = "https://ai2-s2-research-public.s3.us-west-2.amazonaws.com/mmda/models/svm_word_predictor.tar.gz",
    ):
        logger.info("Instantiating recipe...")
        self.parser = PDFPlumberParser()
        self.rasterizer = PDF2ImageRasterizer()

        self.word_predictor = SVMWordPredictor.from_path(svm_word_predictor_path)
        self.effdet_publaynet_predictor = LPBlockPredictor.from_pretrained(effdet_publaynet_predictor_path)
        self.effdet_mfd_predictor = LPBlockPredictor.from_pretrained(effdet_mfd_predictor_path)
        self.ivila_predictor = IVILATokenClassificationPredictor.from_pretrained(ivila_predictor_path)
        self.bio_roberta_predictor = HFBIOTaggerPredictor.from_pretrained(
            bio_roberta_predictor_path,
            entity_name="tokens",
            context_name="pages",
        )
        self.sent_predictor = PysbdSentencePredictor()
        logger.info("Finished instantiating recipe")

    def run(self, pdf: Union[str, Path, Document]) -> Document:
        if isinstance(pdf, str):
            pdf = Path(pdf)
            assert pdf.exists(), f"File {pdf} does not exist."
        assert isinstance(
            pdf, (Document, Path)
        ), f"Unsupported type {type(pdf)} for pdf; should be a Document or a path to a PDF file."
        if isinstance(pdf, Path):
            self.from_path(str(pdf))
        else:
            raise NotImplementedError("Document input not yet supported.")

    def from_path(self, pdfpath: str) -> Document:
        logger.info("Parsing document...")
        doc = self.parser.parse(input_pdf_path=pdfpath)

        logger.info("Rasterizing document...")
        images = self.rasterizer.rasterize(input_pdf_path=pdfpath, dpi=72)
        doc.annotate_images(images=list(images))
        self.rasterizer.attach_images(images=images, doc=doc)

        logger.info("Predicting words...")
        words = self.word_predictor.predict(doc=doc)
        doc.annotate_entity(field_name=WordsFieldName, entities=words)

        logger.info("Predicting sentences...")
        sentences = self.sent_predictor.predict(doc=doc)
        doc.annotate_entity(field_name=SentencesFieldName, entities=sentences)

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
        doc.annotate_entity(field_name=BlocksFieldName, entities=blocks)

        logger.info("Predicting vila...")
        vila_entities = self.ivila_predictor.predict(doc=doc)
        doc.annotate_entity(field_name="vila_entities", entities=vila_entities)

        return doc
