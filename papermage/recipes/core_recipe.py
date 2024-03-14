"""

@kylel

"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Union

from papermage.magelib import (
    AbstractsFieldName,
    AlgorithmsFieldName,
    AuthorsFieldName,
    BibliographiesFieldName,
    BlocksFieldName,
    Box,
    CaptionsFieldName,
    Document,
    EntitiesFieldName,
    Entity,
    EquationsFieldName,
    FiguresFieldName,
    FootersFieldName,
    FootnotesFieldName,
    HeadersFieldName,
    ImagesFieldName,
    KeywordsFieldName,
    ListsFieldName,
    PagesFieldName,
    ParagraphsFieldName,
    RelationsFieldName,
    RowsFieldName,
    SectionsFieldName,
    SentencesFieldName,
    SymbolsFieldName,
    TablesFieldName,
    TitlesFieldName,
    TokensFieldName,
    WordsFieldName,
)
from papermage.parsers.pdfplumber_parser import PDFPlumberParser
from papermage.predictors import (
    HFBIOTaggerPredictor,
    IVILATokenClassificationPredictor,
    LPEffDetFormulaPredictor,
    LPEffDetPubLayNetBlockPredictor,
    PysbdSentencePredictor,
    SVMWordPredictor,
)
from papermage.predictors.word_predictors import make_text
from papermage.rasterizers.rasterizer import PDF2ImageRasterizer
from papermage.recipes.recipe import Recipe
from papermage.utils.annotate import group_by

VILA_LABELS_MAP = {
    "Title": TitlesFieldName,
    "Paragraph": ParagraphsFieldName,
    "Author": AuthorsFieldName,
    "Abstract": AbstractsFieldName,
    "Keywords": KeywordsFieldName,
    "Section": SectionsFieldName,
    "List": ListsFieldName,
    "Bibliography": BibliographiesFieldName,
    "Equation": EquationsFieldName,
    "Algorithm": AlgorithmsFieldName,
    # "Figure": FiguresFieldName,
    # "Table": TablesFieldName,
    "Caption": CaptionsFieldName,
    "Header": HeadersFieldName,
    "Footer": FootersFieldName,
    "Footnote": FootnotesFieldName,
}


class CoreRecipe(Recipe):
    def __init__(
        self,
        ivila_predictor_path: str = "allenai/ivila-row-layoutlm-finetuned-s2vl-v2",
        bio_roberta_predictor_path: str = "allenai/vila-roberta-large-s2vl-internal",
        svm_word_predictor_path: str = "https://ai2-s2-research-public.s3.us-west-2.amazonaws.com/mmda/models/svm_word_predictor.tar.gz",
        dpi: int = 72,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dpi = dpi

        self.logger.info("Instantiating recipe...")
        self.parser = PDFPlumberParser()
        self.rasterizer = PDF2ImageRasterizer()

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     self.word_predictor = SVMWordPredictor.from_path(svm_word_predictor_path)

        self.publaynet_block_predictor = LPEffDetPubLayNetBlockPredictor.from_pretrained()
        self.ivila_predictor = IVILATokenClassificationPredictor.from_pretrained(ivila_predictor_path)
        self.sent_predictor = PysbdSentencePredictor()
        self.logger.info("Finished instantiating recipe")

    def from_pdf(self, pdf: Path) -> Document:
        self.logger.info("Parsing document...")
        doc = self.parser.parse(input_pdf_path=pdf)

        self.logger.info("Rasterizing document...")
        images = self.rasterizer.rasterize(input_pdf_path=pdf, dpi=self.dpi)
        doc.annotate_images(images=list(images))
        self.rasterizer.attach_images(images=images, doc=doc)
        return self.from_doc(doc=doc)

    def from_doc(self, doc: Document) -> Document:
        # self.logger.info("Predicting words...")
        # words = self.word_predictor.predict(doc=doc)
        # doc.annotate_layer(name=WordsFieldName, entities=words)

        self.logger.info("Predicting sentences...")
        sentences = self.sent_predictor.predict(doc=doc)
        doc.annotate_layer(name=SentencesFieldName, entities=sentences)

        self.logger.info("Predicting blocks...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blocks = self.publaynet_block_predictor.predict(doc=doc)
        doc.annotate_layer(name=BlocksFieldName, entities=blocks)

        self.logger.info("Predicting figures and tables...")
        figures = []
        tables = []
        for block in blocks:
            if block.metadata.type == "Figure":
                figure = Entity(boxes=block.boxes)
                figures.append(figure)
            elif block.metadata.type == "Table":
                table = Entity(boxes=block.boxes)
                tables.append(table)
        doc.annotate_layer(name=FiguresFieldName, entities=figures)
        doc.annotate_layer(name=TablesFieldName, entities=tables)

        # self.logger.info("Predicting vila...")
        vila_entities = self.ivila_predictor.predict(doc=doc)
        doc.annotate_layer(name="vila_entities", entities=vila_entities)

        for entity in vila_entities:
            entity.boxes = [
                Box.create_enclosing_box(
                    [b for t in doc.intersect_by_span(entity, name=TokensFieldName) for b in t.boxes]
                )
            ]
            # entity.text = make_text(entity=entity, document=doc)
        preds = group_by(entities=vila_entities, metadata_field="label", metadata_values_map=VILA_LABELS_MAP)
        doc.annotate(*preds)
        return doc


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, type=str, help="Path to PDF file.")
    parser.add_argument("--output", type=str, help="Path to output JSON file.")
    args = parser.parse_args()

    recipe = CoreRecipe()
    doc = recipe.from_pdf(pdf=args.pdf)
    with open(args.output, "w") as f:
        json.dump(doc.to_json(), f, indent=2)
