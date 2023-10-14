"""

Create a Document using only text <str> input. Useful for test and development.

@kylel

"""

import logging
from pathlib import Path
from typing import Dict, List, Union

from papermage.magelib import Box, Document, Entity, SentencesFieldName, TokensFieldName
from papermage.predictors import HFWhitspaceTokenPredictor, PysbdSentencePredictor
from papermage.recipes.recipe import Recipe


class TextRecipe(Recipe):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tokenizer_predictor = HFWhitspaceTokenPredictor()
        self.sent_predictor = PysbdSentencePredictor()

    def from_str(self, text: str) -> Document:
        doc = Document(symbols=text)
        return self.from_doc(doc=doc)

    def from_doc(self, doc: Document) -> Document:
        self.logger.info("Predicting tokens...")
        tokens = self.tokenizer_predictor.predict(doc=doc)
        doc.annotate_layer(name=TokensFieldName, entities=tokens)

        self.logger.info("Predicting sentences...")
        sentences = self.sent_predictor.predict(doc=doc)
        doc.annotate_layer(name=SentencesFieldName, entities=sentences)
        return doc
