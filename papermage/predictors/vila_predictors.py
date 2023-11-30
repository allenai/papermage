"""

This file rewrites the PDFPredictor classes in
https://github.com/allenai/VILA/blob/dd242d2fcbc5fdcf05013174acadb2dc896a28c3/src/vila/predictors.py#L1
to reduce the dependency on the VILA package.

@shannons, @kylel

"""
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from vila.predictors import LayoutIndicatorPDFPredictor, SimplePDFPredictor

from papermage.magelib import (
    BlocksFieldName,
    Document,
    Entity,
    Metadata,
    PagesFieldName,
    RowsFieldName,
    Span,
    TokensFieldName,
)
from papermage.predictors import BasePredictor

from .utils.vila_utils import (
    convert_document_page_to_pdf_dict,
    convert_sequence_tagging_to_spans,
)

# Two constants for the constraining the size of the page for
# inputs to the model.
# TODO: Move this to somewhere else.
MAX_PAGE_WIDTH = 1000
MAX_PAGE_HEIGHT = 1000

# these are the labels that are used in the VILA model
VILA_LABELS = [
    "Title",
    "Author",
    "Abstract",
    "Keywords",
    "Section",
    "Paragraph",
    "List",
    "Bibliography",
    "Equation",
    "Algorithm",
    "Figure",
    "Table",
    "Caption",
    "Header",
    "Footer",
    "Footnote",
]


class BaseSinglePageTokenClassificationPredictor(BasePredictor):
    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [PagesFieldName, TokensFieldName]

    DEFAULT_SUBPAGE_PER_RUN = 2  # TODO: Might remove this in the future for longformer-like models

    @property
    @abstractmethod
    def VILA_MODEL_CLASS(self):
        pass

    def __init__(self, predictor, subpage_per_run: Optional[int] = None):
        self.predictor = predictor

        # TODO: Make this more robust
        self.id2label = self.predictor.model.config.id2label
        self.label2id = self.predictor.model.config.label2id

        self.subpage_per_run = subpage_per_run or self.DEFAULT_SUBPAGE_PER_RUN

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        preprocessor=None,
        device: Optional[str] = None,
        subpage_per_run: Optional[int] = None,
        **preprocessor_config,
    ):
        predictor = cls.VILA_MODEL_CLASS.from_pretrained(
            model_path=model_name_or_path, preprocessor=preprocessor, device=device, **preprocessor_config
        )

        return cls(predictor, subpage_per_run)

    def _predict(self, doc: Document, subpage_per_run: Optional[int] = None) -> List[Entity]:
        page_prediction_results = []
        for page_id, page in enumerate(doc.pages):
            if page.tokens:
                page_width, page_height = doc.images[page_id].pilimage.size

                pdf_dict = self.preprocess(page, page_width=page_width, page_height=page_height)

                model_predictions = self.predictor.predict(
                    page_data=pdf_dict,
                    page_size=(page_width, page_height),
                    batch_size=subpage_per_run or self.subpage_per_run,
                    return_type="list",
                )

                assert len(model_predictions) == len(
                    page.tokens
                ), f"Model predictions and tokens are not the same length ({len(model_predictions)} != {len(page.tokens)}) for page {page_id}"

                page_prediction_results.extend(self.postprocess(page, model_predictions))

        return page_prediction_results

    def preprocess(self, page: Document, page_width: float, page_height: float) -> Dict:
        # In the latest vila implementations (after 0.4.0), the predictor will
        # handle all other preprocessing steps given the pdf_dict input format.

        return convert_document_page_to_pdf_dict(page, page_width=page_width, page_height=page_height)

    def postprocess(self, doc: Document, model_predictions) -> List[Entity]:
        token_prediction_spans = convert_sequence_tagging_to_spans(model_predictions)

        prediction_spans = []
        for token_start, token_end, label in token_prediction_spans:
            cur_spans = doc.tokens[token_start:token_end]

            start = min([ele.start for ele in cur_spans])
            end = max([ele.end for ele in cur_spans])
            sg = Entity(spans=[Span(start, end)], metadata=Metadata(label=label))
            prediction_spans.append(sg)
        return prediction_spans


class IVILATokenClassificationPredictor(BaseSinglePageTokenClassificationPredictor):
    VILA_MODEL_CLASS = LayoutIndicatorPDFPredictor

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List:
        base_reqs = [PagesFieldName, TokensFieldName]
        if self.predictor.preprocessor.config.agg_level == "row":
            base_reqs.append(RowsFieldName)
        elif self.predictor.preprocessor.config.agg_level == "block":
            base_reqs.append(BlocksFieldName)
        return base_reqs
