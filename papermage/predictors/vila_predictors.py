"""

This file rewrites the PDFPredictor classes in
https://github.com/allenai/VILA/blob/dd242d2fcbc5fdcf05013174acadb2dc896a28c3/src/vila/predictors.py#L1
to reduce the dependency on the VILA package.

@shannons, @kylel

"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import inspect
import itertools
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from tqdm import tqdm
from vila.predictors import LayoutIndicatorPDFPredictor

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


# util
def columns_used_in_model_inputs(model):
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    return signature_columns


# util
def normalize_bbox(
    bbox,
    page_width,
    page_height,
    target_width,
    target_height,
):
    """
    Normalize bounding box to the target size.
    """

    x1, y1, x2, y2 = bbox

    # Right now only execute this for only "large" PDFs
    # TODO: Change it for all PDFs
    if page_width > target_width or page_height > target_height:
        x1 = float(x1) / page_width * target_width
        x2 = float(x2) / page_width * target_width
        y1 = float(y1) / page_height * target_height
        y2 = float(y2) / page_height * target_height

    return (x1, y1, x2, y2)


# util
def shift_index_sequence_to_zero_start(sequence):
    """
    Shift a sequence to start at 0.
    """
    sequence_start = min(sequence)
    return [i - sequence_start for i in sequence]


# util
def get_visual_group_id(token: Entity, field_name: str, defaults=-1) -> int:
    field_value = token.intersect_by_span(name=field_name)
    if not field_value:
        return defaults
    if len(field_value) == 0 or field_value[0].id is None:
        return defaults
    return field_value[0].id

    # if not hasattr(token, field_name):
    #     return defaults
    # field_value = getattr(token, field_name)
    # if len(field_value) == 0 or field_value[0].id is None:
    #     return defaults
    # return field_value[0].id


# util
def convert_document_page_to_pdf_dict(page: Entity, page_width: int, page_height: int) -> Dict[str, List]:
    """Convert a document to a dictionary of the form:
        {
            'words': ['word1', 'word2', ...],
            'bbox': [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
            'block_ids': [0, 0, 0, 1 ...],
            'line_ids': [0, 1, 1, 2 ...],
            'labels': [0, 0, 0, 1 ...], # could be empty
        }

    Args:
        document (Document):
            The input document object
        page_width (int):
            Typically the transformer model requires to use
            the absolute coordinates for encoding the coordinates.
            Set the correspnding page_width and page_height to convert the
            relative coordinates to the absolute coordinates.
        page_height (int):
            Typically the transformer model requires to use
            the absolute coordinates for encoding the coordinates.
            Set the correspnding page_width and page_height to convert the
            relative coordinates to the absolute coordinates.

    Returns:
        Dict[str, List]: The pdf_dict object
    """

    token_data = [
        (
            token.symbols_from_spans[0],  # words
            token.boxes[0].to_absolute(page_width=page_width, page_height=page_height).xy_coordinates,  # bbox
            get_visual_group_id(token, RowsFieldName, -1),  # line_ids
            get_visual_group_id(token, BlocksFieldName, -1),  # block_ids
        )
        for token in page.intersect_by_span(name=TokensFieldName)
    ]

    words, bbox, line_ids, block_ids = (list(l) for l in zip(*token_data))
    line_ids = shift_index_sequence_to_zero_start(line_ids)
    block_ids = shift_index_sequence_to_zero_start(block_ids)

    labels = [None] * len(words)
    # TODO: We provide an empty label list.

    return {
        "words": words,
        "bbox": bbox,
        "block_ids": block_ids,
        "line_ids": line_ids,
        "labels": labels,
    }


# util
def convert_sequence_tagging_to_spans(
    token_prediction_sequence: List,
) -> List[Tuple[int, int, int]]:
    """For a sequence of token predictions, convert them to spans
    of consecutive same predictions.

    Args:
        token_prediction_sequence (List)

    Returns:
        List[Tuple[int, int, int]]: A list of (start, end, label)
            of consecutive prediction of the same label.
    """
    prev_len = 0
    spans = []
    for gp, seq in itertools.groupby(token_prediction_sequence):
        cur_len = len(list(seq))
        spans.append((prev_len, prev_len + cur_len, gp))
        prev_len = prev_len + cur_len
    return spans


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

            # skip pages without tokens
            tokens_on_page = page.intersect_by_span(name=TokensFieldName)
            if not tokens_on_page:
                continue

            page_width, page_height = doc.images[page_id].pilimage.size

            pdf_dict = self.preprocess(page=page, page_width=page_width, page_height=page_height)

            model_predictions = self.predictor.predict(
                page_data=pdf_dict,
                page_size=(page_width, page_height),
                batch_size=subpage_per_run or self.subpage_per_run,
                return_type="list",
            )

            assert len(model_predictions) == len(
                tokens_on_page
            ), f"Model predictions and tokens are not the same length ({len(model_predictions)} != {len(tokens_on_page)}) for page {page_id}"

            page_prediction_results.extend(self.postprocess(page=page, model_predictions=model_predictions))

        return page_prediction_results

    def preprocess(self, page: Entity, page_width: float, page_height: float) -> Dict:
        # In the latest vila implementations (after 0.4.0), the predictor will
        # handle all other preprocessing steps given the pdf_dict input format.

        return convert_document_page_to_pdf_dict(page=page, page_width=page_width, page_height=page_height)

    def postprocess(self, page: Entity, model_predictions) -> List[Entity]:
        token_prediction_spans = convert_sequence_tagging_to_spans(model_predictions)

        prediction_spans = []
        for token_start, token_end, label in token_prediction_spans:
            cur_spans = page.intersect_by_span(name=TokensFieldName)[token_start:token_end]

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
