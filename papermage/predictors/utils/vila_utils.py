"""

@shannons

"""

import inspect
import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
    if not hasattr(token, field_name):
        return defaults
    field_value = getattr(token, field_name)
    if len(field_value) == 0 or field_value[0].id is None:
        return defaults
    return field_value[0].id


# util
def convert_document_page_to_pdf_dict(doc: Document, page_width: int, page_height: int) -> Dict[str, List]:
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
        for token in doc.tokens
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
