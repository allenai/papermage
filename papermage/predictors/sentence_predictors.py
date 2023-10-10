"""

Sentence Splitter using PySBD

@shannons, @kylel

"""

import itertools
from typing import List, Tuple

import numpy as np
import pysbd

from papermage.magelib import (
    Document,
    Entity,
    PagesFieldName,
    Span,
    TokensFieldName,
    WordsFieldName,
)
from papermage.predictors import BasePredictor
from papermage.utils.merge import cluster_and_merge_neighbor_spans


class PysbdSentencePredictor(BasePredictor):
    """Sentence Boundary based on Pysbd

    Examples:
        >>> doc: Document = parser.parse("path/to/pdf")
        >>> predictor = PysbdSentenceBoundaryPredictor()
        >>> sentence_spans = predictor.predict(doc)
        >>> doc.annotate(sentences=sentence_spans)
    """

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [TokensFieldName]  # type: ignore

    def __init__(self) -> None:
        self._segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)

    def split_token_based_on_sentences_boundary(self, words: List[str]) -> List[Tuple[int, int]]:
        """
        Split a list of words into a list of (start, end) indices, indicating
        the start and end of each sentence.
        Duplicate of https://github.com/allenai/VILA/\blob/dd242d2fcbc5fdcf05013174acadb2dc896a28c3/src/vila/dataset/preprocessors/layout_indicator.py#L14      # noqa: E501

        Returns: List[Tuple(int, int)]
            a list of (start, end) for token indices within each sentence
        """

        if len(words) == 0:
            return [(0, 0)]
        combined_words = " ".join(words)

        char2token_mask = np.zeros(len(combined_words), dtype=np.int64)

        acc_word_len = 0
        for idx, word in enumerate(words):
            word_len = len(word) + 1
            char2token_mask[acc_word_len : acc_word_len + word_len] = idx
            acc_word_len += word_len

        segmented_sentences = self._segmenter.segment(combined_words)
        sent_boundary = [(ele.start, ele.end) for ele in segmented_sentences]

        split = []
        token_id_start = 0
        for start, end in sent_boundary:
            token_id_end = char2token_mask[start:end].max()
            if end + 1 >= len(char2token_mask) or char2token_mask[end + 1] != token_id_end:
                token_id_end += 1  # (Including the end)
            split.append((token_id_start, token_id_end))
            token_id_start = token_id_end
        return split

    def _predict(self, doc: Document) -> List[Entity]:
        if hasattr(doc, WordsFieldName):
            words = [word.text for word in getattr(doc, WordsFieldName)]
            attr_name = WordsFieldName
            # `words` is preferred as it should has better reading
            # orders and text representation
        else:
            words = [token.text for token in doc.tokens]
            attr_name = TokensFieldName

        split = self.split_token_based_on_sentences_boundary(words)

        sentence_spans: List[Entity] = []
        for start, end in split:
            if end - start == 0:
                continue
            if end - start < 0:
                raise ValueError

            cur_spans = getattr(doc, attr_name)[start:end]

            all_token_spans = list(itertools.chain.from_iterable([ele.spans for ele in cur_spans]))
            results = cluster_and_merge_neighbor_spans(all_token_spans)
            sentence_spans.append(Entity(spans=results.merged))

        return sentence_spans
