"""

Uses a whitespace tokenizer on Document.symbols to predict which `tokens` were originally
part of the same segment/chunk (e.g. "few-shot" if tokenized as ["few", "-", "shot"]).

@kylel

"""

import os
from typing import List, Optional, Set, Tuple

import tokenizers

os.environ["TOKENIZERS_PARALLELISM"] = "false"


from papermage.magelib import Document, Entity, Metadata, Span
from papermage.predictors import BasePredictor


class HFWhitspaceTokenPredictor(BasePredictor):
    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return []

    _dictionary: Optional[Set[str]] = None

    def __init__(self) -> None:
        self.whitespace_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()

    def _predict(self, doc: Document) -> List[Entity]:
        self._doc_field_checker(doc)

        # 1) whitespace tokenization on symbols. each token is a nested tuple ('text', (start, end))
        ws_tokens: List[Tuple] = self.whitespace_tokenizer.pre_tokenize_str(doc.symbols)

        # 2) filter to just the chunks that are greater than 1 token. Reformat.
        # chunks = []
        # for text, (start, end) in ws_tokens:
        #     overlapping_tokens = document.find_overlapping(
        #         query=SpanGroup(spans=[Span(start=start, end=end)]),
        #         field_name=Tokens
        #     )
        #     if len(overlapping_tokens) > 1:
        #         chunk = SpanGroup(spans=[Span(start=start, end=end)], metadata=Metadata(text=text))
        #         chunks.append(chunk)
        chunks = []
        for text, (start, end) in ws_tokens:
            chunk = Entity(spans=[Span(start=start, end=end)], metadata=Metadata(text=text))
            chunks.append(chunk)
        return chunks
