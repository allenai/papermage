"""

An annotated "unit" on a Document.

"""

from typing import TYPE_CHECKING, Dict, List, Optional, Union

from papermage.types import Span
from papermage.types import Box


class Entity:
    def __init__(
            self,
            spans: List[Span],
            boxes: Optional[List[Box]] = None,
            metadata: Optional[Metadata] = None
    ):
        self.spans = spans
        self.boxes = boxes
        self.metadata = metadata if metadata else Metadata
        super().__init__()

    @property
    def symbols(self) -> List[str]:
        if self.doc is None:
            raise ValueError(f'No document attached.')
        return [self.doc.symbols[span.start: span.end] for span in self.spans]
