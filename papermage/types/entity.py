"""

An annotated "unit" on a Document.

"""

from typing import TYPE_CHECKING, Dict, List, Optional, Union

from papermage.types import Annotation, Box, Image, Metadata, Span

if TYPE_CHECKING:
    from papermage.types.document import Document


class Entity(Annotation):
    def __init__(
        self,
        spans: Optional[List[Span]] = None,
        boxes: Optional[List[Box]] = None,
        images: Optional[List[Image]] = None,
        metadata: Optional[Metadata] = None,
    ):
        if not spans and not boxes:
            raise ValueError(f"At least one of `spans` or `boxes` must be set.")
        self.spans = spans if spans else []
        self.boxes = boxes if boxes else []
        self.images = images if images else []
        self.metadata = metadata if metadata else Metadata()
        super().__init__()

    def __repr__(self):
        if self.doc and self.spans:
            symbols: List[str] = [self.doc.symbols[span.start : span.end] for span in self.spans]
            return f"Entity\tSymbols\t{symbols}"
        return f"Entity{self.to_json()}"

    def to_json(self) -> Dict:
        entity_dict = dict(
            spans=[span.to_json() for span in self.spans],
            boxes=[box.to_json() for box in self.boxes],
            metadata=self.metadata.to_json(),
        )
        # only serialize non-null/non-empty values
        return {k: v for k, v in entity_dict.items() if v}

    @classmethod
    def from_json(cls, entity_json: Dict) -> "Entity":
        return cls(
            spans=[Span.from_json(span_json=span_json) for span_json in entity_json.get("spans", [])],
            boxes=[Box.from_json(box_json=box_json) for box_json in entity_json.get("boxes", [])],
            metadata=Metadata.from_json(entity_json.get("metadata", {})),
        )

    @property
    def start(self) -> Union[int, float]:
        return min([span.start for span in self.spans]) if len(self.spans) > 0 else float("-inf")

    @property
    def end(self) -> Union[int, float]:
        return max([span.end for span in self.spans]) if len(self.spans) > 0 else float("inf")

    @property
    def symbols(self) -> List[str]:
        if self.doc is not None:
            return [self.doc.symbols[span.start : span.end] for span in self.spans]
        else:
            return []

    @property
    def text(self) -> str:
        maybe_text = self.metadata.get("text", None)
        if maybe_text is None:
            return " ".join(self.symbols)
        return maybe_text

    @text.setter
    def text(self, text: Union[str, None]) -> None:
        self.metadata.text = text

    def __iter__(self):
        """By default, iterate over the spans"""
        yield from self.spans

    def __lt__(self, other: "Entity"):
        if self.id and other.id:
            return self.id < other.id
        else:
            return self.start < other.start
