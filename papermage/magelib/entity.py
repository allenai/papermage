"""

An annotated "unit" on a Document.

"""

from typing import TYPE_CHECKING, Dict, List, Optional, Union

from .box import Box
from .image import Image
from .metadata import Metadata
from .span import Span

if TYPE_CHECKING:
    from .document import Document


class Entity:
    __slots__ = ["spans", "boxes", "images", "metadata", "_id", "_doc"]

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
        self._id = None
        self._doc = None

    def __repr__(self):
        if self.doc:
            return f"Annotated Entity:\tID: {self.id}\tSpans: {True if self.spans else False}\tBoxes: {True if self.boxes else False}\tText: {self.text}"
        return f"Unannotated Entity: {self.to_json()}"

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
    def doc(self) -> Optional["Document"]:
        return self._doc

    @doc.setter
    def doc(self, doc: Optional["Document"]) -> None:
        """This method attaches a Document to this Entity, allowing the Entity
        to access things beyond itself within the Document (e.g. neighboring entities)"""
        if self.doc and doc:
            raise AttributeError(
                "Already has an attached Document. Since Entity should be"
                "specific to a given Document, we recommend creating a new"
                "Entity from scratch and then attaching your Document."
            )
        self._doc = doc

    @property
    def id(self) -> Optional[int]:
        return self._id

    @id.setter
    def id(self, id: int) -> None:
        """This method assigns an ID to an Entity. Requires a Document to be attached
        to this Entity. ID basically gives the Entity itself awareness of its
        position within the broader Document."""
        if self.id:
            raise AttributeError(f"This Entity already has an ID: {self.id}")
        if not self.doc:
            raise AttributeError("This Entity is missing a Document")
        self._id = id

    def __getattr__(self, field: str) -> List["Entity"]:
        """This Overloading is convenient syntax since the `entity.layer` operation is intuitive for folks."""
        try:
            return self.find_by_span(field=field)
        except ValueError:
            # maybe users just want some attribute of the Entity object
            return self.__getattribute__(field)

    @property
    def start(self) -> Union[int, float]:
        return min([span.start for span in self.spans]) if len(self.spans) > 0 else float("-inf")

    @property
    def end(self) -> Union[int, float]:
        return max([span.end for span in self.spans]) if len(self.spans) > 0 else float("inf")

    @property
    def symbols_from_spans(self) -> List[str]:
        if self.doc is not None:
            return [self.doc.symbols[span.start : span.end] for span in self.spans]
        else:
            return []

    @property
    def symbols_from_boxes(self) -> List[str]:
        if self.doc is not None:
            matched_tokens = self.doc.find_by_box(query=self, field_name="tokens")
            return [self.doc.symbols[span.start : span.end] for t in matched_tokens for span in t.spans]
        else:
            return []

    @property
    def text(self) -> str:
        # return stored metadata
        maybe_text = self.metadata.get("text", None)
        if maybe_text:
            return maybe_text
        # return derived from symbols
        if self.symbols_from_spans:
            return " ".join(self.symbols_from_spans)
        # return derived from boxes and tokens
        if self.symbols_from_boxes:
            return " ".join(self.symbols_from_boxes)
        return ""

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

    def find_by_span(self, field: str) -> List["Entity"]:
        """This method allows you to access overlapping Entities
        within the Document based on Span"""
        if self.doc is None:
            raise ValueError("This entity is not attached to a document")

        if field in self.doc.fields:
            return self.doc.find_by_span(self, field)
        else:
            raise ValueError(f"Field {field} not found in Document")

    def find_by_box(self, field: str) -> List["Entity"]:
        """This method allows you to access overlapping Entities
        within the Document based on Box"""

        if self.doc is None:
            raise ValueError("This entity is not attached to a document")

        if field in self.doc.fields:
            return self.doc.find_by_box(self, field)
        else:
            raise ValueError(f"Field {field} not found in Document")
