"""

An annotated "unit" in a Layer.

"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from .box import Box
from .image import Image
from .metadata import Metadata
from .names import TokensFieldName
from .span import Span

if TYPE_CHECKING:
    from .layer import Layer


class Entity:
    __slots__ = ["spans", "boxes", "images", "metadata", "_id", "_layer"]

    def __init__(
        self,
        spans: Optional[List[Span]] = None,
        boxes: Optional[List[Box]] = None,
        images: Optional[List[Image]] = None,
        metadata: Optional[Metadata] = None,
    ):
        if not spans and not boxes:
            raise ValueError(f"At least one of `spans` or `boxes` must be set.")
        self.spans = spans if spans is not None else []
        self.boxes = boxes if boxes is not None else []
        self.images = images if images is not None else []
        self.metadata = metadata if metadata else Metadata()
        # TODO: it's confusing that `id` is both reading order as well as direct reference
        # TODO: maybe Layer() should house reading order, and Entity() should have a unique ID
        # TODO: hashing would be interesting, but Metadata() is allowed to mutate so that's a problem
        self._id = None
        self._layer = None

    def __repr__(self):
        if self.layer:
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
    def layer(self) -> Optional["Layer"]:
        return self._layer

    @layer.setter
    def layer(self, layer: Optional["Layer"]) -> None:
        """This method attaches a Layer to this Entity, allowing the Entity
        to access things beyond itself within the Layer (e.g. neighboring Entities)"""
        if self.layer and layer:
            raise AttributeError(
                "Already has an attached Layer. Since Entity should correspond"
                "to only a specific Layer, we recommend creating a new"
                "Entity from scratch and then attaching your Layer."
            )
        self._layer = layer

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
        if not self.layer:
            raise AttributeError("This Entity is missing a Document")
        self._id = id

    def __getattr__(self, name: str) -> List["Entity"]:
        """This Overloading is convenient syntax since the `entity.layer` operation is intuitive for folks."""
        # add method deprecation warning
        logger = logging.getLogger(__name__)
        logger.warning(
            "Entity.__getattr__ is deprecated due to ambiguity and will be removed in a future release."
            "Please use Entity.intersect_by_span or Entity.intersect_by_box instead."
        )
        try:
            if len(self.spans) > 0:
                intersection = self.intersect_by_span(name=name)
                if len(intersection) == 0 and len(self.boxes) > 0:
                    intersection = self.intersect_by_box(name=name)
                return intersection
            else:
                return self.intersect_by_box(name=name)
        except ValueError:
            # maybe users just want some attribute of the Entity object
            return self.__getattribute__(name)

    def intersect_by_span(self, name: str) -> List["Entity"]:
        """This method allows you to access overlapping Entities
        within the Document based on Span"""
        if self.layer is None:
            raise ValueError("This Entity is not attached to a Layer")

        if self.layer.doc is None:
            raise ValueError("This Entity's Layer is not attached to a Document")

        return self.layer.doc.intersect_by_span(query=self, name=name)

    def intersect_by_box(self, name: str) -> List["Entity"]:
        """This method allows you to access overlapping Entities
        within the Document based on Box"""
        if self.layer is None:
            raise ValueError("This Entity is not attached to a Layer")

        if self.layer.doc is None:
            raise ValueError("This Entity's Layer is not attached to a Document")

        return self.layer.doc.intersect_by_box(query=self, name=name)

    @property
    def start(self) -> Union[int, float]:
        return min([span.start for span in self.spans]) if len(self.spans) > 0 else float("-inf")

    @property
    def end(self) -> Union[int, float]:
        return max([span.end for span in self.spans]) if len(self.spans) > 0 else float("inf")

    @property
    def symbols_from_spans(self) -> List[str]:
        if self.layer is None:
            raise ValueError("This Entity is not attached to a Layer")

        if self.layer.doc is None:
            raise ValueError("This Entity's Layer is not attached to a Document")

        if self.layer.doc.symbols is None:
            raise ValueError("This Entity's Document is missing symbols")

        return [self.layer.doc.symbols[span.start : span.end] for span in self.spans]

    @property
    def symbols_from_boxes(self) -> List[str]:
        if self.layer is None:
            raise ValueError("This Entity is not attached to a Layer")

        if self.layer.doc is None:
            raise ValueError("This Entity's Layer is not attached to a Document")

        if self.layer.doc.symbols is None:
            raise ValueError("This Entity's Document is missing symbols")

        matched_tokens = self.intersect_by_box(name=TokensFieldName)
        return [self.layer.doc.symbols[span.start : span.end] for t in matched_tokens for span in t.spans]

    @property
    def text(self) -> str:
        # return stored metadata
        maybe_text = self.metadata.get("text", None)
        if maybe_text:
            return maybe_text
        # return derived from symbols
        if self.symbols_from_spans:
            return " ".join(self.symbols_from_spans).replace("\n", " ")
        # return derived from boxes and tokens
        if self.symbols_from_boxes:
            return " ".join(self.symbols_from_boxes).replace("\n", " ")
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
