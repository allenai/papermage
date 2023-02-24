"""

Annotations are objects that are 'aware' of the Document. For example, imagine an entity
in a document; representing it as an Annotation data type would allow you to access the
Document object directly from within the Entity itself.

@kylel

"""

import logging

from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from papermage.types import Span
from papermage.types import Box
from papermage.types.metadata import Metadata

if TYPE_CHECKING:
    from mmda.types.document import Document

__all__ = ["Annotation", "Entity", "Relation"]



class Annotation:
    """Annotation allows us to layer different model predictions on a single document."""

    @abstractmethod
    def __init__(self):
        self._id: Optional[int] = None
        self._doc: Optional['Document'] = None
        logging.warning('Unless testing or developing, we dont recommend creating Annotations '
                        'manually. Annotations need to store things like `id` and references '
                        'to a `Document` to be valuable. These are all handled automatically in '
                        '`Parsers` and `Predictors`.')

    @property
    def doc(self) -> Optional['Document']:
        return self._doc

    @doc.setter
    def doc(self, doc: Document) -> None:
        """This method attaches a Document to this Annotation, allowing the Annotation
                to access things beyond itself within the Document (e.g. neighbors)"""
        if self.doc:
            raise AttributeError("This annotation already has an attached document")
        self._doc = doc

    @property
    def id(self) -> Optional[int]:
        return self._id

    @id.setter
    def id(self, id: int) -> None:
        """This method assigns an ID to an Annotation. Requires a Document to be attached
        to this Annotation. ID basically gives the Annotation itself awareness of its
        position within the broader Document."""
        if self.id:
            raise AttributeError("This annotation already has an ID")
        if not self.doc:
            raise AttributeError('This annotation is missing a Document')
        self._id = id

    @abstractmethod
    def to_json(self) -> Dict:
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, annotation_dict: Dict) -> "Annotation":
        pass

    def __getattr__(self, field: str) -> List["Annotation"]:
        """This method """
        if self.doc is None:
            raise ValueError("This annotation is not attached to a document")

        if field in self.doc.fields:
            return self.doc.find_overlapping(self, field)

        if field in self.doc.fields:
            return self.doc.find_overlapping(self, field)

        return self.__getattribute__(field)


class Entity(Annotation):
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

    @property
    def text(self) -> str:
        maybe_text = self.metadata.get("text", None)
        if maybe_text is None:
            return " ".join(self.symbols)
        return maybe_text

    @text.setter
    def text(self, text: Union[str, None]) -> None:
        self.metadata.text = text

    def to_json(self) -> Dict:
        entity_dict = dict(
            spans=[span.to_json() for span in self.spans],
            boxes=[box.to_json() for box in self.boxes] if self.boxes else None,
            metadata=self.metadata.to_json()
        )
        # only serialize non-null values
        return {k: v for k, v in entity_dict.items() if v is not None}

    @classmethod
    def from_json(cls, entity_dict: Dict) -> "Entity":
        return cls(
            spans=[Span.from_json(span_json=span_dict) for span_dict in entity_dict["spans"]],
            boxes=[Box.from_json(box_dict=box_dict) for box_dict in entity_dict['boxes']]
            if entity_dict.get('boxes') else None,
            metadata=Metadata.from_json(entity_dict['metadata'])
            if entity_dict.get('metadata') else None
        )

    @property
    def start(self) -> int:
        return min([span.start for span in self.spans])

    @property
    def end(self) -> int:
        return max([span.end for span in self.spans])


class Relation(Annotation):
    def __init__(
            self,
            source: Entity,
            target: Entity,
            metadata: Optional[Metadata] = None
    ):
        self.source = source
        self.target = target
        self.metadata = metadata if metadata else Metadata
        super().__init__()

    def to_json(self) -> Dict:
        relation_dict = dict(
            source=Entity.id,
            target=Entity.id,
            metadata=self.metadata.to_json()
        )
        # only serialize non-null values
        return {k: v for k, v in relation_dict.items() if v is not None}

    @classmethod
    def from_json(cls, relation_dict: Dict, doc: Document) -> "Relation":
        return cls(
            source=None,
            target=None,
            metadata=Metadata.from_json(relation_dict['metadata'])
            if relation_dict.get('metadata') else None
        )
