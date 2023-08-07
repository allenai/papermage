"""

Annotations are objects that are 'aware' of the Document. For example, imagine an entity
in a document; representing it as an Annotation data type would allow you to access the
Document object directly from within the Entity itself.

@kylel

"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .document import Document


class Annotation:
    """Represent a "unit" (e.g. highlighted span, drawn boxes) layered on a Document."""

    @abstractmethod
    def __init__(self) -> None:
        self._id: Optional[int] = None
        self._doc: Optional["Document"] = None

    @property
    def doc(self) -> Optional["Document"]:
        return self._doc

    @doc.setter
    def doc(self, doc: Optional["Document"]) -> None:
        """This method attaches a Document to this Annotation, allowing the Annotation
        to access things beyond itself within the Document (e.g. neighboring annotations)"""
        if self.doc and doc:
            raise AttributeError(
                "Already has an attached Document. Since Annotations should be"
                "specific to a given Document, we recommend creating a new"
                "Annotation from scratch and then attaching your Document."
            )
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
            raise AttributeError(f"This Annotation already has an ID: {self.id}")
        if not self.doc:
            raise AttributeError("This Annotation is missing a Document")
        self._id = id

    @abstractmethod
    def to_json(self) -> Union[Dict, List]:
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, annotation_json: Union[Dict, List]) -> "Annotation":
        pass

    def __getattr__(self, field: str) -> List["Annotation"]:
        """This Overloading is convenient syntax since the `entity.layer` operation is intuitive for folks."""
        try:
            return self.find_by_span(field=field)
        except ValueError:
            # maybe users just want some attribute of the Annotation object
            return self.__getattribute__(field)

    def find_by_span(self, field: str) -> List["Annotation"]:
        """This method allows you to access overlapping Annotations
        within the Document based on Span"""
        if self.doc is None:
            raise ValueError("This annotation is not attached to a document")

        if field in self.doc.fields:
            return self.doc.find_by_span(self, field)
        else:
            raise ValueError(f"Field {field} not found in Document")

    def find_by_box(self, field: str) -> List["Annotation"]:
        """This method allows you to access overlapping Annotations
        within the Document based on Box"""

        if self.doc is None:
            raise ValueError("This annotation is not attached to a document")

        if field in self.doc.fields:
            return self.doc.find_by_box(self, field)
        else:
            raise ValueError(f"Field {field} not found in Document")
