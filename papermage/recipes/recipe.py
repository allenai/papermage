"""
@kylel
"""

from abc import abstractmethod

from papermage.types import Document


class Recipe:
    @abstractmethod
    def from_path(self, pdfpath: str) -> Document:
        raise NotImplementedError

    @abstractmethod
    def from_doc(self, doc: Document) -> Document:
        raise NotImplementedError
