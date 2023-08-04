"""

Protocol for creating token streams from a document

@kylel, shannons, bnewm0609

"""

from abc import abstractmethod
from typing import Protocol

from papermage.magelib import Document


class Parser(Protocol):
    @abstractmethod
    def parse(self, input_pdf_path: str, **kwargs) -> Document:
        """Given an input PDF return a Document with at least symbols

        Args:
            input_pdf_path (str): Path to the input PDF to process

        Returns:
            Document: Depending on parser support at least symbols in the PDF
        """
        raise NotImplementedError
