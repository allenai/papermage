"""
@kylel
"""


import os
from abc import abstractmethod
from pathlib import Path
from typing import Any

from papermage.magelib import Document


class Recipe:
    @abstractmethod
    def run(self, input: Any) -> Document:
        if isinstance(input, Path):
            if input.suffix == ".pdf":
                return self.from_pdf(pdf=input)
            if input.suffix == ".json":
                return self.from_json(doc=input)

            raise NotImplementedError("Filetype not yet supported.")

        if isinstance(input, Document):
            return self.from_doc(doc=input)

        if isinstance(input, str):
            if os.path.exists(input):
                input = Path(input)
                return self.run(input=input)
            else:
                return self.from_str(text=input)

        raise NotImplementedError("Document input not yet supported.")

    @abstractmethod
    def from_str(self, text: str) -> Document:
        raise NotImplementedError

    @abstractmethod
    def from_pdf(self, pdf: Path) -> Document:
        raise NotImplementedError

    @abstractmethod
    def from_json(self, json: str) -> Document:
        raise NotImplementedError

    @abstractmethod
    def from_doc(self, doc: Document) -> Document:
        raise NotImplementedError
