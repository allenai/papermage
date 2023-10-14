"""

Base class for Predictors.

@shannons, @kylel

"""

from abc import abstractmethod
from typing import Any, Dict, List, Union

from papermage.magelib import Document, Entity


class BasePredictor:
    @property
    @abstractmethod
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        """Due to the dynamic nature of the document class as well the
        models, we require the model creator to provide a list of required
        fields in the document class. If not None, the predictor class
        will perform the check to ensure that the document contains all
        the specified fields.
        """
        raise NotImplementedError

    def _doc_field_checker(self, doc: Document) -> None:
        if self.REQUIRED_DOCUMENT_FIELDS is not None:
            for field in self.REQUIRED_DOCUMENT_FIELDS:
                assert (
                    field in doc.layers
                ), f"The input Document object {doc} doesn't contain the required field {field}"

    def predict(self, doc: Document) -> List[Entity]:
        """For all the predictors, the input is a document object, and
        the output is a list of annotations.
        """
        self._doc_field_checker(doc)
        return self._predict(doc=doc)

    @abstractmethod
    def _predict(self, doc: Document) -> List[Entity]:
        raise NotImplementedError
