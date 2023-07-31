"""

Base class for Predictors.

@shannons, @kylel

"""

from abc import abstractmethod
from typing import Any, Dict, List, Union

from papermage.types import Annotation, Document


class BasePredictor:
    @property
    @abstractmethod
    def REQUIRED_BACKENDS(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def REQUIRED_DOCUMENT_FIELDS(self):
        """Due to the dynamic nature of the document class as well the
        models, we require the model creator to provide a list of required
        fields in the document class. If not None, the predictor class
        will perform the check to ensure that the document contains all
        the specified fields.
        """
        raise NotImplementedError

    def _doc_field_checker(self, document: Document) -> None:
        if self.REQUIRED_DOCUMENT_FIELDS is not None:
            for field in self.REQUIRED_DOCUMENT_FIELDS:
                assert (
                    field in document.fields
                ), f"The input Document object {document} doesn't contain the required field {field}"

    def predict(self, document: Document) -> List[Annotation]:
        """For all the predictors, the input is a document object, and
        the output is a list of annotations.
        """
        self._doc_field_checker(document)
        return self._predict(document=document)

    @abstractmethod
    def _predict(document: Document) -> List[Annotation]:
        raise NotImplementedError
