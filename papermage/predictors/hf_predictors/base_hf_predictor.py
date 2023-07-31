from abc import abstractmethod
from typing import Union, List, Dict, Any

from transformers import AutoTokenizer, AutoConfig, AutoModel

from papermage.types import Annotation, Document
from papermage.predictors.base_predictors.base_predictor import BasePredictor


class BaseHFPredictor(BasePredictor):
    REQUIRED_BACKENDS = ["transformers", "torch"]

    def __init__(self, model: Any, config: Any, tokenizer: Any):

        self.model = model
        self.config = config
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *args, **kwargs):
        config = AutoConfig.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(
            model_name_or_path, config=config, *args, **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(model, config, tokenizer)

    @abstractmethod
    def preprocess(self, document: Document) -> List:
        """Convert the input document into the format that is required
        by the model.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, model_outputs: Any) -> List[Annotation]:
        """Convert the model outputs into the Annotation format"""
        raise NotImplementedError