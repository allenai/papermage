"""

@kylel, benjaminn

"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers
from smashed.interfaces.simple import (
    FixedBatchSizeMapper,
    FromTokenizerListCollatorMapper,
    Python2TorchMapper,
    TokenizerMapper,
    UnpackingMapper,
)

from papermage.predictors.base_predictor import BasePredictor
from papermage.types import Annotation, Document, Entity, Metadata, Span


class EntityClassificationBatch:
    def __init__(
        self,
        input_ids: List[List[int]],
        attention_mask: List[List[int]],
        entity_ids: List[List[Optional[int]]],
        context_id: List[int],
    ):
        if not (len(input_ids) == len(attention_mask) == len(entity_ids) == len(context_id)):
            raise ValueError("Inputs to batch arent same length")

        self.batch_size = len(input_ids)
        if not (
            [len(example) for example in input_ids]
            == [len(example) for example in attention_mask]
            == [len(example) for example in entity_ids]
        ):
            raise ValueError("Examples in batch arent same length")

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.entity_ids = entity_ids
        self.context_id = context_id


class EntityClassificationPrediction:
    def __init__(self, context_id: int, entity_id: int, label: str, score: float):
        self.context_id = context_id
        self.entity_id = entity_id
        self.label = label
        self.score = score


class EntityClassificationPredictor(BasePredictor):
    """
    This is a generic wrapper around Huggingface Token Classification models.

    First, we need an `entity_name` which defines the Document field that we will
    receive a prediction. For example, if `entity_name` is 'tokens', then we expect
    to classify every Document.token.  But technically, `entity_name` could be anything,
    such as `words` or `rows` or any Entity.

    Second, we need a `context_name` which defines the Document field that we will
    treat as an "example" that we want to run our model over. For example, if
    `context_name` is 'pages', then we'll loop over each page, running our classifier
    over all the 'tokens' in each page. If the `context_name` is `bib_entries`, then we'll
    loop over each bib entry, running our classifier over the 'tokens' in each page.

    The key consequence of defining a `context_name` is, when the model constructs batches
    of sequences that fit within the Huggingface transformer's window, it will *not*
    mix sequences from different contexts into the same batch. # BEN: I don't get this.

    @kylel
    """

    REQUIRED_BACKENDS = ["transformers", "torch", "smashed"]

    _INPUT_FIELD_NAME = "inputs"
    _CONTEXT_ID = "context_id"

    _HF_RESERVED_INPUT_IDS = "input_ids"
    _HF_RESERVED_ATTN_MASK = "attention_mask"
    _HF_RESERVED_WORD_IDS = "word_ids"
    _HF_RESERVED_WORD_PAD_VALUE = -1

    def __init__(
        self,
        model: Any,
        config: Any,
        tokenizer: Any,
        entity_name: str,
        context_name: str,
        batch_size: Optional[int] = 2,
        device: Optional[str] = "cpu",
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.entity_name = entity_name
        self.context_name = context_name
        self.batch_size = batch_size
        self.device = device

        # For some reason, the roberta max_position_embeddings is larger than the actual model context window
        # so use the model_max_length instead. (We can't replace model_max_length with max_position_embeddings
        # in general, because model_max_legnth will be set to a very large number if the model length is not
        # explicitly included in the tokenizer (like in the case of alleani/scibert). As a compromise, check
        # if the model_max_legnth is big, and it's too big, then use the max_position_embeddings as the context
        # window size.
        if tokenizer.model_max_length > 1e29:
            max_length = self.config.max_position_embeddings
        else:
            max_length = tokenizer.model_max_length

        # handles tokenization, sliding window, truncation, subword to input word mapping, etc.
        self.tokenizer_mapper = TokenizerMapper(
            input_field=self._INPUT_FIELD_NAME,
            tokenizer=tokenizer,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_word_ids=True,
        )
        # since input data is automatically chunked into segments (e.g. 512 length),
        # each example <dict> actually becomes many input sequences.
        # this mapper unpacks all of this into one <dict> per input sequence.
        # we set `repeat` because we want other fields (`context_id`) to repeat across sequnces
        # E.g. {"input_ids": [[1,2,3]]} => {"input_ids": [1,2,3]}
        self.unpacking_mapper = UnpackingMapper(
            fields_to_unpack=[
                self._HF_RESERVED_INPUT_IDS,
                self._HF_RESERVED_ATTN_MASK,
                self._HF_RESERVED_WORD_IDS,
            ],
            ignored_behavior="repeat",
        )
        # at the end of this, each <dict> contains <lists> of length `batch_size`
        # where each element is variable length within the `max_length` limit.
        # `keep_last` controls whether we want partial batches, which we always do
        # for token classification (i.e. we dont want to miss any token!)
        self.batch_size_mapper = FixedBatchSizeMapper(batch_size=batch_size, keep_last=True)
        # this performs padding so all sequences in a batch are of same length
        self.list_collator_mapper = FromTokenizerListCollatorMapper(
            tokenizer=tokenizer,
            pad_to_length=None,  # keeping this `None` is best because of dynamic padding
            fields_pad_ids={
                self._HF_RESERVED_WORD_IDS: self._HF_RESERVED_WORD_PAD_VALUE,
                "labels": -100,
            },
        )
        # this casts python Dict[List] into tensors.  if using GPU, would do `device='gpu'`
        self.python_to_torch_mapper = Python2TorchMapper(device=device)
        # combining everything
        self.preprocess_mapper = self.tokenizer_mapper >> self.unpacking_mapper >> self.batch_size_mapper

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        entity_name: str,
        context_name: str,
        batch_size: Optional[int] = 2,
        device: Optional[str] = "cpu",
        *args,
        **kwargs,
    ):
        """If `model_name_or_path` is a path, should be a directory
        containing `vocab.txt`, `config.json`, and `pytorch_model.bin`

        NOTE: slightly annoying, but if loading in this way, the `_name_or_path`
        in `model.config` != `config`.
        """
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, *args, **kwargs
        )

        if "add_prefix_space" in kwargs:
            kwargs.pop("add_prefix_space")

        config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, *args, **kwargs
        )
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path, *args, **kwargs
        )
        predictor = cls(
            model=model,
            config=config,
            tokenizer=tokenizer,
            entity_name=entity_name,
            context_name=context_name,
            batch_size=batch_size,
            device=device,
        )
        return predictor

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [self.context_name, self.entity_name]

    def preprocess(self, doc: Document, context_name: str) -> List[EntityClassificationBatch]:
        """Processes document into whatever makes sense for the Huggingface model"""
        # (1) get it into a dictionary format that Smashed expects

        dataset = [
            {
                self._INPUT_FIELD_NAME: [entity.text for entity in getattr(context, self.entity_name)],
                self._CONTEXT_ID: i,
            }
            for i, context in enumerate(getattr(doc, context_name))
        ]

        # (2) apply Smashed
        batch_dicts = self.preprocess_mapper.map(dataset=dataset)

        # (3) convert dicts to objects
        return [
            # slightly annoying, but the names `input_ids`, `attention_mask` and `word_ids` are
            # reserved and produced after tokenization, which is why hard-coded here.
            EntityClassificationBatch(
                input_ids=batch_dict[self._HF_RESERVED_INPUT_IDS],
                attention_mask=batch_dict[self._HF_RESERVED_ATTN_MASK],
                entity_ids=batch_dict[self._HF_RESERVED_WORD_IDS],
                context_id=batch_dict[self._CONTEXT_ID],
            )
            for batch_dict in batch_dicts
        ]

    def postprocess(
        self, doc: Document, context_name: str, preds: List[EntityClassificationPrediction]
    ) -> List[Annotation]:
        """This function handles a bunch of nonsense that happens with Huggingface models &
        how we processed the data.  Namely:

        Because Huggingface might drop tokens during the course of tokenization
        we need to organize our predictions into a Lookup <dict> and cross-reference
        with the original input SpanGroups to make sure they all got classified.
        """
        # (1) organize predictions into a Lookup at the (Context, SpanGroup) level.
        context_id_to_entity_id_to_pred: Dict[int, Dict[int, EntityClassificationPrediction]] = defaultdict(dict)
        for prediction in preds:
            context_id_to_entity_id_to_pred[prediction.context_id][prediction.entity_id] = prediction

        # (2) iterate through original data to check against that Lookup
        annotations: List[Annotation] = []
        for i, context in enumerate(getattr(doc, context_name)):
            for j, entity in enumerate(getattr(context, self.entity_name)):
                pred: Optional[EntityClassificationPrediction] = context_id_to_entity_id_to_pred[i].get(j, None)
                # TODO: double-check whether this deepcopy is needed...
                new_metadata = Metadata.from_json(entity.metadata.to_json())
                if pred is not None:
                    new_metadata.label = pred.label
                    new_metadata.score = pred.score
                else:
                    new_metadata.label = None
                    new_metadata.score = None
                new_entity = Entity(
                    spans=entity.spans,
                    boxes=entity.boxes,
                    metadata=new_metadata,
                )
                annotations.append(new_entity)
        return annotations

    def _predict(self, doc: Document) -> List[Annotation]:
        # (1) Make batches
        batches: List[EntityClassificationBatch] = self.preprocess(doc=doc, context_name=self.context_name)

        # (2) Predict each batch.
        preds: List[EntityClassificationPrediction] = []
        for batch in batches:
            for pred in self._predict_batch(batch=batch):
                preds.append(pred)

        # (3) Postprocess into proper Annotations
        annotations = self.postprocess(doc=doc, context_name=self.context_name, preds=preds)
        return annotations

    def _predict_batch(
        self,
        batch: EntityClassificationBatch,
        device: str = "cpu",
    ) -> List[EntityClassificationPrediction]:
        #
        #   preprocessing!!  (padding & tensorification)
        #
        pytorch_batch = self.python_to_torch_mapper.transform(
            data=self.list_collator_mapper.transform(
                data={
                    self._HF_RESERVED_INPUT_IDS: batch.input_ids,
                    self._HF_RESERVED_ATTN_MASK: batch.attention_mask,
                }
            )
        )
        #
        #   inference!! (preferably on gpu)
        #
        # TODO: add something here for gpu migration
        pytorch_batch = {k: v.to(device) for k, v in pytorch_batch.items()}
        self.model.eval()
        with torch.no_grad():
            pytorch_output = self.model(**pytorch_batch)
        scores_tensor = torch.softmax(pytorch_output.logits, dim=2)
        token_scoresss = [
            [token_scores for token_scores, yn in zip(token_scoress, yns) if yn == 1]
            for token_scoress, yns in zip(scores_tensor.tolist(), batch.attention_mask)
        ]
        #
        #   postprocessing (map back to original inputs)!!
        #
        preds = []
        for j, (context_id, word_ids, token_scoress) in enumerate(
            zip(batch.context_id, batch.entity_ids, token_scoresss)
        ):
            for word_id, token_scores, is_valid_pred in zip(
                word_ids, token_scoress, self._token_pooling_strategy_mask(word_ids=word_ids)
            ):
                if word_id is None or is_valid_pred is False:
                    continue
                else:
                    label_id = np.argmax(token_scores)
                    pred = EntityClassificationPrediction(
                        context_id=context_id,
                        entity_id=word_id,
                        label=self.config.id2label[label_id],
                        score=token_scores[label_id],
                    )
                    preds.append(pred)
        return preds

    def _token_pooling_strategy_mask(
        self,
        token_ids: Optional[List[int]] = None,
        word_ids: Optional[List[Optional[int]]] = None,
        token_scores: Optional[List[Tuple[float, float]]] = None,
        strategy: str = "first",
    ) -> List[bool]:
        """
        words are split into multiple tokens, each of which has a prediction.
        there are multiple strategies to decide the model prediction at a word-level:
        1) 'first': take only the first token prediction for whole word
        2) 'max': take the highest scoring token prediction for whole word
        3) ...
        """
        if word_ids is None:
            word_ids = []

        if strategy == "first":
            mask = [True]
            prev_word_id = word_ids[0]
            for current_word_id in word_ids[1:]:
                if current_word_id == prev_word_id:
                    mask.append(False)
                else:
                    mask.append(True)
                prev_word_id = current_word_id
        else:
            raise NotImplementedError(f"mode {strategy} not implemented yet")

        # if no word ID (e.g. [cls], [sep]), always mask
        mask = [is_word if word_id is not None else False for is_word, word_id in zip(mask, word_ids)]
        return mask
