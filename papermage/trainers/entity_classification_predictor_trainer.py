"""
benjaminn
"""
import dataclasses
import inspect
import json
import os
from pathlib import Path
from typing import List, Optional, Union, Literal

import pytorch_lightning as pl
import springs
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader

from papermage.types import Document, Entity, Span, Box
from papermage.predictors.hf_predictors.entity_classification_predictor import EntityClassificationPredictor

class EntityClassificationPredictorWrapper(pl.LightningModule):
    def __init__(self, predictor: EntityClassificationPredictor):
        super().__init__()
        self.predictor = predictor

    def training_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pytorch_output = self.predictor.model(**batch)
        scores_tensor = torch.softmax(pytorch_output.logits, dim=2)
        return pytorch_output.loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.predictor.model.parameters(), lr=self.predictor.learning_rate)

    def preprocess(self, doc: Document) -> List:
        # is this the default context we should use?
        return self.predictor.preprocess(doc, "pages")

    def __getattr__(self, name):
        """Allow access to the predictor's attributes if the wrapper doesn't have them."""
        if name not in self.__dict__:
            return getattr(self.predictor, name)


class HFCheckpoint(pl.Callback):
    def __init__(self, save_dir: str) -> None:
        """Initialize the HFCheckpoint Callback for saving huggingface checkpoints

        Args:
            save_dir (str): the directory to save the huggingface checkpoints.
        """

        super().__init__()
        self.save_dir = save_dir

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # save the model in hf format for easy loading
        pl_module.predictor.model.save_pretrained(self.save_dir)
        pl_module.predictor.tokenizer.save_pretrained(self.save_dir)

        # don't want to save the state twice bc it might be big. But not saving it might cause issues if we want
        # to load the model later, that's a TODO for now.
        # del checkpoint["state_dict"]


class EntityClassificationPredictorTrainer:

    CACHE_PATH = Path(os.environ.get("PAPERMAGE_CACHE_DIR", 
        Path.home() / ".cache/papermage"))

    def __init__(self, predictor: EntityClassificationPredictor, config: DictConfig):
        if not self.CACHE_PATH.exists():
            self.CACHE_PATH.mkdir(parents=True)

        self.predictor = EntityClassificationPredictorWrapper(predictor)
        self.config = config

        self.model_id = "_".join([
            str(self.predictor.config.name_or_path).replace("/", "-"),
            str(self.config.label_field),
            str(self.predictor.entity_name),
            str(self.predictor.context_name),
            str(self.predictor.batch_size),
            str(self.config.learning_rate),
        ])

        if self.config.default_root_dir is None:
            self.config.default_root_dir = self.CACHE_PATH / self.model_id

        callbacks = [
            HFCheckpoint(f"{self.config.default_root_dir}/checkpoints")
        ]

        print("Checkpoints will be saved at:", self.config.default_root_dir)

        # Filter out the args that aren't valid for pl.Trainer using the signature of __init__ function
        # This is kind of ugly...
        pl_trainer_params = inspect.signature(pl.Trainer).parameters
        pl_trainer_config = {k: v for k, v in dict(self.config).items() if k in pl_trainer_params}

        self.trainer = pl.Trainer(
            callbacks=callbacks,
            **pl_trainer_config
        )

    def preprocess(self, docs_path: Path, labels_field: Optional[str], annotations_path: Optional[Path] = None):

        with open(docs_path, "r") as docs_file:
            all_pytorch_batches = []

            for line in docs_file:
                doc = Document.from_json(json.loads(line))
                batches = self.predictor.preprocess(doc)
                label_id_batches: List[List[List[int]]] = []

                if labels_field is not None:
                    # assume the label value is a binary value. (either 1, the token is in the span specified by
                    # the labels_field or 0, it is not)

                    label_entities = getattr(doc, labels_field)

                    # masking stategy is "first"
                    # based on here: https://huggingface.co/docs/transformers/tasks/token_classification
                    for batch in batches:
                        label_ids: List[List[int]] = []
                        previous_entity_idx = None        
                        for context_idx, entity_idxs in zip(batch.context_id, batch.entity_ids):
                            labels: List[int] = []
                            for entity_idx in entity_idxs:
                                if entity_idx is None:
                                    labels.append(-100)
                                elif entity_idx != previous_entity_idx:
                                    # if the token is within the field, label it as 1
                                    context = getattr(doc, self.predictor.context_name)[context_idx]
                                    entity = getattr(context, self.predictor.entity_name)[entity_idx]
                                    overlap = getattr(entity, labels_field)
                                    if overlap:
                                        # assumption: overlap has at most one element (each entity has only one label associated with it)
                                        # assumption: all spans are contiguous
                                        if getattr(overlap[0], self.predictor.entity_name)[0] == entity:
                                            label = 1  # B tag
                                        else:
                                            label = 2  # I tag
                                    else:  # O tag
                                        label = 0
                                    labels.append(label)
                                else:
                                    labels.append(-100)
                                previous_entity_idx = entity_idx
                            label_ids.append(labels)
                        label_id_batches.append(label_ids)

                pytorch_batches = [
                    self.predictor.python_to_torch_mapper.transform(
                    data=self.predictor.list_collator_mapper.transform(
                        data={
                            self.predictor._HF_RESERVED_INPUT_IDS: batch.input_ids,
                            self.predictor._HF_RESERVED_ATTN_MASK: batch.attention_mask,
                            "labels": label_ids,
                        }
                    ))
                    for batch, label_ids in zip(batches, label_id_batches)
                ]

                all_pytorch_batches.extend(pytorch_batches)

            return all_pytorch_batches

    def train(self, docs_path: Path, annotations_entity_name: Optional[str] = None, annotations_path: Optional[Path] = None, model_id: Optional[str] = None):
        # If pytorch tensors haven't been created and cached yet
        # preprocess the document to convert it to tensors and cache them
        preprocessed_batches = []
        if model_id is None or not (self.CACHE_PATH / f"{model_id}.pt").exists():
            if model_id is None:
                model_id = "default"  # TODO make this some function of the data and the predictor

            preprocessed_batches = self.preprocess(
                docs_path=docs_path, labels_field=annotations_entity_name, annotations_path=annotations_path
            )
            
            cache_file = self.CACHE_PATH / f"{model_id}.pt"
            
            torch.save(preprocessed_batches, cache_file)
            
        else:    
            # load the tensors from the cache
            cache_file = self.CACHE_PATH / f"{model_id}.pt"
            print(f"Loading cached preprocessed batches from {cache_file}")
            preprocessed_batches = torch.load(cache_file)

        # create the dataloader
        docs_dataloader = DataLoader(preprocessed_batches, batch_size=None)  # disable automatic batching

        # Run the training loop
        self._train(docs_dataloader)
    
    def _train(self, docs: DataLoader):
        self.trainer.fit(self.predictor, docs)


@springs.dataclass
class EntityClassificationTrainConfig:
    """Stores the default training args"""
    data_path: Path  # Path to the data to train on. Should be a jsonl file where each row is a doc.
    label_field: str  # The field in the document to use as the labels for the tokens.
    entity_name: str = "tokens"  # The field in the document to use as the unit of prediction.
    context_name: str = "pages"  # The field in the document to use as the input example to the model.
    model_name_or_path: str = "allenai/scibert_scivocab_uncased"
    learning_rate: float = 5e-4

    # pytorch-lightning trainer args (these are the defaults). Some of the types are wrong because OmeagConf can't
    # load in arbitrary types.
    accelerator: str = "auto"
    accumulate_grad_batches: int = 1
    precision: Union[int, str] = 32
    devices: Optional[Union[str, int]] = (
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    )
    max_epochs: int = 5
    max_steps: Optional[int] = None
    check_val_every_n_epoch: int = 1
    default_root_dir: Optional[Union[str, Path]] = None
    log_every_n_steps: int = 1
    val_check_interval: Union[float, int] = 1.0
    # Args for development/testing
    fast_dev_run: bool = False
    overfit_batches: bool = False


@springs.cli(EntityClassificationTrainConfig)
def main(config: EntityClassificationTrainConfig):
    """Launch a training run.
    
    For now, the simplest way to do this is to just pass the data. We can abstract this later and add some way to
    configure the training run.
    """

    # Initialize the trainer
    trainer = EntityClassificationPredictorTrainer(
        predictor=EntityClassificationPredictor.from_pretrained(
            model_name_or_path=config.model_name_or_path,
            entity_name=config.entity_name,
            context_name=config.context_name,
        ),
        config=config
    )

    trainer.train(docs_path=config.data_path, annotations_entity_name=config.label_field)


if __name__ == "__main__":
     main()