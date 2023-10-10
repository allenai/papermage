"""
benjaminn
"""
import dataclasses
import inspect
import json
import os
from pathlib import Path
from typing import List, Literal, Optional, Union

import pytorch_lightning as pl
import seqeval.metrics
import sklearn.metrics
import springs
import torch
import transformers
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from tqdm import tqdm

from papermage.magelib import Box, Document, Entity, Span
from papermage.predictors import HFBIOTaggerPredictor


class HFBioTaggerPredictorWrapper(pl.LightningModule):
    def __init__(self, predictor: HFBIOTaggerPredictor):
        super().__init__()
        self.predictor = predictor
        self.learning_rate = None  # this will be set by the trainer
        self.warmup_steps = None  # this will be set by the trainer
        self.num_training_steps = None  # this will be set by the trainer

    def training_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pytorch_output = self.predictor.model(**batch)
        scores_tensor = torch.softmax(pytorch_output.logits, dim=2)
        self.log("train_loss", pytorch_output.loss.cpu(), on_step=True)
        return pytorch_output.loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pytorch_output = self.predictor.model(**batch)
        self.log("val_loss", pytorch_output.loss.cpu())
        return pytorch_output.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.predictor.model.parameters(), lr=self.learning_rate)
        scheduler = transformers.get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps * self.num_training_steps,
            num_training_steps=self.num_training_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def preprocess(self, doc: Document, context_name: str = "pages") -> List:
        # is this the default context we should use?
        return self.predictor.preprocess(doc, context_name)

    def on_train_start(self) -> None:
        self.predictor.device = self.device
        return super().on_train_start()

    def on_validation_start(self) -> None:
        self.predictor.device = self.device
        return super().on_validation_start()

    def on_test_start(self) -> None:
        self.predictor.device = self.device
        return super().on_test_start()

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

        trainer.logged_metrics.get("val_loss")

        # don't want to save the state twice bc it might be big. But not saving it might cause issues if we want
        # to load the model later, that's a TODO for now.
        # del checkpoint["state_dict"]


class HFBIOTaggerPredictorTrainer:
    CACHE_PATH = Path(os.environ.get("PAPERMAGE_CACHE_DIR", Path.home() / ".cache/papermage"))

    def __init__(self, predictor: HFBIOTaggerPredictor, config: DictConfig):
        if not self.CACHE_PATH.exists():
            self.CACHE_PATH.mkdir(parents=True)

        transformers.set_seed(config.seed)
        self.predictor = HFBioTaggerPredictorWrapper(predictor)
        self.predictor.learning_rate = config.learning_rate
        self.predictor.warmup_steps = config.warmup_steps

        self.config = config
        if self.config.accelerator == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = self.config.accelerator

        if config.mode == "eval":
            # just extract the original model name from the path
            # assumes that the model path starts with self.CACHE_PATH
            model_name = self.config.model_name_or_path[len(str(self.CACHE_PATH)) :].strip("/").split("_")[0]
        else:
            model_name = str(self.predictor.config.name_or_path).replace("/", "-")

        self.data_id = "_".join(
            [
                model_name,
                str(self.config.label_field),
                str(self.predictor.entity_name),
                str(self.predictor.context_name),
                str(self.predictor.batch_size),
            ]
        )

        if self.config.data_notes:
            self.data_id += f"_{self.config.data_notes}"
        (self.CACHE_PATH / self.data_id).mkdir(exist_ok=True)

        self.model_id = "_".join(
            [
                self.data_id,
                str(self.config.learning_rate),
                str(self.config.warmup_steps),
                str(self.config.seed),
                str(self.config.max_epochs),
            ]
        )

        if self.config.notes:
            self.model_id += "_" + self.config.notes

        if self.config.default_root_dir is None:
            self.config.default_root_dir = self.CACHE_PATH / self.model_id
        self.config.default_root_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [HFCheckpoint(f"{self.config.default_root_dir}/checkpoints")]

        loggers = []
        if config.wandb:
            loggers.append(
                pl.loggers.WandbLogger(
                    name=self.model_id,
                    project="papermage",
                    save_dir=self.config.default_root_dir,
                )
            )

        print("Checkpoints will be saved at:", self.config.default_root_dir)

        # Filter out the args that aren't valid for pl.Trainer using the signature of __init__ function
        # This is kind of ugly...
        pl_trainer_params = inspect.signature(pl.Trainer).parameters
        pl_trainer_config = {k: v for k, v in dict(self.config).items() if k in pl_trainer_params}

        self.trainer = pl.Trainer(callbacks=callbacks, logger=loggers, **pl_trainer_config)

    def compute_labels_from_entities(self, doc: Document, labels_fields: List[str], return_labels_by_entity=False):
        """
        Computes the gold labels given a document and which field of the doc contains the label.

        if `return_labels_by_entity` is True, the labels returned will be in the same shape as the entity list:
            ie. they will be a list of labels, with one label per entity. For example, if the entity_name is
            "token", then each element of the list would be the label associated with that token.
        if `return_labels_by_entity` is False, the labels are returned in the same shape as the batches. These
            are ready to directly be fed into the Huggingface model as the labels (eg the correct tokens will
            be masked out by setting the label to -100.
        """
        batches = self.predictor.preprocess(doc, context_name=self.config.context_name)
        label_id_batches: List[List[List[int]]] = []

        # assume the label value is a binary value. (either 1, the token is in the span specified by
        # the labels_field or 0, it is not)

        # label_entities = getattr(doc, labels_field)

        # masking stategy is "first"
        # based on here: https://huggingface.co/docs/transformers/tasks/token_classification
        labels_by_entity = []
        for batch in batches:
            label_id_batch: List[List[int]] = []
            previous_entity_idx = None
            for context_idx, entity_idxs in zip(batch.context_id, batch.entity_ids):
                label_ids: List[int] = []
                for entity_idx in entity_idxs:
                    if entity_idx is None:
                        label_ids.append(-100)
                    elif entity_idx != previous_entity_idx:
                        # if the token is within the field, label it as 1
                        context = getattr(doc, self.predictor.context_name)[context_idx]
                        entity = getattr(context, self.predictor.entity_name)[entity_idx]
                        for labels_field in labels_fields:
                            overlap = getattr(entity, labels_field)
                            if overlap:
                                # assumption: overlap has at most one element (each entity has only one label associated with it)
                                # assumption: all spans are contiguous
                                if getattr(overlap[0], self.predictor.entity_name)[0] == entity:
                                    label = self.predictor.config.label2id[f"B-{labels_field}"]
                                    # label = 1  # B tag
                                else:
                                    label = self.predictor.config.label2id[f"I-{labels_field}"]
                                    # label = 2  # I tag
                                break
                        else:  # O tag
                            label = 0
                        label_ids.append(label)
                        labels_by_entity.append(label)
                    else:
                        label_ids.append(-100)
                    previous_entity_idx = entity_idx
                label_id_batch.append(label_ids)
            label_id_batches.append(label_id_batch)

        if return_labels_by_entity:
            return batches, labels_by_entity
        else:
            return batches, label_id_batches

    def preprocess(
        self, docs_path: Path, labels_fields: Optional[List[str]], annotations_path: Optional[Path] = None
    ):
        with open(docs_path, "r") as docs_file:
            all_pytorch_batches = []

            docs = [Document.from_json(json.loads(line)) for line in docs_file]
            # breakpoint()
            for doc in tqdm(docs, desc="preprocessing"):
                if labels_fields is not None:
                    batches, label_id_batches = self.compute_labels_from_entities(doc, labels_fields)

                    pytorch_batches = [
                        self.predictor.python_to_torch_mapper.transform(
                            data=self.predictor.list_collator_mapper.transform(
                                data={
                                    self.predictor._HF_RESERVED_INPUT_IDS: batch.input_ids,
                                    self.predictor._HF_RESERVED_ATTN_MASK: batch.attention_mask,
                                    "labels": label_ids,
                                }
                            )
                        )
                        for batch, label_ids in zip(batches, label_id_batches)
                    ]

                    all_pytorch_batches.extend(pytorch_batches)

            return all_pytorch_batches

    def load_docs_from_path(
        self, docs_path: Path, annotations_entity_names: Optional[List[str]] = None, prefix: str = ""
    ) -> DataLoader:
        # If pytorch tensors haven't been created and cached yet
        # preprocess the document to convert it to tensors and cache them
        preprocessed_batches = []
        cache_file = self.CACHE_PATH / self.data_id / f"{prefix}inputs.pt"
        if not cache_file.exists():
            preprocessed_batches = self.preprocess(
                docs_path=docs_path, labels_fields=annotations_entity_names, annotations_path=None
            )

            print(f"Caching preprocessed batches in: {cache_file}")
            torch.save(preprocessed_batches, cache_file)

        else:
            # load the tensors from the cache
            print(f"Loading precomputed input batches from: {cache_file}")
            preprocessed_batches = torch.load(cache_file)

        # if not prefix:
        #     preprocessed_batches = preprocessed_batches[:10]  # just for now, do a small amount of training.
        # create the dataloader
        preprocessed_batches = [{k: v.to(self.device) for k, v in batch.items()} for batch in preprocessed_batches]
        docs_dataloader = DataLoader(preprocessed_batches, batch_size=None)  # disable automatic batching
        return docs_dataloader

    def train(
        self,
        docs_path: Path,
        val_docs_path: Optional[Path],
        annotations_entity_names: Optional[List[str]] = None,
        annotations_path: Optional[Path] = None,
    ):
        train_dataloader = self.load_docs_from_path(
            docs_path=docs_path,
            annotations_entity_names=annotations_entity_names,
        )

        if val_docs_path is not None:
            val_dataloader = self.load_docs_from_path(
                docs_path=val_docs_path, annotations_entity_names=annotations_entity_names, prefix="val_"
            )
        else:
            val_dataloader = None

        # Run the training loop
        self._train(train_dataloader, val_dataloader)

    def _train(self, train_docs: DataLoader, val_docs: Optional[DataLoader] = None):
        # The module should automatically be put on the correct device, so I'm not sure why I need to do this.
        # set the number of training steps for the optimizer.
        self.predictor.num_training_steps = len(train_docs) * self.config.max_epochs
        self.predictor.model.to(self.device)
        if val_docs is not None:
            self.trainer.fit(model=self.predictor, train_dataloaders=train_docs, val_dataloaders=val_docs)
        else:
            self.trainer.fit(model=self.predictor, train_dataloaders=train_docs)

    def eval(self, docs_path: Path, annotations_entity_names: List[str]):
        """This is going to be a bit different from just calling `predict` on the predictor.

        This is because we might want to cache the inputs to pytorch and because we need
        to compute the labels.
        """
        self.predictor.model.to(self.device)
        docs = []
        with open(docs_path) as f:
            for line in f:
                docs.append(Document.from_json(json.loads(line)))

        all_batches = []
        all_gold_labels = []
        all_pred_labels = []
        # cache_file = self.config.default_root_dir / "test_inputs.pt"
        cache_file = self.CACHE_PATH / self.data_id / "test_inputs.pt"
        if not cache_file.exists():
            for document in tqdm(docs, desc="preprocessing", total=len(docs)):
                # Wraps the preprocess step - this should be cached
                batches, gold_labels = self.compute_labels_from_entities(
                    doc=document,
                    labels_fields=annotations_entity_names,
                    return_labels_by_entity=True,
                )
                gold_labels = [self.predictor.config.id2label[lab] for lab in gold_labels]
                all_gold_labels.append(gold_labels)
                all_batches.append(batches)
            torch.save({"batches": all_batches, "labels": all_gold_labels}, cache_file)
        else:
            cached_data = torch.load(cache_file)
            all_batches = cached_data["batches"]
            all_gold_labels = cached_data["labels"]

        all_entities = []
        all_extracted_text = []
        for document, batches in tqdm(zip(docs, all_batches), desc="evaluating", total=len(all_batches)):
            # What I want: two lists. one of predicted labels and one of gold labels
            # each item of the list is associated with a entity (eg token)
            # basically, this amounts to removing the masked tokens from `label_ids`
            # and combining the ones with the same entity id
            preds = []
            for batch in batches:
                for pred in self.predictor.predictor._predict_batch(batch=batch, device=self.device):
                    preds.append(pred)
            all_pred_labels.append([pred.label for pred in preds])

            # (3) Postprocess into proper Entities
            entities = self.predictor.predictor.postprocess(
                doc=document, context_name=self.predictor.context_name, preds=preds, merge_tokens=False
            )

            all_entities.append(entities)

            extracted_text = []
            for entity in entities:
                if entity.metadata["label"] != "O":
                    extracted_text.append(document.symbols[entity.spans[0].start : entity.spans[0].end])
            all_extracted_text.append(extracted_text)

        # Calculate precision, recall, f1, accuracy at the span level and token level
        try:
            clf_report_span = seqeval.metrics.classification_report(all_gold_labels, all_pred_labels)
        except ValueError:
            clf_report_span = "No predicted fields."

        # Calculate the p, r, f1, at the token level (like in the VILA paper).
        # To do this, remove the B-/I- tag from the label and just use the field name
        # (E.g. "B-Title"[:2] => "Title")
        clf_report_token = sklearn.metrics.classification_report(
            [gold_lab[2:] for gold_labels in all_gold_labels for gold_lab in gold_labels],
            [pred_lab[2:] for pred_labels in all_pred_labels for pred_lab in pred_labels],
        )

        (self.config.default_root_dir / "results").mkdir(exist_ok=True)

        with open(self.config.default_root_dir / "results" / "clf_report_token.txt", "w") as f:
            f.write(clf_report_token)

        with open(self.config.default_root_dir / "results" / "clf_report_span.txt", "w") as f:
            f.write(clf_report_span)

        # breakpoint()
        with open(self.config.default_root_dir / "results" / "results.json", "w") as f:
            json.dump(
                {
                    "y_gold": all_gold_labels,
                    "y_hat": all_pred_labels,
                    "annotations": [
                        [annotation.to_json() for annotation in annotations] for annotations in all_entities
                    ],
                },
                f,
            )

        print("Token level prediction")
        print(clf_report_token)
        print("Results saved to:", (self.config.default_root_dir / "results"))

        # return all_gold_labels, all_pred_labels, all_extracted_text


@springs.dataclass
class HFBIOTaggerPredictorTrainConfig:
    """Stores the default training args"""

    data_path: Path  # Path to the data to train on. Should be a jsonl file where each row is a doc.
    label_field: str  # The field in the document to use as the labels for the tokens.
    val_data_path: Optional[Path] = None
    entity_name: str = "tokens"  # The field in the document to use as the unit of prediction.
    context_name: str = "pages"  # The field in the document to use as the input example to the model.
    model_name_or_path: str = "allenai/scibert_scivocab_uncased"
    learning_rate: float = 5e-4
    warmup_steps: float = 0.0  # proportion of training steps allocated to warmup
    mode: str = "train"  # One of "train", "eval"
    notes: str = ""
    data_notes: str = ""
    seed: int = 470

    wandb: bool = False
    # pytorch-lightning trainer args (these are the defaults). Some of the types are wrong because OmeagConf can't
    # load in arbitrary types.
    accelerator: str = "auto"
    accumulate_grad_batches: int = 1
    precision: Union[int, str] = 32
    devices: Optional[Union[str, int]] = torch.cuda.device_count() if torch.cuda.is_available() else 1
    max_epochs: int = 5
    max_steps: int = -1
    check_val_every_n_epoch: int = 1
    default_root_dir: Optional[Union[str, Path]] = None
    log_every_n_steps: int = 1
    val_check_interval: Union[float, int] = 1.0
    # Args for development/testing
    fast_dev_run: bool = False
    overfit_batches: bool = False


@springs.cli(HFBIOTaggerPredictorTrainConfig)
def main(config: HFBIOTaggerPredictorTrainConfig):
    """Launch a training run.

    For now, the simplest way to do this is to just pass the data. We can abstract this later and add some way to
    configure the training run.
    """

    # parse out the label fields:
    label_fields = config.label_field.split(",")
    labels = ["O"] + [f"{initial}-{label}" for initial in "BI" for label in label_fields]
    id2label = dict(zip(range(len(labels)), labels))
    label2id = {v: k for k, v in id2label.items()}
    print(id2label)
    print(label2id)
    kwargs = {"num_labels": len(id2label), "id2label": id2label, "label2id": label2id}

    if "roberta" in config.model_name_or_path:
        kwargs["add_prefix_space"] = True
        # kwargs["max_position_embeddings"] = 512  # This is set to 514 for some reason...

    # Initialize the trainer
    trainer = HFBIOTaggerPredictorTrainer(
        predictor=HFBIOTaggerPredictor.from_pretrained(
            model_name_or_path=config.model_name_or_path,
            entity_name=config.entity_name,
            context_name=config.context_name,
            **kwargs,
        ),
        config=config,
    )

    if config.mode == "train":
        trainer.train(
            docs_path=config.data_path, val_docs_path=config.val_data_path, annotations_entity_names=label_fields
        )
    elif config.mode == "eval":
        trainer.eval(docs_path=config.data_path, annotations_entity_names=label_fields)
    else:
        raise ValueError(f"Invalid mode: {config.mode}")


if __name__ == "__main__":
    main()
