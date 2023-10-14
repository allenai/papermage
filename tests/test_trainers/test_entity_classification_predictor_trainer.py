"""

@benjaminn

"""

import json
import unittest
from pathlib import Path

import springs
import torch
import transformers

from papermage.magelib import Document, Entity, Span
from papermage.predictors import HFBIOTaggerPredictor
from papermage.trainers.bio_tagger_predictor_trainer import (
    HFBIOTaggerPredictorTrainConfig,
    HFBIOTaggerPredictorTrainer,
)

TEST_SCIBERT_WEIGHTS = "allenai/scibert_scivocab_uncased"


class TestEntityClassificationPredictorTrainer(unittest.TestCase):
    def setUp(self):
        transformers.set_seed(407)
        self.fixture_path = Path("tests/fixtures")
        with open(self.fixture_path / "entity_classification_predictor_test_doc_papermage.json", "r") as f:
            test_doc_json = json.load(f)
        self.doc = Document.from_json(doc_json=test_doc_json)
        ent1 = Entity(spans=[Span(start=86, end=456)])
        ent2 = Entity(spans=[Span(start=457, end=641)])
        self.doc.annotate_layer(name="bibs", entities=[ent1, ent2])

        self.predictor = HFBIOTaggerPredictor.from_pretrained(
            model_name_or_path=TEST_SCIBERT_WEIGHTS,
            entity_name="tokens",
            context_name="pages",
            num_labels=3,
            id2label={0: "O", 1: "B_Title", 2: "I_Title"},
            label2id={"O": 0, "B_Title": 1, "I_Title": 2},
        )

        config = springs.from_dataclass(HFBIOTaggerPredictorTrainConfig)
        config = springs.merge(
            config,
            springs.from_dict(
                {
                    "data_path": self.fixture_path / "predictor_training_docs_tiny.jsonl",
                    "label_field": "words_starting_with_td",
                    "max_steps": 1,
                    "seed": 407,
                }
            ),
        )
        self.trainer = HFBIOTaggerPredictorTrainer(
            predictor=self.predictor,
            config=config,
        )

    def test_train(self) -> None:
        if (self.trainer.CACHE_PATH / self.trainer.data_id / "inputs.pt").exists():
            (self.trainer.CACHE_PATH / self.trainer.data_id / "inputs.pt").unlink()

        # self.trainer.train(docs_path="tests/fixtures/predictor_training_docs.jsonl", annotations_entity_name="bibs")
        id2label = {0: "O", 1: "B-words_starting_with_td", 2: "I-words_starting_with_td"}
        label2id = {"O": 0, "B-words_starting_with_td": 1, "I-words_starting_with_td": 2}

        self.trainer.predictor.predictor.config.id2label = id2label
        self.trainer.predictor.predictor.config.label2id = label2id
        self.trainer.train(
            docs_path=self.fixture_path / "predictor_training_docs_tiny.jsonl",
            val_docs_path=None,
            annotations_entity_names=["words_starting_with_td"],
        )

        # check that the cache file exists
        assert (self.trainer.CACHE_PATH / self.trainer.data_id / "inputs.pt").exists()

        # check that we can load in the trained model and run it on the test doc (`self.doc`)
        new_predictor = HFBIOTaggerPredictor.from_pretrained(
            model_name_or_path=self.trainer.config.default_root_dir / "checkpoints",
            entity_name=self.predictor.entity_name,
            context_name=self.predictor.context_name,
            id2label=id2label,
            label2id=label2id,
        )

        token_tags = new_predictor.predict(doc=self.doc)
        assert len(token_tags) == 20

    def test_preprocess(self):
        self.trainer.predictor.predictor.config.id2label = {
            0: "O",
            1: "B-words_starting_with_td",
            2: "I-words_starting_with_td",
        }
        self.trainer.predictor.predictor.config.label2id = {
            "O": 0,
            "B-words_starting_with_td": 1,
            "I-words_starting_with_td": 2,
        }
        preprocessed_batches = self.trainer.preprocess(
            docs_path=self.fixture_path / "predictor_training_docs_tiny.jsonl",
            labels_fields=["words_starting_with_td"],
        )

        gold_preprocessed_batches = torch.load(self.fixture_path / "preprocessed_training_docs_tiny.pt")
        for gold_batch, test_batch in zip(gold_preprocessed_batches, preprocessed_batches):
            for key in set(gold_batch.keys()) | set(test_batch.keys()):
                assert torch.allclose(gold_batch[key], test_batch[key])

        # test multi-word entities
        self.trainer.predictor.predictor.config.id2label = {
            0: "O",
            1: "B-multi_word_entity",
            2: "I-multi_word_entity",
        }
        self.trainer.predictor.predictor.config.label2id = {
            "O": 0,
            "B-multi_word_entity": 1,
            "I-multi_word_entity": 2,
        }
        preprocessed_batches_mwe = self.trainer.preprocess(
            docs_path=self.fixture_path / "predictor_training_docs_tiny.jsonl",
            labels_fields=["multi_word_entity"],
        )
        gold_preprocessed_batches_mwe = torch.load(self.fixture_path / "preprocessed_training_docs_tiny_mwe.pt")
        for gold_batch, test_batch in zip(gold_preprocessed_batches_mwe, preprocessed_batches_mwe):
            for key in set(gold_batch.keys()) | set(test_batch.keys()):
                assert torch.allclose(gold_batch[key], test_batch[key])

        # test multiple fields
        id2label = {
            0: "O",
            1: "B-words_starting_with_td",
            2: "I-words_starting_with_td",
            3: "B-multi_word_entity",
            4: "I-multi_word_entity",
        }
        label2id = {lab: id for id, lab in id2label.items()}
        self.trainer.predictor.predictor.config.id2label = id2label
        self.trainer.predictor.predictor.config.label2id = label2id
        preprocessed_batches_multifields = self.trainer.preprocess(
            docs_path=self.fixture_path / "predictor_training_docs_tiny.jsonl",
            labels_fields=["words_starting_with_td", "multi_word_entity"],
        )
        gold_preprocessed_batches_multifields = torch.load(
            self.fixture_path / "preprocessed_training_docs_tiny_multifields.pt"
        )
        for gold_batch, test_batch in zip(gold_preprocessed_batches_multifields, preprocessed_batches_multifields):
            for key in set(gold_batch.keys()) | set(gold_batch.keys()):
                assert torch.allclose(gold_batch[key], test_batch[key])

    def test_eval(self):
        if (self.trainer.CACHE_PATH / self.trainer.data_id / "test_inputs.pt").exists():
            (self.trainer.CACHE_PATH / self.trainer.data_id / "test_inputs.pt").unlink()
        transformers.set_seed(407)
        self.trainer.predictor.predictor.config.id2label = {
            0: "O",
            1: "B-multi_word_entity",
            2: "I-multi_word_entity",
        }
        self.trainer.predictor.predictor.config.label2id = {
            "O": 0,
            "B-multi_word_entity": 1,
            "I-multi_word_entity": 2,
        }
        self.trainer.eval(
            docs_path=self.fixture_path / "predictor_training_docs_tiny.jsonl",
            annotations_entity_names=["multi_word_entity"],
        )

        # Check that inputs are cached
        assert (self.trainer.CACHE_PATH / self.trainer.data_id / "test_inputs.pt").exists()

        # Check that the outputs are correct
        with open(self.trainer.config.default_root_dir / "results" / "results.json") as f:
            pred_results = json.load(f)

        with open(self.fixture_path / "eval_results.json") as f:
            gold_results = json.load(f)

        assert gold_results["y_gold"] == pred_results["y_gold"]
        assert gold_results["y_hat"] == pred_results["y_hat"]

    def save_and_load_checkpoint(self):
        pass
