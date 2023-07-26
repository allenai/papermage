"""

@benjaminn

"""

import unittest

import json
from pathlib import Path

import springs
import torch

from papermage.types import Document, Entity, Span
from papermage.parsers import PDFPlumberParser
from papermage.predictors.hf_predictors.entity_classification_predictor import (
    EntityClassificationPredictor
)
from papermage.trainers.entity_classification_predictor_trainer import (
    EntityClassificationPredictorTrainer,
    EntityClassificationTrainConfig,
)

TEST_SCIBERT_WEIGHTS = 'allenai/scibert_scivocab_uncased'


class TestEntityClassificationPredictorTrainer(unittest.TestCase):
    def setUp(self):
        self.fixture_path = Path("tests/fixtures")
        with open(self.fixture_path / "entity_classification_predictor_test_doc_papermage.json", "r") as f:
            test_doc_json = json.load(f)
        self.doc = Document.from_json(doc_json=test_doc_json)
        ent1 = Entity(spans=[Span(start=86, end=456)])
        ent2 = Entity(spans=[Span(start=457, end=641)])
        self.doc.annotate_entity(field_name="bibs", entities=[ent1, ent2])
        ent1.id = 0
        ent2.id = 1


        self.predictor = EntityClassificationPredictor.from_pretrained(
            model_name_or_path=TEST_SCIBERT_WEIGHTS,
            entity_name='tokens',
            context_name='pages'
        )

        config = springs.from_dataclass(EntityClassificationTrainConfig)
        config = springs.merge(config, springs.from_dict({
            'data_path': self.fixture_path / "predictor_training_docs_tiny.jsonl",
            'label_field': "words_starting_with_td",
            'max_steps': 1,
        }))
        self.trainer = EntityClassificationPredictorTrainer(
            predictor=self.predictor,
            config=config,
        )

    def test_train(self):
        if (self.trainer.CACHE_PATH / "default.pt").exists():
            (self.trainer.CACHE_PATH / "default.pt").unlink()

        # self.trainer.train(docs_path="tests/fixtures/predictor_training_docs.jsonl", annotations_entity_name="bibs")
        self.trainer.train(docs_path=self.fixture_path / "predictor_training_docs_tiny.jsonl", annotations_entity_name="words_starting_with_td")

        # check that the cache file exists
        assert (self.trainer.CACHE_PATH / "default.pt").exists()

        # check that we can load in the trained model and run it on the test doc (`self.doc`)
        new_predictor = EntityClassificationPredictor.from_pretrained(
            model_name_or_path=self.trainer.CACHE_PATH / self.trainer.model_id / "checkpoints",
            entity_name=self.predictor.entity_name,
            context_name=self.predictor.context_name,
        )

        token_tags = new_predictor.predict(document=self.doc)
        assert len(token_tags) == len([token for page in self.doc.pages for token in page.tokens])


    def test_preprocess(self):
        preprocessed_batches = self.trainer.preprocess(
            docs_path=self.fixture_path / "predictor_training_docs_tiny.jsonl",
            labels_field="words_starting_with_td",
        )

        gold_preprocessed_batches = torch.load(self.fixture_path / "preprocessed_training_docs_tiny.pt")
        for gold_batch, test_batch in zip(gold_preprocessed_batches, preprocessed_batches):
            for key in set(gold_batch.keys()) | set(test_batch.keys()):
                assert torch.allclose(gold_batch[key], test_batch[key])


    def save_and_load_checkpoint(self):
        pass