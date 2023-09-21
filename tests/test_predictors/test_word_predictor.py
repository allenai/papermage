"""
Tests for SVM Word Predictor

@kylel
"""

import json
import os
import unittest
from typing import List, Optional, Set

import numpy as np

from papermage.magelib import Document, Entity, Span
from papermage.predictors import SVMWordPredictor
from papermage.predictors.word_predictors import SVMClassifier


class TestSVMClassifier(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.fixture_path = os.path.join(os.path.dirname(__file__), "../fixtures/")
        self.classifier = SVMClassifier.from_path(
            tar_path=os.path.join(self.fixture_path, "svm_word_predictor/svm_word_predictor.tar.gz")
        )
        with open(os.path.join(self.fixture_path, "svm_word_predictor/pos_words.txt")) as f_in:
            self.pos_words = [line.strip() for line in f_in]
        with open(os.path.join(self.fixture_path, "svm_word_predictor/neg_words.txt")) as f_in:
            self.neg_words = [line.strip() for line in f_in]

    def test_batch_predict_unit(self):
        pos_words = [
            "wizard-of-oz",
            "moment-to-moment",
            "batch-to-batch",
            "Seven-day-old",
            "slow-to-fast",
            "HTLV-1-associated",
            "anti-E-selectin",
        ]
        neg_words = [
            "sig-nal-to-noise",
            "nonre-turn-to-zero",
            "comput-er-assisted",
            "concentra-tion-dependent",
            "ob-ject-oriented",
            "cog-nitive-behavioral",
            "deci-sion-makers",
        ]
        THRESHOLD = -1.5
        pos_results = self.classifier.batch_predict(words=pos_words, threshold=THRESHOLD)
        self.assertEqual(len(pos_results), len(pos_words))
        self.assertTrue(all([r.is_edit is False for r in pos_results]))
        neg_results = self.classifier.batch_predict(words=neg_words, threshold=THRESHOLD)
        self.assertEqual(len(neg_results), len(neg_words))
        self.assertTrue(all([r.is_edit is True for r in neg_results]))

    def test_batch_predict_eval(self):
        """
        As a guideline, we want Recall to be close to 1.0 because we want
        the model to favor predicting things as "negative" (i.e. not an edit).
        If the classifier predicts a "1", then essentially we don't do anything.
        Meaning in all cases where the ground truth is "1" (dont do anything),
        we want to recover all these cases nearly perfectly, and ONLY
        take action when absolutely safe.

        THRESHOLD = -1.7    --> P: 0.9621262458471761 R: 1.0
        THRESHOLD = -1.6    --> P: 0.9674346429879954 R: 1.0
        THRESHOLD = -1.5    --> P: 0.9716437941036409 R: 1.0
        THRESHOLD = -1.4    --> P: 0.9755705281460552 R: 0.9999554446622705
        THRESHOLD = -1.0    --> P: 0.9866772193641999 R: 0.9998217786490822
        THRESHOLD = -0.5    --> P: 0.9955352184633155 R: 0.9984405631794689
        THRESHOLD = 0.0     --> P: 0.9985657299090135 R: 0.9926483692746391
        THRESHOLD = 1.0     --> P: 0.9997759019944723 R: 0.8944929602566387
        """
        THRESHOLD = -1.5
        preds_pos = self.classifier.batch_predict(words=self.pos_words, threshold=THRESHOLD)
        self.assertEqual(len(preds_pos), len(self.pos_words))
        preds_pos_as_ints = [int(r.is_edit is False) for r in preds_pos]
        tp = sum(preds_pos_as_ints)
        fn = len(preds_pos_as_ints) - tp

        preds_neg = self.classifier.batch_predict(words=self.neg_words, threshold=THRESHOLD)
        self.assertEqual(len(preds_neg), len(self.neg_words))
        preds_neg_as_ints = [int(r.is_edit is True) for r in preds_neg]
        tn = sum(preds_neg_as_ints)
        fp = len(preds_neg_as_ints) - tn

        self.assertEqual(tp + fn + tn + fp, len(preds_pos) + len(preds_neg))

        p = tp / (tp + fp)
        r = tp / (tp + fn)

        # uncomment for debugging
        # print(f"P: {p} R: {r}")

        self.assertGreaterEqual(p, 0.9)
        self.assertGreaterEqual(r, 0.9)

    def test_get_features(self):
        (
            all_features,
            word_id_to_feature_ids,
        ) = self.classifier._get_features(words=self.pos_words)
        self.assertEqual(len(word_id_to_feature_ids), len(self.pos_words))
        self.assertEqual(
            all_features.shape[0],
            sum([len(feature) for feature in word_id_to_feature_ids.values()]),
        )
        (
            all_features,
            word_id_to_feature_ids,
        ) = self.classifier._get_features(words=self.neg_words)
        self.assertEqual(len(word_id_to_feature_ids), len(self.neg_words))
        self.assertEqual(
            all_features.shape[0],
            sum([len(feature) for feature in word_id_to_feature_ids.values()]),
        )

    def test_exception_with_start_or_end_hyphen(self):
        words = ["-wizard-of-", "wizard-of-"]
        for word in words:
            with self.assertRaises(ValueError):
                self.classifier.batch_predict(words=[word], threshold=0.0)
