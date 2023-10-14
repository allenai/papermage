"""

SVM Word Predictor

Given a list of tokens, predict which tokens were originally part of the same word.
This does this in two phases: First, it uses a whitespace tokenizer to inform
whether tokens were originally part of the same word. Second, it uses a SVM
classifier to predict whether hyphenated segments should be considered a single word.

@kylel, @amanpreets

"""

import logging
import os
import re
import tarfile
import tempfile
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union
from urllib.parse import urlparse

import requests
from joblib import load
from scipy.sparse import hstack

from papermage.magelib import (
    Document,
    Entity,
    Metadata,
    Span,
    TokensFieldName,
    WordsFieldName,
)
from papermage.parsers.pdfplumber_parser import PDFPlumberParser
from papermage.predictors import BasePredictor
from papermage.predictors.token_predictors import HFWhitspaceTokenPredictor

logger = logging.getLogger(__name__)


def make_text(entity: Entity, document: Document, field: str = WordsFieldName) -> str:
    candidate_words = document.intersect_by_span(entity, field)
    candidate_text: List[str] = []

    for i in range(len(candidate_words)):
        candidate_text.append(str(candidate_words[i].text))
        if i < len(candidate_words) - 1:
            next_word_start = candidate_words[i + 1].start
            curr_word_end = candidate_words[i].end
            assert isinstance(next_word_start, int), f"{candidate_words[i + 1]} has no span (non-int start)"
            assert isinstance(curr_word_end, int), f"{candidate_words[i]} has no span (non-int end)"
            if curr_word_end != next_word_start:
                candidate_text.append(document.symbols[curr_word_end:next_word_start])

    return "".join(candidate_text)


class IsWordResult:
    def __init__(self, original: str, new: str, is_edit: bool) -> None:
        self.original = original
        self.new = new
        self.is_edit = is_edit


class SVMClassifier:
    def __init__(self, ohe_encoder, scaler, estimator, unigram_probs):
        self.ohe_encoder = ohe_encoder
        self.scaler = scaler
        self.estimator = estimator
        self.unigram_probs = unigram_probs
        self.default_prob = unigram_probs["<unk>"]
        self.sparse_columns = [
            "shape",
            "s_bg1",
            "s_bg2",
            "s_bg3",
            "p_bg1",
            "p_bg2",
            "p_bg3",
            "p_lower",
            "s_lower",
        ]
        self.dense_columns = [
            "p_upper",
            "s_upper",
            "p_number",
            "s_number",
            "p_isalpha",
            "s_isalpha",
            "p_len",
            "s_len",
            "multi_hyphen",
            "uni_prob",
        ]

    @classmethod
    def from_path(cls, tar_path: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            if urlparse(url=tar_path).scheme != "":
                r = requests.get(tar_path)
                with open(os.path.join(tmp_dir, "svm_word_predictor.tar.gz"), "wb") as f:
                    f.write(r.content)
                tar_path = os.path.join(tmp_dir, "svm_word_predictor.tar.gz")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=tmp_dir)
                return cls.from_directory(tmp_dir)

    @classmethod
    def from_directory(cls, dir: str):
        classifier = SVMClassifier.from_paths(
            ohe_encoder_path=os.path.join(dir, "svm_word_predictor/ohencoder.joblib"),
            scaler_path=os.path.join(dir, "svm_word_predictor/scaler.joblib"),
            estimator_path=os.path.join(dir, "svm_word_predictor/hyphen_clf.joblib"),
            unigram_probs_path=os.path.join(dir, "svm_word_predictor/unigram_probs.pkl"),
        )
        return classifier

    @classmethod
    def from_paths(
        cls,
        ohe_encoder_path: str,
        scaler_path: str,
        estimator_path: str,
        unigram_probs_path: str,
    ):
        ohe_encoder = load(ohe_encoder_path)
        scaler = load(scaler_path)
        estimator = load(estimator_path)
        unigram_probs = load(unigram_probs_path)
        classifier = SVMClassifier(
            ohe_encoder=ohe_encoder,
            scaler=scaler,
            estimator=estimator,
            unigram_probs=unigram_probs,
        )
        return classifier

    def batch_predict(self, words: List[str], threshold: float) -> List[IsWordResult]:
        if any([word.startswith("-") or word.endswith("-") for word in words]):
            raise ValueError("Words should not start or end with hyphens.")

        all_features, word_id_to_feature_ids = self._get_features(words)
        all_scores = self.estimator.decision_function(all_features)
        results = []
        for word_id, feature_ids in word_id_to_feature_ids.items():
            word = words[word_id]
            word_segments = word.split("-")
            score_per_hyphen_in_word = all_scores[feature_ids]
            new_word = word_segments[0]
            for word_segment, hyphen_score in zip(word_segments[1:], score_per_hyphen_in_word):
                if hyphen_score > threshold:
                    new_word += "-"
                else:
                    new_word += ""
                new_word += word_segment
            results.append(IsWordResult(original=word, new=new_word, is_edit=word != new_word))
        return results

    def _get_dense_features(self, part: str, name_prefix: str):
        upper = int(part[0].isupper())
        number = int(part.isnumeric())
        alpha = int(part.isalpha())
        lower = part.lower()
        plen = len(part)
        return {
            f"{name_prefix}_upper": upper,
            f"{name_prefix}_number": number,
            f"{name_prefix}_isalpha": alpha,
            f"{name_prefix}_len": plen,
        }

    def _get_features(self, words: List[str]):
        sparse_all, dense_all = [], []
        idx, word_id_to_feature_ids = 0, dict()
        for widx, word in enumerate(words):
            split = word.split("-")
            for i, s in enumerate(split[:-1]):
                sparse_features, dense_features = dict(), dict()
                prefix = "-".join(split[: i + 1])
                suffix = "-".join(split[i + 1 :])
                if widx not in word_id_to_feature_ids:
                    word_id_to_feature_ids[widx] = []
                word_id_to_feature_ids[widx].append(idx)
                idx += 1
                dense_features.update(self._get_dense_features(prefix, "p"))
                dense_features.update(self._get_dense_features(suffix, "s"))
                orig_uni_prob = self.unigram_probs.get(word, self.default_prob)
                presuf_uni_prob = self.unigram_probs.get(f"{prefix}{suffix}", self.default_prob)
                dense_features["uni_prob"] = orig_uni_prob - presuf_uni_prob
                dense_features["multi_hyphen"] = int(word.count("-") > 1)
                sparse_features["shape"] = re.sub("\w", "x", word)
                sparse_features["s_lower"] = suffix.lower()
                sparse_features["s_bg1"] = suffix[:2]
                sparse_features["s_bg2"] = suffix[1:3] if len(suffix) > 2 else ""
                sparse_features["s_bg3"] = suffix[2:4] if len(suffix) > 3 else ""
                sparse_features["p_lower"] = prefix.lower()
                sparse_features["p_bg1"] = prefix[-2:][::-1] if len(prefix) > 1 else ""
                sparse_features["p_bg2"] = prefix[-3:-1][::-1] if len(prefix) > 2 else ""
                sparse_features["p_bg3"] = prefix[-4:-2][::-1] if len(prefix) > 3 else ""
                sparse_all.append([sparse_features[k] for k in self.sparse_columns])
                dense_all.append([dense_features[k] for k in self.dense_columns])
        dense_transformed = self.scaler.transform(dense_all)
        sparse_transformed = self.ohe_encoder.transform(sparse_all)

        return hstack([sparse_transformed, dense_transformed]), word_id_to_feature_ids


class SVMWordPredictor(BasePredictor):
    def __init__(
        self,
        classifier: SVMClassifier,
        threshold: float = -1.5,
        punct_as_words: str = PDFPlumberParser.DEFAULT_PUNCTUATION_CHARS.replace("-", ""),
    ):
        self.classifier = classifier
        self.whitespace_predictor = HFWhitspaceTokenPredictor()
        self.threshold = threshold
        self.punct_as_words = punct_as_words

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [TokensFieldName]

    @classmethod
    def from_path(cls, tar_path: str):
        classifier = SVMClassifier.from_path(tar_path=tar_path)
        predictor = SVMWordPredictor(classifier=classifier)
        return predictor

    @classmethod
    def from_directory(cls, dir: str, threshold: float = -1.5):
        classifier = SVMClassifier.from_directory(dir=dir)
        predictor = SVMWordPredictor(classifier=classifier, threshold=threshold)
        return predictor

    def _predict(self, doc: Document) -> List[Entity]:
        # clean input
        doc = self._make_clean_document(doc=doc)

        # validate input
        self._validate_tokenization(doc=doc)

        # initialize output data using whitespace tokenization
        # also avoid grouping specific punctuation. each instance should be their own word.
        (
            token_id_to_word_id,
            word_id_to_token_ids,
            word_id_to_text,
        ) = self._predict_with_whitespace(doc=doc)
        self._validate_token_word_assignments(word_id_to_token_ids=word_id_to_token_ids)

        # split up words back into tokens based on punctuation
        (
            token_id_to_word_id,
            word_id_to_token_ids,
            word_id_to_text,
        ) = self._keep_punct_as_words(
            doc=doc,
            word_id_to_token_ids=word_id_to_token_ids,
            punct_as_words=self.punct_as_words,
        )
        self._validate_token_word_assignments(word_id_to_token_ids=word_id_to_token_ids)

        # get hyphen word candidates
        hyphen_word_candidates = self._find_hyphen_word_candidates(
            tokens=doc.tokens,
            token_id_to_word_id=token_id_to_word_id,
            word_id_to_token_ids=word_id_to_token_ids,
            word_id_to_text=word_id_to_text,
        )
        candidate_texts = [
            word_id_to_text[prefix_word_id] + word_id_to_text[suffix_word_id]
            for prefix_word_id, suffix_word_id in hyphen_word_candidates
        ]

        # filter candidate texts
        hyphen_word_candidates, candidate_texts = self._filter_word_candidates(
            hyphen_word_candidates=hyphen_word_candidates,
            candidate_texts=candidate_texts,
        )

        # only triggers if there are hyphen word candidates
        if hyphen_word_candidates:
            # classify hyphen words
            results = self.classifier.batch_predict(words=candidate_texts, threshold=self.threshold)

            # update output data based on hyphen word candidates
            # first, we concatenate words based on prefix + suffix. this includes hyphen.
            # second, we modify the text value (e.g. remove hyphens) if classifier says.
            for (prefix_word_id, suffix_word_id), result in zip(hyphen_word_candidates, results):
                impacted_token_ids = word_id_to_token_ids[prefix_word_id] + word_id_to_token_ids[suffix_word_id]
                word_id_to_token_ids[prefix_word_id] = impacted_token_ids
                word_id_to_token_ids.pop(suffix_word_id)
                word_id_to_text[prefix_word_id] += word_id_to_text[suffix_word_id]
                word_id_to_text.pop(suffix_word_id)
                if result.is_edit is True:
                    word_id_to_text[prefix_word_id] = result.new
            token_id_to_word_id = {
                token_id: word_id for word_id, token_ids in word_id_to_token_ids.items() for token_id in token_ids
            }
            self._validate_token_word_assignments(word_id_to_token_ids=word_id_to_token_ids)

        # make into spangroups
        words = self._create_words(
            doc=doc,
            token_id_to_word_id=token_id_to_word_id,
            word_id_to_text=word_id_to_text,
        )
        return words

    def _make_clean_document(self, doc: Document) -> Document:
        """Word predictor doesnt work on documents with poor tokenizations,
        such as when there are empty tokens. This cleans up the document.
        We keep this pretty minimal, such as ignoring Span Boxes since dont need
        them for word prediction."""
        new_tokens = []
        for token in doc.tokens:
            if token.text.strip() != "":
                new_token = Entity(spans=[Span(start=span.start, end=span.end) for span in token.spans])
                new_tokens.append(new_token)
        new_doc = Document(symbols=doc.symbols)
        new_doc.annotate_layer(name="tokens", entities=new_tokens)
        return new_doc

    def _recursively_remove_trailing_hyphens(self, word: str) -> str:
        if word.endswith("-"):
            return self._recursively_remove_trailing_hyphens(word=word[:-1])
        else:
            return word

    def _validate_tokenization(self, doc: Document):
        """This Word Predictor relies on a specific type of Tokenization
        in which hyphens ('-') must be their own token. This verifies.

        Additionally, doesnt work unless there's an `.id` field on each token.
        See `_cluster_tokens_by_whitespace()` for more info.
        """
        for token in doc.tokens:
            if "-" in token.text and token.text != "-":
                raise ValueError(f"Document contains Token with hyphen, but not as its own token.")
            if token.id is None:
                raise ValueError(
                    f"Document contains Token without an `.id` field, which is necessary for this word Predictor's whitespace clustering operation."
                )

            if token.text.strip() == "":
                raise ValueError(f"Document contains Token with empty text, which is not allowed.")

    def _validate_token_word_assignments(self, word_id_to_token_ids, allow_missed_tokens: bool = True):
        for word_id, token_ids in word_id_to_token_ids.items():
            start = min(token_ids)
            end = max(token_ids)
            assert len(token_ids) == end - start + 1, f"word={word_id} comprised of disjoint token_ids={token_ids}"

        if not allow_missed_tokens:
            all_token_ids = {token_id for token_id in word_id_to_token_ids.values() for token_id in token_ids}
            if len(all_token_ids) < max(all_token_ids):
                raise ValueError(f"Not all tokens are assigned to a word.")

    def _cluster_tokens_by_whitespace(self, doc: Document) -> List[List[int]]:
        """
        `whitespace_tokenization` is necessary because lack of whitespace is an indicator that
        adjacent tokens belong in a word together.
        """
        _ws_tokens: List[Entity] = self.whitespace_predictor.predict(doc=doc)
        doc.annotate_layer(name="_ws_tokens", entities=_ws_tokens)

        # token -> ws_tokens
        token_id_to_ws_token_id = {}
        for token in doc.tokens:
            overlap_ws_tokens = token._ws_tokens
            if overlap_ws_tokens:
                token_id_to_ws_token_id[token.id] = overlap_ws_tokens[0].id

        # ws_token -> tokens
        ws_token_id_to_tokens = defaultdict(list)
        for token_id, ws_token_id in token_id_to_ws_token_id.items():
            ws_token_id_to_tokens[ws_token_id].append(token_id)

        # cluster tokens by whitespace
        clusters = []
        for ws_token_id, tokens in ws_token_id_to_tokens.items():
            clusters.append(sorted(tokens))

        # cleanup
        doc.remove_layer(name="_ws_tokens")
        return clusters

    def _predict_with_whitespace(self, doc: Document):
        """Predicts word boundaries using whitespace tokenization."""
        # precompute whitespace tokenization
        whitespace_clusters = self._cluster_tokens_by_whitespace(doc=doc)
        # assign word ids
        token_id_to_word_id = {}
        word_id_to_token_ids = defaultdict(list)
        for word_id, token_ids_in_cluster in enumerate(whitespace_clusters):
            for token_id in token_ids_in_cluster:
                token_id_to_word_id[token_id] = word_id
                word_id_to_token_ids[word_id].append(token_id)
        # get word strings
        word_id_to_text = {}
        for word_id, token_ids in word_id_to_token_ids.items():
            word_id_to_text[word_id] = "".join(doc.tokens[token_id].text for token_id in token_ids)
        return token_id_to_word_id, word_id_to_token_ids, word_id_to_text

    def _keep_punct_as_words(self, doc: Document, word_id_to_token_ids: Dict, punct_as_words: str):
        # keep track of which tokens are punctuation
        token_ids_are_punct = set()
        for token_id, token in enumerate(doc.tokens):
            if token.text in punct_as_words:
                token_ids_are_punct.add(token_id)
        # re-cluster punctuation tokens into their own words
        new_clusters = []
        for old_cluster in word_id_to_token_ids.values():
            for new_cluster in self._group_adjacent_with_exceptions(
                adjacent=old_cluster, exception_ids=token_ids_are_punct
            ):
                new_clusters.append(new_cluster)
        # reorder
        new_clusters = sorted(new_clusters, key=lambda x: min(x))
        # reassign word ids
        new_token_id_to_word_id = {}
        new_word_id_to_token_ids = defaultdict(list)
        for word_id, token_ids_in_cluster in enumerate(new_clusters):
            for token_id in token_ids_in_cluster:
                new_token_id_to_word_id[token_id] = word_id
                new_word_id_to_token_ids[word_id].append(token_id)
        # get word strings
        new_word_id_to_text = {}
        for word_id, token_ids in new_word_id_to_token_ids.items():
            new_word_id_to_text[word_id] = "".join(doc.tokens[token_id].text for token_id in token_ids)
        return new_token_id_to_word_id, new_word_id_to_token_ids, new_word_id_to_text

    def _group_adjacent_with_exceptions(self, adjacent: List[int], exception_ids: Set[int]) -> List[List[int]]:
        result = []
        group = []
        for e in adjacent:
            if e in exception_ids:
                if group:
                    result.append(group)
                result.append([e])
                group = []
            else:
                group.append(e)
        if group:
            result.append(group)
        return result

    def _find_hyphen_word_candidates(
        self,
        tokens,
        token_id_to_word_id,
        word_id_to_token_ids,
        word_id_to_text,
    ) -> Tuple[int, int]:
        """Finds the IDs of hyphenated words (in prefix + suffix format)."""
        # get all hyphen tokens
        # TODO: can refine this further by restricting to only tokens at end of `rows`
        hyphen_token_ids = []
        for token_id, token in enumerate(tokens):
            if token.text == "-":
                hyphen_token_ids.append(token_id)
        # get words that contain hyphen token, but only at the end (i.e. broken word)
        # these form the `prefix` of a potential hyphenated word
        #
        # edge case: sometimes the prefix word is *just* a hyphen. this means the word
        # itself is actually just a hyphen (e.g. like in tables of results)
        # dont consider these as candidates
        prefix_word_ids = set()
        for hyphen_token_id in hyphen_token_ids:
            prefix_word_id = token_id_to_word_id[hyphen_token_id]
            prefix_word_text = word_id_to_text[prefix_word_id]
            if prefix_word_text.endswith("-") and not prefix_word_text == "-":
                prefix_word_ids.add(prefix_word_id)
        # get words right after the prefix (assumed words in reading order)
        # these form the `suffix` of a potential hyphenated word
        # together, a `prefix` and `suffix` form a candidate pair
        #
        # edge case: sometimes the token stream ends with a hyphenated
        # word. this means suffix_word_id wont exist. dont consider these
        word_id_pairs = []
        for prefix_word_id in prefix_word_ids:
            suffix_word_id = prefix_word_id + 1
            suffix_word_text = word_id_to_text.get(suffix_word_id)
            if suffix_word_text is None:
                continue
            word_id_pairs.append((prefix_word_id, suffix_word_id))
        return sorted(word_id_pairs)

    def _filter_word_candidates(self, hyphen_word_candidates: list, candidate_texts: list) -> tuple:
        hyphen_word_candidates_filtered = []
        candidate_texts_filtered = []
        for hyphen_word_candidate, candidate_text in zip(hyphen_word_candidates, candidate_texts):
            if candidate_text.endswith("-") or candidate_text.startswith("-"):
                continue
            else:
                hyphen_word_candidates_filtered.append(hyphen_word_candidate)
                candidate_texts_filtered.append(candidate_text)
        return hyphen_word_candidates_filtered, candidate_texts_filtered

    def _create_words(self, doc: Document, token_id_to_word_id, word_id_to_text) -> List[Entity]:
        words = []
        tokens_in_word = [doc.tokens[0]]
        current_word_id = 0
        new_word_id = 0
        for token_id in range(1, len(doc.tokens)):
            token = doc.tokens[token_id]
            word_id = token_id_to_word_id.get(token_id)
            if word_id is None:
                logger.debug(f"Token {token_id} has no word ID. Likely PDF Parser produced empty tokens.")
                continue
            if word_id == current_word_id:
                tokens_in_word.append(token)
            else:
                spans = [
                    Span.create_enclosing_span(spans=[span for token in tokens_in_word for span in token.spans])
                ]
                metadata = Metadata(text=word_id_to_text[current_word_id]) if len(tokens_in_word) > 1 else None
                word = Entity(
                    spans=spans,
                    metadata=metadata,
                )
                words.append(word)
                tokens_in_word = [token]
                current_word_id = word_id
                new_word_id += 1
        # last bit
        spans = [Span.create_enclosing_span(spans=[span for token in tokens_in_word for span in token.spans])]
        metadata = Metadata(text=word_id_to_text[current_word_id]) if len(tokens_in_word) > 1 else None
        word = Entity(
            spans=spans,
            metadata=metadata,
        )
        words.append(word)
        new_word_id += 1
        return words
