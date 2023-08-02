import json
import os
import pathlib
import unittest

from papermage.types import Document, Span, Entity
from papermage.predictors.span_qa_predictor import SpanQAPredictor

class TestSpanQAPredictor(unittest.TestCase):
    def setUp(self):
        self.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"
        with open(self.fixture_path / "2304.02623v1.json", "r") as f:
            test_doc_json = json.load(f)
        self.doc = Document.from_json(doc_json=test_doc_json)
        
        user_selected_span = Entity(spans=[Span(start=2784, end=2803)], metadata={"question": "What does this mean?"})
        self.doc.annotate_entity(field_name="user_selected_span", entities=[user_selected_span])

        self.span_qa_predictor = SpanQAPredictor(
            context_unit_name="rows",
        )

        # Should throw an error if the GPT4 prompt hasn't been cached
        self.span_qa_predictor.retrieval_qa_step.model.cache.enforce_cached=True

        self.using_github_actions = (
            "USING_GITHUB_ACTIONS" in os.environ and os.environ["USING_GITHUB_ACTIONS"] == "true"
        )

    def test_preprocess(self):
        paper_snippet = self.span_qa_predictor.preprocess(self.doc)
        assert paper_snippet.snippet == "AI-assisted writing"
        assert paper_snippet.qae[0].question == "What does this mean?"

    def test_retrieval(self):
        if self.using_github_actions:
            self.skipTest(
                "Skipping test_retrieval because it requires downloading and running a huggingface model."
            )
        paper_snippet = self.span_qa_predictor.preprocess(self.doc)

        self.span_qa_predictor.retrieval_qa_step.retrieve(paper_snippet)

        ev = [ev.paragraph for ev in paper_snippet.qae[0].evidence]
        assert ev  == [
            "knowledge or information. For example, this could be researchers",
            "Position-Sensitive Definitions of Terms and Symbols. Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems (2020).",
            "[1] Griffin Adams, Emily Alsentzer, Mert Ketenci, Jason Zucker, and Noémie El- hadad. 2021. What’s in a Summary? Laying the Groundwork for Advances in",
        ]

    def test_predict(self):
        if self.using_github_actions:
            self.skipTest(
                "Skipping test_retrieval because it requires an openai key."
            )

        new_span = self.span_qa_predictor.predict(doc=self.doc)

        assert new_span[0].metadata["context_with_span"] == {'section': '', 'paragraph': '1 In the remainder of this work, unless otherwise specified, AI-assisted writing refers to the use of LLMs to support writing.', 'paper_id': None}
        assert new_span[0].metadata["retrieved_evidence"] == [
            {'section': '', 'paragraph': 'knowledge or information. For example, this could be researchers', 'paper_id': None},
            {'section': '', 'paragraph': 'Position-Sensitive Definitions of Terms and Symbols. Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems (2020).', 'paper_id': None},
            {'section': '', 'paragraph': '[1] Griffin Adams, Emily Alsentzer, Mert Ketenci, Jason Zucker, and Noémie El- hadad. 2021. What’s in a Summary? Laying the Groundwork for Advances in', 'paper_id': None},
        ]

        assert new_span[0].metadata["answer"] == 'The term "AI-assisted writing" refers to the utilization of Language Models (LLMs) to aid in writing tasks, as specified in the provided text.'