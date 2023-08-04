import json
from typing import List

from decontext.data_types import (
    EvidenceParagraph,
    PaperContext,
    PaperSnippet,
    QuestionAnswerEvidence,
)
from decontext.step.qa import TemplateRetrievalQAStep

from papermage.magelib import Annotation, Document
from papermage.predictors.base_predictor import BasePredictor


class APISpanQAPredictor(BasePredictor):
    REQUIRED_BACKENDS = ["transformers", "torch", "decontext"]

    def __init__(self, context_unit_name: str = "paragraph", span_name: str = "user_selected_span"):
        self.context_unit_name = context_unit_name
        self.span_name = span_name

        self.retrieval_qa_step = TemplateRetrievalQAStep()

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [self.context_unit_name, self.span_name]

    def preprocess(self, doc: Document) -> PaperSnippet:
        paper_context = PaperContext.parse_raw(
            json.dumps(
                {
                    "title": getattr(doc, "title") if "title" in doc.fields else "",
                    "abstract": getattr(doc, "abstract") if "abstract" in doc.fields else "",
                    "full_text": [
                        {
                            "section_name": "",
                            "paragraphs": [entity.text for entity in getattr(doc, self.context_unit_name)],
                        }
                    ],
                }
            )
        )

        paper_snippet = PaperSnippet(
            snippet=getattr(doc, self.span_name)[0].text,
            context=paper_context,
            qae=[],
            paragraph_with_snippet=EvidenceParagraph(
                section="",
                paragraph=getattr(getattr(doc, self.span_name)[0], self.context_unit_name)[0].text,
            ),
        )

        paper_snippet.add_question(getattr(doc, self.span_name)[0].metadata["question"])

        return paper_snippet

    def _predict(self, doc: Document) -> List[Annotation]:
        # preprocess to format this for decontext
        paper_snippet = self.preprocess(doc)

        # run the decontext qa step
        self.retrieval_qa_step.run(paper_snippet)

        # postprocess to format back into papermage doc format
        user_selected_span = getattr(doc, self.span_name)

        user_selected_span[0].metadata["answer"] = paper_snippet.qae[0].answer
        user_selected_span[0].metadata["context_with_span"] = paper_snippet.paragraph_with_snippet.dict()
        user_selected_span[0].metadata["retrieved_evidence"] = [ev.dict() for ev in paper_snippet.qae[0].evidence]

        return user_selected_span
