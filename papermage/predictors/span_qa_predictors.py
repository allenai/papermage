"""

QA given an Entity of interest (with span)

@benjaminn

"""

import json
from typing import List

from decontext.data_types import (
    EvidenceParagraph,
    PaperContext,
    PaperSnippet,
    QuestionAnswerEvidence,
)
from decontext.step.qa import TemplateRetrievalQAStep

from papermage.magelib import Document, Entity
from papermage.predictors import BasePredictor


class APISpanQAPredictor(BasePredictor):
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
                    "title": getattr(doc, "title") if "title" in doc.layers else "",
                    "abstract": getattr(doc, "abstract") if "abstract" in doc.layers else "",
                    "full_text": [
                        {
                            "section_name": "",
                            "paragraphs": [entity.text for entity in getattr(doc, self.context_unit_name)],
                        }
                    ],
                }
            )
        )

        context = getattr(getattr(doc, self.span_name)[0], self.context_unit_name)[0]
        # Get the index of the context with the span
        if context.id is not None:
            context_index = context.id
        else:
            context_index = [i for i, c in enumerate(doc.get_layer(self.context_unit_name)) if c == context][0]

        paper_snippet = PaperSnippet(
            snippet=getattr(doc, self.span_name)[0].text,
            context=paper_context,
            qae=[],
            paragraph_with_snippet=EvidenceParagraph(
                section="",
                paragraph=context.text,
                index=context_index,
            ),
        )

        paper_snippet.add_question(getattr(doc, self.span_name)[0].metadata["question"])

        return paper_snippet

    def _predict(self, doc: Document) -> List[Entity]:
        # preprocess to format this for decontext
        paper_snippet = self.preprocess(doc)

        # run the decontext qa step
        self.retrieval_qa_step.run(paper_snippet)

        # postprocess to format back into papermage doc format
        entities: List[Entity] = []

        # add the question and answer to a copy of the user selected span
        user_selected_span = getattr(doc, self.span_name)
        new_user_selected_span = Entity.from_json(user_selected_span[0].to_json())
        new_user_selected_span.metadata["type"] = "answer"
        new_user_selected_span.metadata["question"] = paper_snippet.qae[0].question
        new_user_selected_span.metadata["answer"] = paper_snippet.qae[0].answer
        entities.append(new_user_selected_span)

        # add the context with span
        context_with_span = getattr(getattr(doc, self.span_name)[0], self.context_unit_name)[0]
        # context_with_span = doc.get_entity(self.context_unit_name)[paper_snippet.paragraph_with_snippet.index]
        new_context_with_span = Entity.from_json(context_with_span.to_json())
        new_context_with_span.metadata["type"] = "context_with_span"
        entities.append(new_context_with_span)

        # add the evidence entities
        for evidence in paper_snippet.qae[0].evidence:
            entity = doc.get_layer(self.context_unit_name)[evidence.index]
            new_entity = Entity.from_json(entity.to_json())
            new_entity.metadata["type"] = "evidence"
            entities.append(new_entity)

        return entities
