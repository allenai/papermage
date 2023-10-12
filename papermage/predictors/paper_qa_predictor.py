from typing import List, Set

import numpy as np
import pysbd
from tokreate import CallAction, ParseAction

from papermage.magelib import Document, Entity, ParagraphsFieldName, SentencesFieldName
from papermage.predictors import BasePredictor

from .utils_paper_qa.hashing import create_hash, int_to_bin, similarity

FULL_DOC_QA_ATTRIBUTED_PROMPT = """\
Answer a question using the provided scientific paper.
Your response should be a JSON object with the following fields:
  - answer: The answer to the question. The answer should use concise language, but be comprehensive. Only provide answers that are objectively supported by the text in paper.
  - excerpts: A list of one or more *EXACT* text spans extracted from the paper that support the answer. Return between at most ten spans, and no more that 800 words. Make sure to cover all aspects of the answer above.
If there is no answer, return an empty dictionary, i.e., `{}`.

Paper:
{{ full_text }}

Given the information above, please answer the question: "{{ question }}"."""

FULL_DOC_QA_SYSTEM_JSON_PROMPT = """\
You are a helpful research assistant, answering questions about scientific papers accurately and concisely.
You ONLY respond to questions that have an objective answer, and return an empty response for subjective requests.
You always return a valid JSON object to each user request."""


class PaperQaPredictor(BasePredictor):
    def __init__(self, model_name: str = "gpt-3.5-turbo-16k", max_tokens: int = 2048):
        self.call = CallAction(
            prompt=FULL_DOC_QA_ATTRIBUTED_PROMPT,
            system=FULL_DOC_QA_SYSTEM_JSON_PROMPT,
            model=model_name,
            parameters={"max_tokens": max_tokens},
        ) >> ParseAction(name="json_parser", parser="json.loads")
        self.sentencizer = pysbd.Segmenter(language="en", clean=False)

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [ParagraphsFieldName, SentencesFieldName]

    def merge_adjacent_sentences(self, locs: List[int], slack: int = 1, max_len: int = 3) -> List[List[int]]:
        """Merge adjacent sentences that are within a certain distance of each other."""
        seen: Set[int] = set()
        grouped: List[List[int]] = []
        end_pos = len(locs) - 1
        for i in range(len(locs)):
            init_pos = locs[i]
            if init_pos in seen:
                grouped.append([])
            elif i < end_pos:
                for j in range(i + 1, len(locs)):
                    curr_pos, prev_pos = locs[j], locs[j - 1]
                    if (curr_pos - prev_pos) > slack or (curr_pos - init_pos) >= max_len or j == end_pos:
                        new_pos = list(range(init_pos, prev_pos + 1))
                        grouped.append(new_pos)
                        seen.update(new_pos)
                        break
            else:
                grouped.append([init_pos])
                seen.add(init_pos)

        return grouped

    def predict(self, doc: Document, *args, **kwargs) -> List[Entity]:
        # TODO: fix base predictor so that it can handle questions
        self._doc_field_checker(doc)
        return self._predict(doc, *args, **kwargs)

    def _predict(self, doc: Document, question: str) -> List[Entity]:  # type: ignore
        full_text = ""
        sentences_vecs = []
        sentences_ids = []

        # build full text representation and hash sentences for similarity
        for i, paragraph in enumerate(doc.paragraphs):  # type: ignore
            header = getattr(paragraph.metadata, "header_text", None)
            for j, sent in enumerate(paragraph.sentences):
                if j == 0 and (header := getattr(sent.metadata, "header_text", header)):
                    # add a header if it exists
                    full_text += f"\n\n## {header}\n"
                else:
                    full_text += "\n\n"

                full_text += f"{sent.text.strip()} "
                sentences_vecs.append(int_to_bin(create_hash(sent.text)))
                sentences_ids.append((i, j))
            full_text.strip()
        sentences_array = np.vstack(sentences_vecs)

        *_, output = self.call.run(full_text=full_text, question=question)
        parsed_output = output.state["json_parser"]

        if not parsed_output:
            # the model could not answer the question
            return []

        excerpts = [s for e in parsed_output.get("excerpts", []) for s in self.sentencizer.segment(e)]

        if not excerpts:
            # the model could not find supporting evidence
            return []

        encoded_context = np.vstack([int_to_bin(create_hash(e)) for e in excerpts])
        similarities = similarity(queries=encoded_context, targets=sentences_array)
        grouped_locs = self.merge_adjacent_sentences(np.argmax(similarities, axis=1))

        extracted_excerpts: List[Entity] = []
        for locs in grouped_locs:
            if len(locs) == 0:
                # nothing matched for this group
                continue

            ids = [sentences_ids[loc] for loc in locs]
            sents = [doc.paragraphs[i].sentences[j] for i, j in ids]  # type: ignore
            for sent in sents:
                matched_sent = Entity.from_json(sent.to_json())
                matched_sent.metadata.score = np.max(similarities[:, locs]).tolist()
                matched_sent.metadata.answer = parsed_output["answer"]
                extracted_excerpts.append(matched_sent)

        return extracted_excerpts
