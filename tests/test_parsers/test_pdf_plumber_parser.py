"""

@kylel, benjaminn
"""

import json
import os
import pathlib
import re
import unittest

import numpy as np

from papermage.parsers import PDFPlumberParser
from papermage.types import Box, Document, Entity, Span


class TestPDFPlumberParser(unittest.TestCase):
    def setUp(cls) -> None:
        cls.fixture_path = pathlib.Path(__file__).parent.parent / "fixtures"

    '''
    def test_parse(self):
        parser = PDFPlumberParser()
        doc = parser.parse(input_pdf_path=self.fixture_path / "1903.10676.pdf")
        # right output type
        assert isinstance(doc, Document)
        # the right fields
        assert doc.symbols
        assert doc.pages
        assert doc.tokens
        assert doc.rows
        # roughly the right content
        for keyword in ["Field", "Task", "SOTA", "Base", "Frozen", "Finetune", "NER"]:
            assert keyword in doc.symbols[:100]

    def test_parse_page_dims(self):
        parser = PDFPlumberParser()
        doc = parser.parse(input_pdf_path=self.fixture_path / "1903.10676.pdf")

        for page in doc.pages:
            self.assertEqual(595.0, page.metadata.width)
            self.assertEqual(842.0, page.metadata.height)
            self.assertEqual(1.0, page.metadata.user_unit)

    def test_non_default_user_unit(self):
        parser = PDFPlumberParser()
        doc = parser.parse(input_pdf_path=self.fixture_path / "test-uu.pdf")

        for page in doc.pages:
            self.assertEqual(595.0, page.metadata.width)
            self.assertEqual(842.0, page.metadata.height)
            self.assertEqual(2.0, page.metadata.user_unit)


    def test_parse_fontinfo(self):
        parser = PDFPlumberParser()
        doc = parser.parse(input_pdf_path=self.fixture_path / "1903.10676.pdf")

        metadata = doc.tokens[0].metadata  # pylint: disable=no-member

        self.assertEqual("HXONRZ+NimbusRomNo9L-Regu", metadata["fontname"])
        self.assertAlmostEqual(8.96638, metadata["size"])

    def test_split_punctuation(self):
        no_split_parser = PDFPlumberParser(split_at_punctuation=False)
        no_split_doc = no_split_parser.parse(
            input_pdf_path=self.fixture_path / "2107.07170.pdf"
        )
        no_split_tokens_with_numbers = [
            token.text
            for token in no_split_doc.tokens
            if re.search(r"[0-9]", token.text)
        ]
        assert "[1-5]" in no_split_tokens_with_numbers
        assert "GPT-3[10]" in no_split_tokens_with_numbers

        custom_split_parser = PDFPlumberParser(split_at_punctuation=",.[]:")
        custom_split_doc = custom_split_parser.parse(
            input_pdf_path=self.fixture_path / "2107.07170.pdf"
        )
        custom_split_tokens_with_numbers = [
            token.text
            for token in custom_split_doc.tokens
            if re.search(r"[0-9]", token.text)
        ]
        assert "[1-5]" not in custom_split_tokens_with_numbers
        assert "1-5" in custom_split_tokens_with_numbers
        assert "GPT-3[10]" not in custom_split_tokens_with_numbers
        assert "GPT-3" in custom_split_tokens_with_numbers

        default_split_parser = PDFPlumberParser(split_at_punctuation=True)
        default_split_doc = default_split_parser.parse(
            input_pdf_path=os.path.join(self.fixture_path, "2107.07170.pdf")
        )
        default_split_tokens_with_numbers = [
            token.text
            for token in default_split_doc.tokens
            if re.search(r"[0-9]", token.text)
        ]
        assert "1-5" not in default_split_tokens_with_numbers
        assert "GPT-3" not in default_split_tokens_with_numbers

        assert (
            len(no_split_tokens_with_numbers)
            < len(custom_split_tokens_with_numbers)
            < len(default_split_tokens_with_numbers)
        )

    def test_align_coarse_and_fine_tokens(self):
        parser = PDFPlumberParser()

        # example
        coarse_tokens = ["abc", "def"]
        fine_tokens = ["ab", "c", "d", "ef"]
        out = parser._align_coarse_and_fine_tokens(
            coarse_tokens=coarse_tokens, fine_tokens=fine_tokens
        )
        assert out == [0, 0, 1, 1]

        # minimal case
        coarse_tokens = []
        fine_tokens = []
        out = parser._align_coarse_and_fine_tokens(
            coarse_tokens=coarse_tokens, fine_tokens=fine_tokens
        )
        assert out == []

        # identical case
        coarse_tokens = ["a", "b", "c"]
        fine_tokens = ["a", "b", "c"]
        out = parser._align_coarse_and_fine_tokens(
            coarse_tokens=coarse_tokens, fine_tokens=fine_tokens
        )
        assert out == [0, 1, 2]

        # misaligned case
        with self.assertRaises(AssertionError):
            coarse_tokens = ["a", "b"]
            fine_tokens = ["ab"]
            parser._align_coarse_and_fine_tokens(
                coarse_tokens=coarse_tokens, fine_tokens=fine_tokens
            )

        # same num of chars, but chars mismatch case
        with self.assertRaises(AssertionError):
            coarse_tokens = ["ab"]
            fine_tokens = ["a", "c"]
            parser._align_coarse_and_fine_tokens(
                coarse_tokens=coarse_tokens, fine_tokens=fine_tokens
            )

    def test_convert_nested_text_to_doc_json(self):
        parser = PDFPlumberParser()

        # example
        token_dicts = [
            {"text": text, "bbox": Box(l=0.0, t=0.1, w=0.2, h=0.3, page=4)}
            for text in ["ab", "c", "d", "ef", "gh", "i", "j", "kl"]
        ]
        word_ids = [0, 0, 1, 2, 3, 4, 5, 5]
        row_ids = [0, 0, 1, 1, 2, 2, 3, 3]
        page_ids = [0, 0, 0, 0, 1, 1, 1, 1]
        page_dims = [(100, 200, 1.), (400, 800, 1.)]
        out = parser._convert_nested_text_to_doc_json(
            token_dicts=token_dicts,
            word_ids=word_ids,
            row_ids=row_ids,
            page_ids=page_ids,
            dims=page_dims,
        )
        assert out["symbols"] == "abc\nd ef\ngh i\njkl"
        tokens = [
            SpanGroup.from_json(span_group_dict=t_dict) for t_dict in out["tokens"]
        ]
        assert [(t.start, t.end) for t in tokens] == [
            (0, 2),
            (2, 3),
            (4, 5),
            (6, 8),
            (9, 11),
            (12, 13),
            (14, 15),
            (15, 17),
        ]
        assert [out["symbols"][t.start : t.end] for t in tokens] == [
            "ab",
            "c",
            "d",
            "ef",
            "gh",
            "i",
            "j",
            "kl",
        ]
        rows = [SpanGroup.from_json(span_group_dict=r_dict) for r_dict in out["rows"]]
        assert [(r.start, r.end) for r in rows] == [(0, 3), (4, 8), (9, 13), (14, 17)]
        assert [out["symbols"][r.start : r.end] for r in rows] == [
            "abc",
            "d ef",
            "gh i",
            "jkl",
        ]
        pages = [SpanGroup.from_json(span_group_dict=p_dict) for p_dict in out["pages"]]
        assert [(p.start, p.end) for p in pages] == [(0, 8), (9, 17)]
        assert [out["symbols"][p.start : p.end] for p in pages] == [
            "abc\nd ef",
            "gh i\njkl",
        ]
    '''

    def test_parser_stability(self):
        """
        We need output to be stable from release to release. Failure of this test is caused
        by changes to core output: document text, tokenization, and bbox localization.
        It deliberately excludes `metadata` from consideration as we are expanding
        its scope of coverage, but that should probably be locked down too the moment
        we depend on particular fields.

        Updates that break this test should be considered potentially breaking to downstream
        models and require re-evaluation and possibly retraining of all components in the DAG.
        """

        parser = PDFPlumberParser()

        current_doc = parser.parse(input_pdf_path=self.fixture_path / "4be952924cd565488b4a239dc6549095029ee578.pdf")

        with open(self.fixture_path / "4be952924cd565488b4a239dc6549095029ee578__pdfplumber_doc.json", "r") as f:
            raw_json = f.read()
            fixture_doc_json = json.loads(raw_json)
            fixture_doc = Document.from_json(fixture_doc_json)


        self.assertEqual(current_doc.symbols, fixture_doc.symbols, msg="Current parse has extracted different text from pdf.")

        def compare_entities(current_doc_ents, fixture_doc_ents, annotation_name):
            current_doc_ents_simplified = [
                [(s.start, s.end) for s in ent.spans] for ent in current_doc_ents
            ]
            fixture_doc_ents_simplified = [
                [(s.start, s.end) for s in ent.spans] for ent in fixture_doc_ents
            ]

            self.assertEqual(
                current_doc_ents_simplified,
                fixture_doc_ents_simplified,
                msg=f"Current parse produces different SpanGroups for `{annotation_name}`"
            )

            current_doc_ent_boxes = [[list(box.xy_coordinates) + [box.page] for box in ent.boxes] for ent in current_doc_ents]
            fixture_doc_ent_boxes = [[list(box.xy_coordinates) + [box.page] for box in ent.boxes] for ent in current_doc_ents]

            self.assertAlmostEqual(
                current_doc_ent_boxes,
                fixture_doc_ent_boxes,
                places=3,
                msg=f"Boxes generated for `{annotation_name}` have changed."
            )

        compare_entities(current_doc.tokens, fixture_doc.tokens, "tokens")
        compare_entities(current_doc.rows, fixture_doc.rows, "rows")
        compare_entities(current_doc.pages, fixture_doc.pages, "pages")
