"""

@kylel

"""

import os
import unittest

from papermage.recipes import CoreRecipe
from papermage.types import Entity, Document, Image
from tests.test_recipes.core_recipe_fixtures import (
    BASE64_PAGE_IMAGE,
    FIRST_3_BLOCKS_JSON,
    FIRST_5_ROWS_JSON,
    FIRST_10_TOKENS_JSON,
    FIRST_10_VILA_JSONS,
    FIRST_1000_SYMBOLS,
    PAGE_JSON,
    # SEGMENT_OF_WORD_JSONS,
)


def round_all_floats(d: dict):
    import numbers

    def formatfloat(x):
        return "%.4g" % float(x)

    def pformat(dictionary, function):
        if isinstance(dictionary, dict):
            return {key: pformat(value, function) for key, value in dictionary.items()}
        if isinstance(dictionary, list):
            return [pformat(element, function) for element in dictionary]
        if isinstance(dictionary, numbers.Number):
            return function(dictionary)
        return dictionary

    return pformat(d, formatfloat)


class TestCoreRecipe(unittest.TestCase):
    def setUp(self):
        self.pdfpath = os.path.join(
            os.path.dirname(__file__), "../fixtures/1903.10676.pdf"
        )
        self.recipe = CoreRecipe()
        self.doc = self.recipe.from_path(pdfpath=self.pdfpath)

    def test_correct_output(self):
        self.assertEqual(self.doc.symbols[:1000], FIRST_1000_SYMBOLS)
        self.assertDictEqual(self.doc.pages[0].to_json(), PAGE_JSON)
        self.assertEqual(self.doc.images[0].to_base64(), BASE64_PAGE_IMAGE)
        self.assertListEqual(
            [round_all_floats(t.to_json()) for t in self.doc.tokens[:10]],
            round_all_floats(FIRST_10_TOKENS_JSON),
        )
        self.assertListEqual(
            [round_all_floats(r.to_json()) for r in self.doc.rows[:5]],
            round_all_floats(FIRST_5_ROWS_JSON),
        )

        self.assertListEqual(
            [round_all_floats(b.to_json()) for b in self.doc.blocks[:3]],
            round_all_floats(FIRST_3_BLOCKS_JSON),
        )
        self.assertListEqual(
            [round_all_floats(v.to_json()) for v in self.doc.vila_entities[:10]],
            round_all_floats(FIRST_10_VILA_JSONS),
        )
        # self.assertListEqual(
        #     [round_all_floats(w.to_json()) for w in self.doc.words[895:900]],
        #     round_all_floats(SEGMENT_OF_WORD_JSONS),
        # )


    def test_manual_create_using_annotate(self):
        """
        This tests whether one can manually reconstruct a Document without using from_json().
        Annotations on a Document are order-invariant once created, so you can see this since the
        fields are being annotated in a different order than they were computed.
        """
        doc_json = self.doc.to_json(with_images=True)

        doc2 = Document(symbols=doc_json["symbols"], metadata=doc_json["metadata"])
        assert doc2.symbols == doc_json["symbols"] == self.doc.symbols
        assert (
            doc2.metadata.to_json()
            == doc_json["metadata"]
            == self.doc.metadata.to_json()
        )

        images = [Image.from_base64(img) for img in doc_json["images"]]
        doc2.annotate_images(images)
        assert (
            doc2.images[0].to_base64()
            == doc_json["images"][0]
            == self.doc.images[0].to_base64()
        )

        rows = [Entity.from_json(entity_json=r) for r in doc_json["entities"]["rows"]]
        doc2.annotate_entity(field_name="rows", entities=rows)
        assert (
            [r.to_json() for r in doc2.rows]
            == doc_json["entities"]["rows"]
            == [r.to_json() for r in self.doc.rows]
        )

        vila_entities = [
            Entity.from_json(entity_json=v) for v in doc_json["entities"]["vila_entities"]
        ]
        doc2.annotate_entity(field_name="vila_entities", entities=vila_entities)
        assert (
            [v.to_json() for v in doc2.vila_entities]
            == doc_json["entities"]["vila_entities"]
            == [v.to_json() for v in self.doc.vila_entities]
        )

        # words = [Entity.from_json(entity_json=w) for w in doc_json["entities"]["words"]]
        # doc2.annotate(words=words)
        # assert (
        #     [w.to_json() for w in doc2.words]
        #     == doc_json["entities"]["words"]
        #     == [w.to_json() for w in self.doc.words]
        # )

        tokens = [Entity.from_json(entity_json=t) for t in doc_json["entities"]["tokens"]]
        doc2.annotate_entity(field_name="tokens", entities=tokens)
        assert (
            [t.to_json() for t in doc2.tokens]
            == doc_json["entities"]["tokens"]
            == [t.to_json() for t in self.doc.tokens]
        )

        blocks = [Entity.from_json(entity_json=b) for b in doc_json["entities"]["blocks"]]
        doc2.annotate_entity(field_name="blocks", entities=blocks)
        assert (
            [b.to_json() for b in doc2.blocks]
            == doc_json["entities"]["blocks"]
            == [b.to_json() for b in self.doc.blocks]
        )
