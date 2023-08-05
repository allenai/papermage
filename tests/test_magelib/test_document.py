"""

@kylel

"""

import unittest

from papermage.magelib import (
    Document,
    EntitiesFieldName,
    Entity,
    MetadataFieldName,
    RelationsFieldName,
    SymbolsFieldName,
)


class TestDocument(unittest.TestCase):
    def test_annotate(self):
        doc = Document("This is a test document!")
        tokens = [
            Entity.from_json({"spans": [[0, 4]]}),
            Entity.from_json({"spans": [[5, 7]]}),
            Entity.from_json({"spans": [[8, 9]]}),
            Entity.from_json({"spans": [[10, 14]]}),
            Entity.from_json({"spans": [[15, 23]]}),
            Entity.from_json({"spans": [[23, 24]]}),
        ]
        with self.assertRaises(AttributeError) as e:
            doc.tokens
        # annotate
        doc.annotate_entity(field_name="tokens", entities=tokens)
        self.assertEqual(len(doc.tokens), len(tokens))
        for t1, t2 in zip(doc.tokens, tokens):
            self.assertEqual(t1, t2)
        # get
        self.assertEqual(doc.tokens, doc.get_entity(field_name="tokens"))
        # remove
        doc.remove_entity(field_name="tokens")
        with self.assertRaises(AttributeError) as e:
            doc.tokens

    def test_empty_annotations_work(self):
        doc = Document("This is a test document!")
        doc.annotate_entity(field_name="my_cool_field", entities=[])
        self.assertEqual(doc.my_cool_field, [])

    def test_metadata_serializes(self):
        symbols = "Hey there y'all!"
        doc = Document(symbols=symbols)
        doc.metadata["a"] = {"b": "c"}
        self.assertDictEqual(
            doc.to_json(),
            {
                SymbolsFieldName: symbols,
                MetadataFieldName: {"a": {"b": "c"}},
                EntitiesFieldName: {},
                RelationsFieldName: {},
            },
        )

    def test_metadata_deserializes(self):
        metadata = {"a": {"b": "c"}}
        symbols = "Hey again peeps!"
        input_json = {
            SymbolsFieldName: symbols,
            MetadataFieldName: metadata,
            EntitiesFieldName: {},
            RelationsFieldName: {},
        }

        doc = Document.from_json(input_json)

        self.assertEqual(symbols, doc.symbols)
        self.assertDictEqual(metadata, doc.metadata.to_json())

    def test_metadata_deserializes_when_empty(self):
        symbols = "That's all folks!"
        input_json = {
            SymbolsFieldName: symbols,
            MetadataFieldName: {},
            EntitiesFieldName: {},
            RelationsFieldName: {},
        }

        doc = Document.from_json(input_json)

        self.assertEqual(symbols, doc.symbols)
        self.assertEqual(0, len(doc.metadata))

    def test_cross_referencing(self):
        doc = Document("This is a test document!")
        # boxes are in a top-left to bottom-right diagonal fashion (same page)
        tokens = [
            Entity.from_json({"spans": [[0, 4]], "boxes": [[0, 0, 0.5, 0.5, 0]]}),
            Entity.from_json({"spans": [[5, 7]], "boxes": [[1, 1, 0.5, 0.5, 0]]}),
            Entity.from_json({"spans": [[8, 9]], "boxes": [[2, 2, 0.5, 0.5, 0]]}),
            Entity.from_json({"spans": [[10, 14]], "boxes": [[3, 3, 0.5, 0.5, 0]]}),
            Entity.from_json({"spans": [[15, 23]], "boxes": [[4, 4, 0.5, 0.5, 0]]}),
            Entity.from_json({"spans": [[23, 24]], "boxes": [[5, 5, 0.5, 0.5, 0]]}),
        ]
        # boxes are also same diagonal fashion, but bigger.
        # last box super big on wrong page.
        chunks = [
            Entity.from_json({"spans": [[0, 9]], "boxes": [[0, 0, 2.01, 2.01, 0]]}),
            Entity.from_json({"spans": [[12, 23]], "boxes": [[3.0, 3.0, 4.0, 4.0, 0]]}),
            Entity.from_json({"spans": [[23, 24]], "boxes": [[0, 0, 10.0, 10.0, 1]]}),
        ]
        doc.annotate_entity(field_name="tokens", entities=tokens)
        doc.annotate_entity(field_name="chunks", entities=chunks)

        # find by span is the default overload of Entity.__attr__
        self.assertListEqual(doc.chunks[0].tokens, tokens[0:3])
        self.assertListEqual(doc.chunks[1].tokens, tokens[3:5])
        self.assertListEqual(doc.chunks[2].tokens, [tokens[5]])

        # find by span works fine
        self.assertListEqual(doc.chunks[0].tokens, doc.find_by_span(query=doc.chunks[0], field_name="tokens"))
        self.assertListEqual(doc.chunks[1].tokens, doc.find_by_span(query=doc.chunks[1], field_name="tokens"))
        self.assertListEqual(doc.chunks[2].tokens, doc.find_by_span(query=doc.chunks[2], field_name="tokens"))

        # find by box
        self.assertListEqual(doc.find_by_box(query=doc.chunks[0], field_name="tokens"), doc.tokens[0:3])
        self.assertListEqual(doc.find_by_box(query=doc.chunks[1], field_name="tokens"), doc.tokens[3:6])
        self.assertListEqual(doc.find_by_box(query=doc.chunks[2], field_name="tokens"), [])
