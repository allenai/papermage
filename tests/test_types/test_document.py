"""

@kylel

"""

import unittest

from papermage.types import (
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
