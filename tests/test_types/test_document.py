"""

@kylel

"""

import unittest

from papermage.types import Document, MetadataFieldName, SymbolsFieldName, \
    EntitiesFieldName, RelationsFieldName


class TestDocument(unittest.TestCase):
    def test_empty_annotations_work(self):
        doc = Document("This is a test document!")
        doc.annotate_entity(field_name='my_cool_field', entities=[])
        self.assertEqual(doc.my_cool_field, [])

    def test_metadata_serializes(self):
        symbols = "Hey there y'all!"
        doc = Document(symbols=symbols)
        doc.metadata['a'] = {'b': 'c'}
        self.assertDictEqual(
            doc.to_json(),
            {SymbolsFieldName: symbols, MetadataFieldName: {'a': {'b': 'c'}},
             EntitiesFieldName: {}, RelationsFieldName: {}}
        )

    def test_metadata_deserializes(self):
        metadata = {"a": {"b": "c"}}
        symbols = "Hey again peeps!"
        input_json = {SymbolsFieldName: symbols, MetadataFieldName: metadata,
                      EntitiesFieldName: {}, RelationsFieldName: {}}

        doc = Document.from_json(input_json)

        self.assertEqual(symbols, doc.symbols)
        self.assertDictEqual(metadata, doc.metadata.to_json())

    def test_metadata_deserializes_when_empty(self):
        symbols = "That's all folks!"
        input_json = {SymbolsFieldName: symbols, MetadataFieldName: {},
                      EntitiesFieldName: {}, RelationsFieldName: {}}

        doc = Document.from_json(input_json)

        self.assertEqual(symbols, doc.symbols)
        self.assertEqual(0, len(doc.metadata))
