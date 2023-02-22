import unittest

from mmda.types.document import Document
from mmda.types.names import MetadataField, SymbolsField


class TestDocument(unittest.TestCase):
    def test__empty_annotations_work(self):
        doc = Document("This is a test document!")
        annotations = []
        doc.annotate(my_cool_field=annotations)
        self.assertEqual(doc.my_cool_field, [])

    def test_metadata_serializes(self):
        metadata = {"a": {"b": "c"}}
        symbols = "Hey there y'all!"
        doc = Document(symbols=symbols)
        doc.add_metadata(**metadata)

        output_json = doc.to_json()
        self.assertDictEqual(
            {SymbolsField: symbols, MetadataField: metadata}, output_json
        )

    def test_metadata_deserializes(self):
        metadata = {"a": {"b": "c"}}
        symbols = "Hey again peeps!"
        input_json = {SymbolsField: symbols, MetadataField: metadata}

        doc = Document.from_json(input_json)

        self.assertEqual(symbols, doc.symbols)
        self.assertDictEqual(metadata, doc.metadata.to_json())

    def test_metadata_deserializes_when_empty(self):
        symbols = "That's all folks!"
        input_json = {SymbolsField: symbols}

        doc = Document.from_json(input_json)

        self.assertEqual(symbols, doc.symbols)
        self.assertEqual(0, len(doc.metadata))
