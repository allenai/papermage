"""

@kylel

"""

import unittest

from papermage.magelib import (
    Document,
    EntitiesFieldName,
    Entity,
    Layer,
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
        doc.annotate_layer(name="tokens", entities=tokens)
        self.assertEqual(len(doc.tokens), len(tokens))
        for t1, t2 in zip(doc.tokens, tokens):
            self.assertEqual(t1, t2)
        # get
        self.assertEqual(doc.tokens, doc.get_layer(name="tokens"))
        # remove
        doc.remove_layer(name="tokens")
        with self.assertRaises(AttributeError) as e:
            doc.tokens

    def test_empty_annotations_work(self):
        doc = Document("This is a test document!")
        doc.annotate_layer(name="my_cool_field", entities=[])
        self.assertEqual(doc.my_cool_field, Layer(entities=[]))

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
        doc.annotate_layer(name="tokens", entities=tokens)
        doc.annotate_layer(name="chunks", entities=chunks)

        # find by span is the default overload of Entity.__attr__
        self.assertListEqual(doc.chunks[0].tokens, tokens[0:3])
        self.assertListEqual(doc.chunks[1].tokens, tokens[3:5])
        self.assertListEqual(doc.chunks[2].tokens, [tokens[5]])

        # backwards
        self.assertListEqual(doc.tokens[0].chunks, [chunks[0]])
        self.assertListEqual(doc.tokens[1].chunks, [chunks[0]])
        self.assertListEqual(doc.tokens[2].chunks, [chunks[0]])
        self.assertListEqual(doc.tokens[3].chunks, [chunks[1]])
        self.assertListEqual(doc.tokens[4].chunks, [chunks[1]])
        self.assertListEqual(doc.tokens[5].chunks, [chunks[2]])

        # find by span works fine
        self.assertListEqual(doc.chunks[0].tokens, doc.intersect_by_span(query=doc.chunks[0], name="tokens"))
        self.assertListEqual(doc.chunks[1].tokens, doc.intersect_by_span(query=doc.chunks[1], name="tokens"))
        self.assertListEqual(doc.chunks[2].tokens, doc.intersect_by_span(query=doc.chunks[2], name="tokens"))

        # backwards
        self.assertListEqual(doc.tokens[0].chunks, doc.intersect_by_span(query=doc.tokens[0], name="chunks"))
        self.assertListEqual(doc.tokens[1].chunks, doc.intersect_by_span(query=doc.tokens[1], name="chunks"))
        self.assertListEqual(doc.tokens[2].chunks, doc.intersect_by_span(query=doc.tokens[2], name="chunks"))
        self.assertListEqual(doc.tokens[3].chunks, doc.intersect_by_span(query=doc.tokens[3], name="chunks"))
        self.assertListEqual(doc.tokens[4].chunks, doc.intersect_by_span(query=doc.tokens[4], name="chunks"))
        self.assertListEqual(doc.tokens[5].chunks, doc.intersect_by_span(query=doc.tokens[5], name="chunks"))

        # find by box
        self.assertListEqual(doc.intersect_by_box(query=doc.chunks[0], name="tokens"), doc.tokens[0:3])
        self.assertListEqual(doc.intersect_by_box(query=doc.chunks[1], name="tokens"), doc.tokens[3:6])
        self.assertListEqual(doc.intersect_by_box(query=doc.chunks[2], name="tokens"), [])

        # backwards
        self.assertListEqual(doc.intersect_by_box(query=doc.tokens[0], name="chunks"), [chunks[0]])
        self.assertListEqual(doc.intersect_by_box(query=doc.tokens[1], name="chunks"), [chunks[0]])
        self.assertListEqual(doc.intersect_by_box(query=doc.tokens[2], name="chunks"), [chunks[0]])
        self.assertListEqual(doc.intersect_by_box(query=doc.tokens[3], name="chunks"), [chunks[1]])
        self.assertListEqual(doc.intersect_by_box(query=doc.tokens[4], name="chunks"), [chunks[1]])
        self.assertListEqual(doc.intersect_by_box(query=doc.tokens[5], name="chunks"), [chunks[1]])

    def test_cross_referencing_with_no_spans(self):
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
        # chunks have no spans
        chunks = [
            Entity.from_json({"boxes": [[0, 0, 2.01, 2.01, 0]]}),
            Entity.from_json({"boxes": [[3.0, 3.0, 4.0, 4.0, 0]]}),
            Entity.from_json({"boxes": [[0, 0, 10.0, 10.0, 1]]}),
        ]
        doc.annotate_layer(name="tokens", entities=tokens)
        doc.annotate_layer(name="chunks", entities=chunks)

        # getattr() should still work when no spans; defers to boxes
        self.assertListEqual(doc.chunks[0].tokens, tokens[:3])
        self.assertListEqual(doc.chunks[1].tokens, tokens[3:])

        # last chunk is on a different page; intersects nothing
        self.assertListEqual(doc.chunks[2].tokens, [])

    def test_cross_referencing_with_missing_entity_fields(self):
        """What happens when annotate a Doc with entiites missing spans or boxes?
        How does the cross-referencing operation behave?"""
        # repeat the above test, but with missing spans and boxes
        doc = Document("This is a test document!")
        tokens = [
            Entity.from_json({"spans": [[0, 4]]}),
            Entity.from_json({"spans": [[5, 7]]}),
            Entity.from_json({"spans": [[8, 9]]}),
            Entity.from_json({"spans": [[10, 14]]}),
            Entity.from_json({"spans": [[15, 23]]}),
            Entity.from_json({"spans": [[23, 24]]}),
        ]
        chunks = [
            Entity.from_json({"boxes": [[0, 0, 2.01, 2.01, 0]]}),
            Entity.from_json({"boxes": [[3.0, 3.0, 4.0, 4.0, 0]]}),
            Entity.from_json({"boxes": [[0, 0, 10.0, 10.0, 1]]}),
        ]
        doc.annotate_layer(name="tokens", entities=tokens)
        doc.annotate_layer(name="chunks", entities=chunks)
        self.assertListEqual(doc.intersect_by_box(query=doc.chunks[0], name="tokens"), [])
        self.assertListEqual(doc.intersect_by_box(query=doc.chunks[1], name="tokens"), [])
        self.assertListEqual(doc.intersect_by_box(query=doc.chunks[2], name="tokens"), [])
        self.assertListEqual(doc.intersect_by_span(query=doc.chunks[0], name="tokens"), [])
        self.assertListEqual(doc.intersect_by_span(query=doc.chunks[1], name="tokens"), [])
        self.assertListEqual(doc.intersect_by_span(query=doc.chunks[2], name="tokens"), [])
        self.assertListEqual(doc.intersect_by_box(query=doc.tokens[0], name="chunks"), [])
        self.assertListEqual(doc.intersect_by_box(query=doc.tokens[1], name="chunks"), [])
        self.assertListEqual(doc.intersect_by_box(query=doc.tokens[2], name="chunks"), [])
        self.assertListEqual(doc.intersect_by_span(query=doc.tokens[0], name="chunks"), [])
        self.assertListEqual(doc.intersect_by_span(query=doc.tokens[1], name="chunks"), [])
        self.assertListEqual(doc.intersect_by_span(query=doc.tokens[2], name="chunks"), [])

    def test_query(self):
        doc = Document("This is a test document!")
        tokens = [
            Entity.from_json({"spans": [[0, 4]], "boxes": [[0, 0, 0.5, 0.5, 0]]}),
            Entity.from_json({"spans": [[5, 7]], "boxes": [[1, 1, 0.5, 0.5, 0]]}),
            Entity.from_json({"spans": [[8, 9]], "boxes": [[2, 2, 0.5, 0.5, 0]]}),
            Entity.from_json({"spans": [[10, 14]], "boxes": [[3, 3, 0.5, 0.5, 0]]}),
            Entity.from_json({"spans": [[15, 23]], "boxes": [[4, 4, 0.5, 0.5, 0]]}),
            Entity.from_json({"spans": [[23, 24]], "boxes": [[5, 5, 0.5, 0.5, 0]]}),
        ]
        chunks = [
            Entity.from_json({"spans": [[0, 9]], "boxes": [[0, 0, 2.01, 2.01, 0]]}),
            Entity.from_json({"spans": [[12, 23]], "boxes": [[3.0, 3.0, 4.0, 4.0, 0]]}),
            Entity.from_json({"spans": [[23, 24]], "boxes": [[0, 0, 10.0, 10.0, 1]]}),
        ]
        doc.annotate_layer(name="tokens", entities=tokens)
        doc.annotate_layer(name="chunks", entities=chunks)

        # test query by span
        self.assertListEqual(
            doc.intersect_by_span(query=doc.chunks[0], name="tokens"),
            doc.find(query=doc.chunks[0].spans[0], name="tokens"),
        )
        # test query by box
        self.assertListEqual(
            doc.intersect_by_box(query=doc.chunks[0], name="tokens"),
            doc.find(query=doc.chunks[0].boxes[0], name="tokens"),
        )
        # calling wrong method w input type should fail
        with self.assertRaises(TypeError):
            doc.intersect_by_box(query=doc.chunks[0].spans[0], name="tokens")
        with self.assertRaises(TypeError):
            doc.intersect_by_span(query=doc.chunks[0].boxes[0], name="tokens")
        with self.assertRaises(TypeError):
            doc.find(query=doc.chunks[0], name="tokens")
