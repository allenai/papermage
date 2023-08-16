import unittest

from papermage import Document, Entity, Layer

class TestLayer(unittest.TestCase):
    def test_layer_slice(self):
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
        doc.annotate_entity(field_name="tokens", entities=tokens)
        doc.annotate_entity(field_name="chunks", entities=chunks)

        assert isinstance(doc.tokens, Layer)
        assert isinstance(doc.chunks[1:3], Layer)

        self.assertSequenceEqual(doc.chunks[1:3], chunks[1:3])
        self.assertSequenceEqual(doc.chunks[1:3].text, ['st document', '!'])
        assert isinstance(doc.chunks[:3].tokens, Layer)