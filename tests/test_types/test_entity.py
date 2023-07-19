"""

@kylel

"""

import unittest
from papermage.types import Entity, Span, Box, Metadata

class TestEntity(unittest.TestCase):
    def test_create_empty(self):
        with self.assertRaises(ValueError):
            e = Entity()
        # both nones not OK
        with self.assertRaises(ValueError):
            e = Entity(spans=None)
        with self.assertRaises(ValueError):
            e = Entity(boxes=None)
        # empty lists also not ok
        with self.assertRaises(ValueError):
            e = Entity(spans=[])
        with self.assertRaises(ValueError):
            e = Entity(boxes=[])
        # metadata-only also not ok
        m = Metadata(x=123, y=456)
        with self.assertRaises(ValueError):
            e = Entity(metadata=m)

    def test_create(self):
        # just spans
        spans = [Span(0, 1), Span(2, 3)]
        entity = Entity(spans=spans)
        self.assertListEqual(entity.spans, spans)
        self.assertListEqual(entity.boxes, [])

        # just boxes
        boxes = [Box(0, 0, 0, 0, 0), Box(1, 1, 1, 1, 1)]
        entity = Entity(boxes=boxes)
        self.assertListEqual(entity.boxes, boxes)
        self.assertListEqual(entity.spans, [])

        # both spans and boxes
        entity = Entity(spans=spans, boxes=boxes)
        self.assertListEqual(entity.spans, spans)
        self.assertListEqual(entity.boxes, boxes)

        # auto-creates metadata
        self.assertDictEqual(entity.metadata.to_json(), {})

        # manual-created metadata
        metadata = Metadata(x=123, y=456)
        entity = Entity(spans=spans, boxes=boxes, metadata=metadata)
        self.assertIs(entity.metadata, metadata)

    def test_to_from_json(self):
        spans = [Span(0, 1), Span(2, 3)]
        boxes = [Box(0, 0, 0, 0, 0), Box(1, 1, 1, 1, 1)]
        metadata = Metadata(x=123, y=456)

        #  spans only
        entity = Entity(spans=spans)
        self.assertEqual(entity.to_json(), {'spans': [[0, 1], [2, 3]]})
        entity2 = Entity.from_json(entity.to_json())
        self.assertDictEqual(entity2.to_json(), entity.to_json())

        #  boxes only
        entity = Entity(boxes=boxes)
        self.assertEqual(entity.to_json(),
                         {'boxes': [[0.0, 0.0, 0.0, 0.0, 0], [1.0, 1.0, 1.0, 1.0, 1]]})
        entity2 = Entity.from_json(entity.to_json())
        self.assertDictEqual(entity2.to_json(), entity.to_json())

        # both
        entity = Entity(spans=spans, boxes=boxes)
        self.assertEqual(entity.to_json(),
                         {'spans': [[0, 1], [2, 3]],
                          'boxes': [[0.0, 0.0, 0.0, 0.0, 0], [1.0, 1.0, 1.0, 1.0, 1]]})
        entity2 = Entity.from_json(entity.to_json())
        self.assertDictEqual(entity2.to_json(), entity.to_json())

        # incl metadata
        entity = Entity(spans=spans, boxes=boxes, metadata=metadata)
        self.assertEqual(entity.to_json(),
                         {'spans': [[0, 1], [2, 3]],
                          'boxes': [[0.0, 0.0, 0.0, 0.0, 0], [1.0, 1.0, 1.0, 1.0, 1]],
                          'metadata': {'x': 123, 'y': 456}})
        entity2 = Entity.from_json(entity.to_json())
        self.assertDictEqual(entity2.to_json(), entity.to_json())
