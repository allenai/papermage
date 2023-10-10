"""

@kylel

"""

import unittest

from papermage.magelib import Box, Entity, Layer, Span


class TestLayer(unittest.TestCase):
    def setUp(self):
        e1 = Entity(spans=[Span(0, 1), Span(2, 3)])
        e2 = Entity(boxes=[Box(0, 0, 0, 0, 0), Box(1, 1, 1, 1, 1)])
        self.layer = Layer(entities=[e1, e2])

    def test_create(self):
        empty = Layer(entities=[])

    def test_index(self):
        self.assertEqual(self.layer[0], self.layer.entities[0])
        self.assertEqual(self.layer[1], self.layer.entities[1])

    def test_len(self):
        self.assertEqual(len(self.layer), 2)

    def test_iter(self):
        self.assertEqual(list(self.layer), self.layer.entities)

    def test_contains(self):
        self.assertIn(self.layer[0], self.layer)
        self.assertIn(self.layer[1], self.layer)

    def test_to_from_json(self):
        self.assertEqual(self.layer.to_json(), [e.to_json() for e in self.layer.entities])

        layer2 = Layer.from_json(self.layer.to_json())
        self.assertDictEqual(layer2[0].to_json(), self.layer[0].to_json())
        self.assertDictEqual(layer2[1].to_json(), self.layer[1].to_json())
