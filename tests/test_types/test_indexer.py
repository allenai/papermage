"""

@kylel

"""

import unittest

from papermage.types import Entity, Span, EntitySpanIndexer, Document


class TestEntitySpanIndexer(unittest.TestCase):
    def test_index_empty(self):
        indexer = EntitySpanIndexer(entities=[])

    def test_overlap_within_single_entity_fails_checks(self):
        entities = [Entity(spans=[Span(0, 5), Span(4, 7)])]

        with self.assertRaises(ValueError):
            EntitySpanIndexer(entities=entities)

    def test_overlap_between_entities_fails_checks(self):
        entities = [
            Entity(spans=[Span(0, 5), Span(5, 8)]),
            Entity(spans=[Span(6, 10)])
        ]

        with self.assertRaises(ValueError):
            EntitySpanIndexer(entities)

    def test_finds_matching_entities(self):
        entities_to_index = [
            Entity(spans=[Span(0, 5), Span(5, 8)]),
            Entity(spans=[Span(9, 10)]),
            Entity(spans=[Span(100, 105)])
        ]

        index = EntitySpanIndexer(entities_to_index)

        # should intersect 1 and 2 but not 3
        probe = Entity(spans=[Span(1, 7), Span(9, 20)])
        matches = index.find(probe)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches, [entities_to_index[0], entities_to_index[1]])

    def test_finds_matching_entities_in_original_order(self):
        entities_to_index = [
            Entity(spans=[Span(100, 105)]),
            Entity(spans=[Span(9, 10)]),
            Entity(spans=[Span(0, 5), Span(5, 8)])
        ]
        
        for i, entity in enumerate(entities_to_index):
            entity.doc = Document(symbols="test")
            entity.id = i

        index = EntitySpanIndexer(entities_to_index)

        # should intersect 2 and 3 but not 1
        probe = Entity(spans=[Span(9, 20), Span(1, 7)])
        matches = index.find(probe)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches, [entities_to_index[1], entities_to_index[2]])
