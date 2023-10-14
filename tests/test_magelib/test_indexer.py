"""

@kylel

"""

import unittest

from papermage.magelib import (
    Box,
    Document,
    Entity,
    EntityBoxIndexer,
    EntitySpanIndexer,
    Span,
)


class TestEntitySpanIndexer(unittest.TestCase):
    def test_index_empty(self):
        indexer = EntitySpanIndexer(entities=[])

    def test_overlap_within_single_entity_fails_checks(self):
        entities = [Entity(spans=[Span(0, 5), Span(4, 7)])]

        with self.assertRaises(ValueError):
            EntitySpanIndexer(entities=entities)

    def test_overlap_between_entities_fails_checks(self):
        entities = [Entity(spans=[Span(0, 5), Span(5, 8)]), Entity(spans=[Span(6, 10)])]

        with self.assertRaises(ValueError):
            EntitySpanIndexer(entities)

    def test_finds_matching_entities(self):
        entities_to_index = [
            Entity(spans=[Span(0, 5), Span(5, 8)]),
            Entity(spans=[Span(9, 10)]),
            Entity(spans=[Span(100, 105)]),
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
            Entity(spans=[Span(0, 5), Span(5, 8)]),
        ]

        for i, entity in enumerate(entities_to_index):
            entity.layer = Document(symbols="test")
            entity.id = i

        index = EntitySpanIndexer(entities_to_index)

        # should intersect 2 and 3 but not 1
        probe = Entity(spans=[Span(9, 20), Span(1, 7)])
        matches = index.find(probe)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches, [entities_to_index[1], entities_to_index[2]])


class TestEntityBoxIndexer(unittest.TestCase):
    def test_overlap_within_single_entity_fails_checks(self):
        entities = [Entity(boxes=[Box(0, 0, 5, 5, page=0), Box(4, 4, 7, 7, page=0)])]

        with self.assertRaises(ValueError):
            EntityBoxIndexer(entities=entities, allow_overlap=False)
        EntityBoxIndexer(entities=entities, allow_overlap=True)

    def test_overlap_between_entities_fails_checks(self):
        entities = [
            Entity(boxes=[Box(0, 0, 5, 5, page=0), Box(5.01, 5.01, 8, 8, page=0)]),
            Entity(boxes=[Box(6, 6, 10, 10, page=0)]),
        ]

        with self.assertRaises(ValueError):
            EntityBoxIndexer(entities=entities, allow_overlap=False)
        EntityBoxIndexer(entities=entities, allow_overlap=True)

    def test_finds_matching_entities_in_doc_order(self):
        entities_to_index = [
            Entity(boxes=[Box(0, 0, 1, 1, page=0), Box(2, 2, 1, 1, page=0)]),
            Entity(boxes=[Box(4, 4, 1, 1, page=0)]),
            Entity(boxes=[Box(100, 100, 1, 1, page=0)]),
        ]
        index = EntityBoxIndexer(entities_to_index)

        # should intersect 1 and 2 but not 3
        probe = Entity(boxes=[Box(1, 1, 5, 5, page=0), Box(9, 9, 5, 5, page=0)])
        matches = index.find(probe)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches, [entities_to_index[0], entities_to_index[1]])

    def test_finds_matching_entities_accounts_for_pages(self):
        entities_to_index = [
            Entity(boxes=[Box(0.0, 0.0, 0.1, 0.1, page=0), Box(0.2, 0.2, 0.1, 0.1, page=1)]),
            Entity(boxes=[Box(0.4, 0.4, 0.1, 0.1, page=1)]),
            Entity(boxes=[Box(10.0, 10.0, 0.1, 0.1, page=0)]),
        ]

        index = EntityBoxIndexer(entities_to_index)

        # shouldnt intersect any given page 0
        probe = Entity(boxes=[Box(0.1, 0.1, 0.5, 0.5, page=0), Box(0.9, 0.9, 0.5, 0.5, page=0)])
        matches = index.find(probe)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches, [entities_to_index[0]])

        # shoudl intersect after switching to page 1 (and the page 2 box doesnt intersect)
        probe = Entity(boxes=[Box(0.1, 0.1, 0.5, 0.5, page=1), Box(10.0, 10.0, 0.1, 0.1, page=2)])
        matches = index.find(probe)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches, [entities_to_index[0], entities_to_index[1]])
