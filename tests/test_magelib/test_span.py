"""


@kylel

"""


import unittest

from papermage.magelib import Span
from papermage.magelib.span import MergeClusterSpans


class TestSpan(unittest.TestCase):
    def test_to_from_json(self):
        span = Span(start=0, end=0)
        self.assertEqual(span.to_json(), [0, 0])

        span2 = Span.from_json(span.to_json())
        self.assertEqual(span2.start, 0)
        self.assertEqual(span2.end, 0)
        self.assertListEqual(span2.to_json(), [0, 0])

    def test_sort_spans(self):
        self.assertFalse(Span(0, 0) < Span(0, 0))
        self.assertTrue(Span(0, 0) < Span(0, 1))
        self.assertTrue(Span(0, 0) < Span(1, 0))
        self.assertTrue(Span(0, 0) < Span(1, 1))
        a = Span(1, 2)
        b = Span(2, 3)
        c = Span(3, 4)
        self.assertListEqual(sorted([c, b, a]), [a, b, c])

    def test_overlap(self):
        self.assertTrue(Span(0, 0).is_overlap(other=Span(0, 0)))
        self.assertTrue(Span(0, 0).is_overlap(other=Span(0, 1)))
        self.assertTrue(Span(0, 1).is_overlap(other=Span(0, 0)))
        self.assertTrue(Span(0, 3).is_overlap(other=Span(1, 2)))
        self.assertTrue(Span(1, 2).is_overlap(other=Span(0, 3)))
        self.assertFalse(Span(0, 1).is_overlap(other=Span(1, 2)))
        self.assertFalse(Span(0, 1).is_overlap(other=Span(2, 2)))

    def test_equiv(self):
        self.assertTrue(Span(0, 0) == Span(0, 0))
        self.assertFalse(Span(0, 0) == Span(0, 1))
        self.assertFalse(Span(1, 0) == Span(0, 0))

    def test_create_enclosing_span(self):
        # nonempty
        with self.assertRaises(ValueError):
            Span.create_enclosing_span(spans=[])
        # singleton
        span = Span.create_enclosing_span(spans=[Span(1, 2)])
        self.assertEqual(span.start, 1)
        self.assertEqual(span.end, 2)
        # proper behavior
        span = Span.create_enclosing_span(spans=[Span(1, 2), Span(2, 3), Span(9, 10)])
        self.assertEqual(span.start, 1)
        self.assertEqual(span.end, 10)


class TestMergeClusterSpans(unittest.TestCase):
    def setUp(self):
        self.a = Span(0, 1)
        self.b = Span(1, 2)
        self.c = Span(3, 4)
        self.d = Span(4, 5)
        self.e = Span(9, 10)

    def test_empty(self):
        mcs = MergeClusterSpans(spans=[])
        self.assertListEqual(mcs.clusters, [[]])
        self.assertListEqual(mcs.merge(), [])

    def test_single(self):
        mcs = MergeClusterSpans(spans=[self.a])
        self.assertListEqual(mcs.clusters, [[self.a]])
        self.assertListEqual(mcs.merge(), [self.a])

    def test_disjoint(self):
        mcs = MergeClusterSpans(spans=[self.a, self.c, self.e])
        self.assertListEqual(mcs.clusters, [[self.a], [self.c], [self.e]])
        self.assertListEqual(mcs.merge(), [self.a, self.c, self.e])
        # reversed order
        mcs = MergeClusterSpans(spans=[self.e, self.c, self.a])
        self.assertListEqual(mcs.clusters, [[self.e], [self.c], [self.a]])
        self.assertListEqual(mcs.merge(), [self.e, self.c, self.a])

    def test_cluster(self):
        mcs = MergeClusterSpans(spans=[self.a, self.b, self.c, self.d, self.e], index_distance=0)
        self.assertListEqual(mcs.clusters, [[self.a, self.b], [self.c, self.d], [self.e]])
        mcs = MergeClusterSpans(spans=[self.a, self.b, self.c, self.d, self.e], index_distance=1)
        self.assertListEqual(mcs.clusters, [[self.a, self.b, self.c, self.d], [self.e]])
        mcs = MergeClusterSpans(spans=[self.a, self.b, self.c, self.d, self.e], index_distance=4)
        self.assertListEqual(mcs.clusters, [[self.a, self.b, self.c, self.d, self.e]])

    def test_merge(self):
        mcs = MergeClusterSpans(spans=[self.a, self.b, self.c, self.d, self.e], index_distance=0)
        self.assertListEqual(mcs.merge(), [Span(0, 2), Span(3, 5), Span(9, 10)])
        mcs = MergeClusterSpans(spans=[self.a, self.b, self.c, self.d, self.e], index_distance=1)
        self.assertListEqual(mcs.merge(), [Span(0, 5), Span(9, 10)])
        mcs = MergeClusterSpans(spans=[self.a, self.b, self.c, self.d, self.e], index_distance=4)
        self.assertListEqual(mcs.merge(), [Span(0, 10)])
