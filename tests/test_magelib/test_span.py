"""


@kylel

"""


import unittest

from papermage.magelib import Span


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
