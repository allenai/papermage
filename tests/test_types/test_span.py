"""


@kylel

"""

import unittest
from papermage.types import Span

class TestSpan(unittest.TestCase):

    def test_to_from_json(self):
        span = Span(start=0, end=0)
        self.assertEqual(span.to_json(), [0, 0])

        span2 = Span.from_json(span.to_json())
        self.assertEqual(span2.start, 0)
        self.assertEqual(span2.end, 0)
        self.assertListEqual(span2.to_json(), [0, 0])
