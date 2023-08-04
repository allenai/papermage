"""

@kylel

"""

import unittest

from papermage.magelib import Box


class TestBox(unittest.TestCase):
    def test_create(self):
        with self.assertRaises(AssertionError):
            Box(l=0.2, t=0.09, w=-0.095, h=0.017, page=3)
        with self.assertRaises(AssertionError):
            Box(l=0.2, t=0.09, w=0.095, h=-0.017, page=3)

    def test_to_from_json(self):
        box = Box(l=0.2, t=0.09, w=0.095, h=0.017, page=3)
        self.assertEqual(box.to_json(), [0.2, 0.09, 0.095, 0.017, 3])

        box2 = Box.from_json(box.to_json())
        self.assertEqual(box2.l, 0.2)
        self.assertEqual(box2.t, 0.09)
        self.assertEqual(box2.w, 0.095)
        self.assertEqual(box2.h, 0.017)
        self.assertEqual(box2.page, 3)
        self.assertListEqual(box2.to_json(), [0.2, 0.09, 0.095, 0.017, 3])

    def test_numeric_type_casting(self):
        self.assertListEqual(Box(1, 2, 3, 4, 5.0).to_json(), [1.0, 2.0, 3.0, 4.0, 5])

    def test_overlap(self):
        # with itself
        b1 = Box(0.2, 0.09, 0.095, 0.017, 3)
        self.assertTrue(b1.is_overlap(other=b1))
        # page matters
        b2 = Box(0.2, 0.09, 0.095, 0.017, 2)
        self.assertFalse(b1.is_overlap(other=b2))
        # x-axis shift to right too much
        b3 = Box(0.3, 0.09, 0.095, 0.017, 3)
        self.assertFalse(b1.is_overlap(other=b3))
        # x-axis shift just a little
        b4 = Box(0.295, 0.09, 0.095, 0.017, 3)
        self.assertTrue(b1.is_overlap(other=b4))
        # y-axis shift to down too much
        b5 = Box(0.2, 0.108, 0.095, 0.017, 3)
        self.assertFalse(b1.is_overlap(other=b5))
        # y-axis shift just a little
        b6 = Box(0.2, 0.106999, 0.095, 0.017, 3)
        self.assertTrue(b1.is_overlap(other=b6))

    def test_xy_coordinates(self):
        box = Box.from_xy_coordinates(x1=0.2, y1=0.09, x2=0.295, y2=0.107, page=3)
        # allow for some floating point errors
        self.assertAlmostEqual(box.l, 0.2)
        self.assertAlmostEqual(box.t, 0.09)
        self.assertAlmostEqual(box.w, 0.095)
        self.assertAlmostEqual(box.h, 0.017)
        self.assertEqual(box.page, 3)
        # it's important that the return value of this does *NOT* look like the JSON serialization
        # hence why we make it a tuple & drop the page.
        # otherwise, it can be confusing if come back to this data at a later date.
        self.assertEqual(box.xy_coordinates, (0.2, 0.09, 0.295, 0.107))

        box = Box.from_xy_coordinates(x1=0.2, y1=0.09, x2=10.295, y2=30.107, page=3, page_width=10, page_height=30)
        # allow for some floating point errors
        self.assertAlmostEqual(box.l, 0.2)
        self.assertAlmostEqual(box.t, 0.09)
        self.assertAlmostEqual(box.w, 9.8)
        self.assertAlmostEqual(box.h, 29.91)
        self.assertEqual(box.page, 3)
        # it's important that the return value of this does *NOT* look like the JSON serialization
        # hence why we make it a tuple & drop the page.
        # otherwise, it can be confusing if come back to this data at a later date.
        self.assertEqual(box.xy_coordinates, (0.2, 0.09, 10, 30))

    def test_relative_absolute(self):
        box = Box(l=0.2, t=0.09, w=0.095, h=0.017, page=3)
        box_abs = box.to_absolute(page_width=100, page_height=300)
        self.assertAlmostEqual(box_abs.l, 20.0)
        self.assertAlmostEqual(box_abs.t, 27.0)
        self.assertAlmostEqual(box_abs.w, 9.5)
        self.assertAlmostEqual(box_abs.h, 5.1)
        self.assertEqual(box_abs.page, 3)
        box_rel = box_abs.to_relative(page_width=100, page_height=300)
        self.assertAlmostEqual(box_rel.l, 0.2)
        self.assertAlmostEqual(box_rel.t, 0.09)
        self.assertAlmostEqual(box_rel.w, 0.095)
        self.assertAlmostEqual(box_rel.h, 0.017)
        self.assertEqual(box_rel.page, 3)

    def test_center(self):
        box = Box(l=0.2, t=0.09, w=0.095, h=0.017, page=3)
        x, y = box.center
        self.assertAlmostEqual(x, 0.2475)
        self.assertAlmostEqual(y, 0.0985)

    def test_create_enclosing_box(self):
        # nonempty
        with self.assertRaises(ValueError):
            Box.create_enclosing_box(boxes=[])
        # singleton
        box = Box.create_enclosing_box(boxes=[Box(l=0.2, t=0.09, w=0.095, h=0.017, page=3)])
        self.assertAlmostEqual(box.l, 0.2)
        self.assertAlmostEqual(box.t, 0.09)
        self.assertAlmostEqual(box.w, 0.095)
        self.assertAlmostEqual(box.h, 0.017)
        self.assertEqual(box.page, 3)
        # proper behavior
        box2 = Box(0.15, 0.05, 0.01, 0.01, 3)
        box3 = Box(0.05, 0.12, 0.01, 0.01, 3)
        box_new = Box.create_enclosing_box(boxes=[box, box2, box3])
        self.assertAlmostEqual(box_new.l, 0.05)
        self.assertAlmostEqual(box_new.t, 0.05)
        self.assertAlmostEqual(box_new.w, 0.245)
        self.assertAlmostEqual(box_new.h, 0.08)
        self.assertEqual(box_new.page, 3)
