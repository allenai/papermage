"""

@kylel

"""

import unittest
from papermage.types import Image

import numpy as np


class TestImage(unittest.TestCase):
    def test_to_from_array(self):
        imarray = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            [[11, 12, 13], [14, 15, 16], [17, 18, 19]]])
        image = Image.from_array(imarray=imarray)
        imarray2 = image.to_array()
        self.assertListEqual(imarray.tolist(), imarray2.tolist())

    def test_to_from_array_dimensions(self):
        # wrong number of color channel
        with self.assertRaises(ValueError):
            imarray = np.array([[[1, 2], [4, 5], [7, 8]], [[11, 12], [14, 15], [17, 18]]])
            image = Image.from_array(imarray=imarray)
        # missing some dimension
        with self.assertRaises(ValueError):
            imarray = np.array([[1, 2, 4, 5, 7, 8], [11, 12, 14, 15, 17, 18]])
            image = Image.from_array(imarray=imarray)

    def test_create_rgb_all_white(self):
        image = Image.create_rgb_all_white()
