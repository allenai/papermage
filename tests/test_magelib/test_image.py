"""

@kylel

"""

import os
import unittest

import numpy as np

from papermage.magelib import Image


class TestImage(unittest.TestCase):
    def test_to_from_array(self):
        imarray = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 12, 13], [14, 15, 16], [17, 18, 19]]])
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

    def test_eq(self):
        imarray = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 12, 13], [14, 15, 16], [17, 18, 19]]])
        image1 = Image.from_array(imarray=imarray)
        image2 = Image.from_array(imarray=imarray)
        self.assertEqual(image1, image2)

        imarray2 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 12, 13], [14, 15, 16], [17, 18, 20]]])
        image3 = Image.from_array(imarray=imarray2)
        self.assertNotEqual(image1, image3)

    def test_greyscale(self):
        imarray = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 12, 13], [14, 15, 16], [17, 18, 19]]])
        image = Image.from_array(imarray=imarray)
        image_grey = image.convert_to_greyscale()
        self.assertListEqual(
            image_grey.to_array().tolist(),
            [[[2, 2, 2], [5, 5, 5], [8, 8, 8]], [[12, 12, 12], [15, 15, 15], [18, 18, 18]]],
        )
        self.assertNotEqual(image, image_grey)
        self.assertEqual(image_grey.mode, "RGB")

    def test_save_open(self):
        # open
        imagefile = os.path.join(os.path.dirname(__file__), "../fixtures/white_page.png")
        image = Image.open(imagefile)
        # save
        imagefile = imagefile.replace("white_page.png", "temp_123_white_page.png")
        image.save(imagefile)
        # clean
        os.remove(imagefile)

    def test_create_rgb_all_white(self):
        image = Image.create_rgb_all_white(width=600, height=800)
        imagefile = os.path.join(os.path.dirname(__file__), "../fixtures/white_page.png")
        image2 = Image.open(imagefile)
        self.assertEqual(image, image2)
