"""

Monkey patch the PIL.Image methods to add base64 conversion

@kylel, adapted some prior code from @shannons

"""

import os
import numpy as np

import base64
from io import BytesIO

import logging

from PIL import Image as pilimage_module
from PIL import ImageChops as pilimagechops_module
from PIL.Image import Image as PILImageClass



class Image:
    def __init__(self):
        self._pilimage = None
        # logging.warning('Unless testing or developing, we dont recommend creating Images '
        #                 'manually. Try passing in PDFs to Rasterizers to get Images'
        #                 'or loading already-created Images.')

    @property
    def pilimage(self) -> PILImageClass:
        if not self._pilimage:
            raise AttributeError(f"This Image missing a PILImage. Try setting it first.")
        return self._pilimage

    @pilimage.setter
    def pilimage(self, pilimage: PILImageClass):
        if self._pilimage:
            raise AttributeError(f"This Image already has a PILImage. Make a new Image.")
        self._pilimage = pilimage

    @property
    def mode(self) -> str:
        return self.pilimage.mode

    def convert_to_greyscale(self) -> 'Image':
        image = Image()
        image.pilimage = self.pilimage.convert('L').convert('RGB')
        return image

    def to_array(self) -> np.ndarray:
        return np.array(self.pilimage)

    @classmethod
    def from_array(cls, imarray: np.ndarray) -> 'Image':
        if len(imarray.shape) != 3:
            raise ValueError(f'Input `imarray` should have 3 dimensions: (rows, cols, channels)')
        if imarray.shape[-1] != 3:
            raise ValueError(f'Input `imarray` final dimension should be length 3 for RGB.')
        my_image = Image()
        my_image._pilimage = pilimage_module.fromarray(imarray.astype('uint8'))
        return my_image

    def to_json(self):
        pass

    @classmethod
    def from_json(cls, image_json):
        pass

    def to_base64(self) -> str:
        # Ref: https://stackoverflow.com/a/31826470
        buffered = BytesIO()
        self.pilimage.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        return img_str.decode("utf-8")

    @classmethod
    def from_base64(cls, img_str: str) -> 'Image':
        buffered = BytesIO(base64.b64decode(img_str))
        im = pilimage_module.open(buffered)
        image = Image()
        image.pilimage = im
        return image

    @classmethod
    def create_rgb_all_white(cls, width: int = 600, height: int = 800) -> 'Image':
        im = pilimage_module.new('RGB', (width, height), (255, 255, 255))
        image = Image()
        image.pilimage = im
        return image

    @classmethod
    def create_rgb_random(cls, width: int = 600, height: int = 800) -> 'Image':
        imarray = np.random.rand(height, width, 3) * 255
        image = Image.from_array(imarray=imarray)
        return image

    def save(self, filepath: str, is_overwrite: bool = False, format: str = None, **params):
        if os.path.exists(filepath) and not is_overwrite:
            raise FileExistsError(f'Already image at {filepath}. Try setting `is_overwrite`')
        self.pilimage.save(filepath, format, **params)

    @classmethod
    def open(cls, filepath: str, mode: str = 'r', formats: str = None) -> 'Image':
        im = pilimage_module.open(filepath, mode, formats)
        image = Image()
        image.pilimage = im
        return image

    def show(self, title: str = None):
        self.pilimage.show(title)

    def __eq__(self, other: 'Image') -> bool:
        diff = pilimagechops_module.difference(image1=self.pilimage, image2=other.pilimage)
        if diff.getbbox():
            return False
        else:
            return True