from typing import Iterable, Protocol, List

from papermage.types.image import Image

try:
    import pdf2image
except ImportError:
    pass


class Rasterizer(Protocol):
    def rasterize(self, input_pdf_path: str, dpi: int, **kwargs) -> Iterable[Image]:
        """Given an input PDF return a List[Image]

        Args:
            input_pdf_path (str): Path to the input PDF to process
            dpi (int): Used for specify the resolution (or `DPI, dots per inch
                       <https://en.wikipedia.org/wiki/Dots_per_inch>`_) when loading images of
                       the pdf. Higher DPI values mean clearer images (also larger file sizes).

        Returns:
            Iterable[Image]
        """
        raise NotImplementedError

class PDF2ImageRasterizer(Rasterizer):
    def rasterize(self, input_pdf_path: str, dpi: int, **kwargs) -> Iterable[Image]:
        """Rasterize the pdf and convert the PIL images to papermage Image objects"""
        pil_images = pdf2image.convert_from_path(pdf_path=input_pdf_path, dpi=dpi)
        images: List[Image] = []
        for pil_image in pil_images:
            image = Image()
            image.pilimage = pil_image
            images.append(image)
        return images
