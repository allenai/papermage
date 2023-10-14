"""

Converts pages of a PDF into Images that can be attached to a Document. 

@shannons, @kylel

"""

from typing import Iterable, List, Protocol

from papermage.magelib import Document, Image, PagesFieldName

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

    @classmethod
    def attach_images(cls, images: List[Image], doc: Document) -> None:
        """Assumes doc already has `pages` annotated."""
        if PagesFieldName not in doc.layers:
            raise ValueError(f"Failed to attach. Document is missing `pages`.")
        pages = doc.get_layer(name=PagesFieldName)
        if len(images) != len(pages):
            raise ValueError(f"Failed to attach. {len(images)} `images` != {len(pages)} pages in `doc`.")
        for page, image in zip(pages, images):
            if page.images:
                raise AttributeError(f"Failed to attach. `images` already exists on this page {page}")
            page.images = [image]


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
