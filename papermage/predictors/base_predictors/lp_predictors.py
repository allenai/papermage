"""

Base predictors of bounding box detection models from layoutparser

@shannons, @kylel

"""

from typing import Any, Dict, List, Optional, Union

import layoutparser as lp
from tqdm import tqdm

from papermage.magelib import (
    Box,
    Document,
    Entity,
    Image,
    ImagesFieldName,
    Metadata,
    PagesFieldName,
)

from .base_predictor import BasePredictor


class LPPredictor(BasePredictor):
    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [PagesFieldName, ImagesFieldName]

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        config_path: str,
        model_path: Optional[str] = None,
        label_map: Optional[Dict] = None,
        extra_config: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        """Initialize a pre-trained layout detection model from
        layoutparser. The parameters currently are the same as the
        default layoutparser Detectron2 models
        https://layout-parser.readthedocs.io/en/latest/api_doc/models.html ,
        and will be updated in the future.
        """

        model = lp.AutoLayoutModel(
            config_path=config_path,
            model_path=model_path,
            label_map=label_map,
            extra_config=extra_config,
            device=device,
        )

        return cls(model)

    def postprocess(self, model_outputs: lp.Layout, page_index: int, image: Image) -> List[Entity]:
        """Convert the model outputs into the papermage format

        Args:
            model_outputs (lp.Layout):
                The layout detection results from layoutparser for
                a page image
            page_index (int):
                The index of the current page, used for creating the
                `Box` object
            image (Image):
                The image of the current page, used for converting
                to relative coordinates for the box objects

        Returns:
            List[Entity]:
            The detected layout stored in a list of Entities.
        """

        # block.coordinates returns the left, top, bottom, right coordinates

        page_width, page_height = image.pilimage.size

        return [
            Entity(
                boxes=[
                    Box(
                        l=block.coordinates[0],
                        t=block.coordinates[1],
                        w=block.width,
                        h=block.height,
                        page=page_index,
                    ).to_relative(
                        page_width=page_width,
                        page_height=page_height,
                    )
                ],
                metadata=Metadata(type=block.type),
            )
            for block in model_outputs
        ]

    def _predict(self, doc: Document) -> List[Entity]:
        """Returns a list of Entities for the detected layouts for all pages

        Args:
            document (Document):
                The input document object

        Returns:
            List[Entity]:
                The returned Entities for the detected layouts for all pages
        """
        document_prediction = []

        images = doc.get_layer(name=ImagesFieldName)
        for image_index, image in enumerate(tqdm(images)):
            model_outputs = self.model.detect(image.pilimage)
            document_prediction.extend(self.postprocess(model_outputs, image_index, image))

        return document_prediction
