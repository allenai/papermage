"""

Tools for visualizing annotations on a Document's pages.

@kylel

"""

from typing import Dict, List, Optional, Union

import layoutparser.elements as lpe
import layoutparser.visualization as lpv

from papermage.magelib import Entity, Image


def plot_entities_on_page(
    page_image: Image,
    entities: List[Entity],
    box_width: Optional[Union[List[int], int]] = None,
    box_alpha: Optional[Union[List[float], float]] = None,
    box_color: Optional[Union[List[str], str]] = None,
    color_map: Optional[Dict] = None,
    show_element_id: bool = False,
    show_element_type: bool = False,
    id_font_size: Optional[int] = None,
    id_font_path: Optional[str] = None,
    id_text_color: Optional[str] = None,
    id_text_background_color: Optional[str] = None,
    id_text_background_alpha: Optional[float] = 1,
) -> Image:
    page_w, page_h = page_image.pilimage.size
    lpe_boxes = []
    for entity in entities:
        for box in entity.boxes:
            lpe_box = lpe.Rectangle(
                x_1=box.l * page_w, y_1=box.t * page_h, x_2=(box.l + box.w) * page_w, y_2=(box.t + box.h) * page_h
            )
            lpe_boxes.append(lpe_box)
    viz = lpv.draw_box(
        canvas=page_image.pilimage,
        layout=lpe_boxes,
        box_width=box_width,
        box_alpha=box_alpha,
        box_color=box_color,
        color_map=color_map,
        show_element_id=show_element_id,
        show_element_type=show_element_type,
        id_font_size=id_font_size,
        id_font_path=id_font_path,
        id_text_color=id_text_color,
        id_text_background_color=id_text_background_color,
        id_text_background_alpha=id_text_background_alpha,
    )
    # return viz as new image
    annotated_image = Image()
    annotated_image.pilimage = viz
    return annotated_image
