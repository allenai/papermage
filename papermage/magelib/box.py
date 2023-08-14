"""

Rectangular region on a document.

@kylel

"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np

from .span import Span


class _BoxSpan(Span):
    def __init__(self, start: float, end: float):
        self.start = start  # type: ignore
        self.end = end  # type: ignore


class Box:

    __slots__ = ["l", "t", "w", "h", "page"]

    def __init__(self, l: float, t: float, w: float, h: float, page: int):
        assert w >= 0.0, "Box width cant be negative"
        assert h >= 0.0, "Box height cant be negative"
        self.l = float(l)
        self.t = float(t)
        self.w = float(w)
        self.h = float(h)
        self.page = int(page)

    def to_json(self) -> List[Union[float, int]]:
        """Returns whatever representation is JSON compatible"""
        return [self.l, self.t, self.w, self.h, self.page]

    @classmethod
    def from_json(cls, box_json: List[Union[float, int]]) -> "Box":
        """Recreates the object from the JSON serialization"""
        l, t, w, h, page = box_json
        return Box(l=l, t=t, w=w, h=h, page=int(page))

    def __repr__(self):
        return f"Box{self.to_json()}"

    @classmethod
    def from_xy_coordinates(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        page: int,
        page_width: Optional[float] = None,
        page_height: Optional[float] = None,
    ):
        """Create a Box from the top-left and bottom-right coordinates.

        If the page width and height are provided and coordinates lie outside the page, they are clipped to the
        page width and height.
        """

        assert x2 >= x1, "Requires x2 >= x1"
        assert y2 >= y1, "Requires y2 >= y1"
        _x1, _x2, _y1, _y2 = x1, x2, y1, y2
        if page_width is not None:
            _x1, _x2 = np.clip([x1, x2], 0, page_width)
        if page_height is not None:
            _y1, _y2 = np.clip([y1, y2], 0, page_height)

        if (_x1, _y1, _x2, _y2) != (x1, y1, x2, y2):
            logging.warning(
                f"The coordinates ({x1}, {y1}, {x2}, {y2}) are not valid and converted to ({_x1}, {_y1}, {_x2}, {_y2})."
            )

        return Box(l=_x1, t=_y1, w=_x2 - _x1, h=_y2 - _y1, page=page)

    @property
    def xy_coordinates(self) -> Tuple[float, float, float, float]:
        return self.l, self.t, self.l + self.w, self.t + self.h

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box):
            return False
        return (
            self.l == other.l
            and self.t == other.t
            and self.w == other.w
            and self.h == other.h
            and self.page == other.page
        )

    def to_relative(self, page_width: float, page_height: float) -> "Box":
        """Get the relative coordinates of self based on page_width, page_height."""
        return self.__class__(
            l=float(self.l) / page_width,
            t=float(self.t) / page_height,
            w=float(self.w) / page_width,
            h=float(self.h) / page_height,
            page=self.page,
        )

    def to_absolute(self, page_width: int, page_height: int) -> "Box":
        """Get the absolute coordinates of self based on page_width, page_height."""
        return self.__class__(
            l=self.l * page_width,
            t=self.t * page_height,
            w=self.w * page_width,
            h=self.h * page_height,
            page=self.page,
        )

    @property
    def center(self) -> Tuple[float, float]:
        return self.l + self.w / 2, self.t + self.h / 2

    def is_overlap(self, other: "Box") -> bool:
        """Whether self overlaps with the other Box object"""

        if self.page != other.page:
            return False

        self_x1, self_y1, self_x2, self_y2 = self.xy_coordinates
        other_x1, other_y1, other_x2, other_y2 = other.xy_coordinates

        # check x-axis
        span_x_self = _BoxSpan(start=self_x1, end=self_x2)
        span_x_other = _BoxSpan(start=other_x1, end=other_x2)
        if not span_x_self.is_overlap(span_x_other):
            return False

        # check y-axis
        span_y_self = _BoxSpan(start=self_y1, end=self_y2)
        span_y_other = _BoxSpan(start=other_y1, end=other_y2)
        if not span_y_self.is_overlap(span_y_other):
            return False

        return True

    @classmethod
    def create_enclosing_box(cls, boxes: List["Box"]) -> "Box":
        """Create the narrowest Box that completely encloses all the input Boxes."""
        if not boxes:
            raise ValueError(f"`boxes` should be non-empty.")
        unique_pages = {box.page for box in boxes}
        if len(unique_pages) != 1:
            raise ValueError(f"Boxes not all on same page. Pages={unique_pages}")
        x1 = min([box.l for box in boxes])
        y1 = min([box.t for box in boxes])
        x2 = max([box.l + box.w for box in boxes])
        y2 = max([box.t + box.h for box in boxes])
        return Box(l=x1, t=y1, w=x2 - x1, h=y2 - y1, page=boxes[0].page)
