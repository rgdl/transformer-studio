from dataclasses import dataclass
from typing import Iterator
from typing import Union

import streamlit as st
from PIL import Image
from PIL import ImageDraw
from typing_extensions import Self

from .consts import RGBA

keys = iter(x for x in range(int(1e6)))

BBox = Union[
    list[tuple[int, int]],
    list[int],
]


@dataclass
class Point:
    x: int
    y: int

    def __iter__(self) -> Iterator[int]:
        return iter([self.x, self.y])

    def __add__(self, other: Self) -> Self:
        return self.__class__(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Self) -> Self:
        return self.__class__(self.x - other.x, self.y - other.y)

    def __hash__(self) -> int:
        return hash(tuple(self))

    def distance(self, other: Self) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


def get_bbox(center: Point, width: int, height: int) -> BBox:
    to_corner = Point(width // 2, height // 2)
    top_left = tuple(center - to_corner)
    bottom_right = tuple([top_left[0] + width, top_left[1] + height])

    return [*top_left, *bottom_right]


@dataclass
class Ellipse:
    center: Point
    width: int
    height: int
    angle: float

    @classmethod
    def from_widgets(cls) -> Self:
        cols = st.columns(3)
        return cls(
            Point(
                cols[0].slider("Center-x", 0, 512, 300, key=next(keys)),
                cols[0].slider("Center-y", 0, 512, 200, key=next(keys)),
            ),
            cols[1].slider("Width", 0, 256, 10, key=next(keys)),
            cols[1].slider("Height", 0, 256, 10, key=next(keys)),
            cols[2].slider("Angle", 0, 260, key=next(keys)),
        )

    def hflip(self, center_point: int) -> Self:
        new_x = 2 * center_point - self.center.x

        return self.__class__(
            Point(new_x, self.center.y),
            self.width,
            self.height,
            360 - self.angle,
        )


def draw_ellipse(
    background: Image.Image,
    ellipse: Ellipse,
) -> Image.Image:
    """
    Draw the ellipse onto a transparent background, then rotate it, then paste
    onto background.
    """

    # Draw Ellipse on transparent background

    layer_dim = max(ellipse.width, ellipse.height)
    layer_center = Point(layer_dim // 2, layer_dim // 2)
    layer = Image.new("RGBA", (layer_dim, layer_dim), RGBA.CLEAR)

    draw = ImageDraw.Draw(layer)
    draw.ellipse(
        get_bbox(layer_center, ellipse.width, ellipse.height),
        RGBA.BLACK,
    )

    # Rotate it

    layer = layer.rotate(ellipse.angle)

    # Paste onto the background

    bbox = get_bbox(ellipse.center, *layer.size)
    background.paste(layer, bbox, layer)  # type: ignore

    return background


def build_shape(background: Image.Image) -> Image.Image:
    with st.sidebar:
        y_shift = -st.slider("Horizonal body shift", 0, 100)

    lower_wing = Ellipse(Point(300, 300 + y_shift), 68, 128, 33)
    upper_wing = Ellipse(Point(300, 231 + y_shift), 187, 80, 45)

    tail = Ellipse(Point(256, 328 + y_shift), 44, 156, 0)
    body = Ellipse(Point(256, 230 + y_shift), 53, 68, 0)
    head = Ellipse(Point(256, 178 + y_shift), 38, 38, 90)

    for ellipse in (
        tail,
        body,
        head,
        lower_wing,
        lower_wing.hflip(256),
        upper_wing,
        upper_wing.hflip(256),
    ):
        draw_ellipse(background, ellipse)

    return background
