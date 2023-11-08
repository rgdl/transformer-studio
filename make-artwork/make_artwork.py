"""
Need to create:
    * an icon (24-bit JPEG or PNG, 512x512)
    * a header image (24-bit JPEG or PNG, 4096x2304)
"""
from typing import Callable
from typing import Union
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import streamlit as st
from PIL import Image

from consts import RGB
from shapes import Point
from shapes import build_shape

Number: TypeAlias = Union[np.float_, np.int_]
Array: TypeAlias = npt.NDArray[Number]

ImageTransform: TypeAlias = Callable[[Array, Array], Array]


class Ray:
    """
    For each pixel on the outside edge of the image, draw a line to the image
    center and select all pixels on that line up until it crosses the image
    outline.

    To avoid redundant redraws, don't add pixels to a ray if they already exist
    on another ray. As long as the shading function is spatially smooth, this
    won't cause problems.
    """

    pixels: set[Point]
    angle: float


def apply_to_neighbourhood(x: Array, func: ImageTransform) -> Array:
    return func(x, get_neighbourhood(x))


def get_neighbourhood(x: Array) -> Array:
    # TODO: func to build neighbourhood for reuse, include diagonals as well

    u_neighbour = np.pad(x, ((1, 0), (0, 0)))[:-1]
    d_neighbour = np.pad(x, ((0, 1), (0, 0)))[1:]
    l_neighbour = np.pad(x, ((0, 0), (1, 0)))[:, :-1]
    r_neighbour = np.pad(x, ((0, 0), (0, 1)))[:, 1:]

    return np.stack(
        [u_neighbour, d_neighbour, l_neighbour, r_neighbour],
        axis=2,
    )


def outline(image: Image.Image) -> Image.Image:
    x = np.array(image).astype(np.float64).mean(axis=2)

    def _find_edges(x: Array, n: Array) -> Array:
        return (x.reshape(*x.shape, 1) - n).max(axis=2)  # type: ignore

    def _smooth(x: Array, n: Array) -> Array:
        n = np.concatenate([x.reshape(*x.shape, 1), n], axis=2)
        smoothed = n.mean(axis=2)

        # Bring max back up
        smoothed *= 255 / smoothed.max()

        return smoothed  # type: ignore

    edges = apply_to_neighbourhood(x, _find_edges)

    # TODO: learn how to do acutal kernel transforms efficiently

    # TODO: trace rays between the outline and the edge of the image

    huge_blur = edges.copy()
    big_blur = edges.copy()
    small_blur = edges.copy()

    cols = st.columns(3)
    n_small_blur = cols[0].slider("Small Blur", 0, 10, 3)
    n_big_blur = cols[1].slider("Big Blur", 10, 100, 23)
    n_huge_blur = cols[2].slider("Huge Blur", 100, 1000, 265)

    total_blur = n_small_blur + n_big_blur + n_huge_blur
    pbar = st.progress(0.0)
    blurs_completed = 0

    for _ in range(n_small_blur):
        small_blur = apply_to_neighbourhood(small_blur, _smooth)
        blurs_completed += 1
        pbar.progress(blurs_completed / total_blur)

    # TODO: uncomment!
    #for _ in range(n_big_blur):
    #    big_blur = apply_to_neighbourhood(big_blur, _smooth)
    #    blurs_completed += 1
    #    pbar.progress(blurs_completed / total_blur)

    # TODO: uncomment!
    #for _ in range(n_huge_blur):
    #    huge_blur = apply_to_neighbourhood(huge_blur, _smooth)
    #    blurs_completed += 1
    #    pbar.progress(blurs_completed / total_blur)

    x = np.stack(
        [
            huge_blur,
            big_blur,
            small_blur,
        ],
        axis=2,
    ).max(axis=2)

    # Ray tracing

    traced_pixels: set[Point] = set()

    max_y = x.shape[0]
    max_x = x.shape[1]
    all_y = range(max_y)
    all_x = range(max_x)

    outer_edge_pixels = {
        *(Point(0, y) for y in all_y),
        *(Point(max_x, y) for y in all_y),
        *(Point(x, 0) for x in all_x),
        *(Point(x, max_y) for x in all_x),
    }

    img_center = Point(max_x // 2, max_y // 2)

    for p in outer_edge_pixels:
        pass

    # TODO: test tracing - colour every pixel on every ray, ensure none missing
    """
    In tracing the ray from the outside to the center, there will be 3 pixels
    to choose from at each step. The 3 neighbours that are closer to the center
    than the current pixel. I can use this formula to work out which of the 3
    is closest to the line:

    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
    The 2 points that define the line are the outside point and the center
    point. Then the next pixel is the neighbour closest to the line (out of the
    3 neighbours between the current pixel and the center).

    """

    colours = np.stack([x for _ in range(3)], axis=2)
    return Image.fromarray(colours.astype(np.uint8))


def main() -> None:
    st.title("Make Artwork for Website")

    icon = Image.new("RGB", (512, 512), RGB.GREY)

    build_shape(icon)
    icon = outline(icon)

    st.image(icon)


main()
