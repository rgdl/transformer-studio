"""
Need to create:
    * an icon (24-bit JPEG or PNG, 512x512)
    * a header image (24-bit JPEG or PNG, 4096x2304)
"""
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import streamlit as st
from PIL import Image

from pages.arctangent import expand_to_range
from utils import timer
from utils.consts import RGB
from utils.shapes import Point
from utils.shapes import build_shape

# TODO: different agg function to make border less jagged (softer than max)

Number: TypeAlias = np.float_ | np.int_
Array: TypeAlias = npt.NDArray[Number]

ImageTransform: TypeAlias = Callable[[Array, Array], Array]


@dataclass
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
    outer_pixel: Point
    inner_pixel: Point
    angle: float

    def __hash__(self) -> int:
        return hash(self.angle)

    def colour_in(
        self,
        A: float,
        image: Array,
        img_center: Point,
        domain_transform: Callable[[float], float] | None = None,
    ) -> Array:
        # Map [-pi, pi] to [0, 1]
        domain = (self.angle / (2 * np.pi)) % 1

        if domain_transform is not None:
            domain = domain_transform(domain)

        colour = 255 * (0.5 + np.sin(2 * np.pi * domain) / 2)

        length = self.inner_pixel.distance(self.outer_pixel)

        for p in self.pixels:
            # This is linear - can it be an S-bend?
            r = p.distance(self.inner_pixel) / length
            saturation = 0.5 + np.arctan(A * (r - 0.5)) / np.pi
            image[p.y, p.x] = colour * saturation

        return image


def apply_to_neighbourhood(x: Array, func: ImageTransform) -> Array:
    return func(x, get_neighbourhood(x))


def get_neighbourhood(x: Array) -> Array:
    # TODO: func to build neighbourhood for reuse, include diagonals as well

    u_neighbour = np.pad(x, ((1, 0), (0, 0)), mode="edge")[:-1]
    d_neighbour = np.pad(x, ((0, 1), (0, 0)), mode="edge")[1:]
    l_neighbour = np.pad(x, ((0, 0), (1, 0)), mode="edge")[:, :-1]
    r_neighbour = np.pad(x, ((0, 0), (0, 1)), mode="edge")[:, 1:]

    return np.stack(
        [u_neighbour, d_neighbour, l_neighbour, r_neighbour],
        axis=2,
    )


def distance_to_line(p: Point, anchor1: Point, anchor2: Point) -> float:
    """
    p: the point in question
    anchor1: a point that forms a line with `anchor2`
    anchor2: a point that forms a line with `anchor1`

    Returns: The distance between the point `p` and the line formed by
    `anchor1` and `anchor2`

    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points

    To save computation time, I'm dropping the denominator, as it only depends
    on the anchors.

    """
    x_delta = anchor2.x - anchor1.x
    y_delta = anchor2.y - anchor1.y

    return abs(x_delta * (anchor1.y - p.y) - (anchor1.x - p.x) * y_delta)


def smooth_mat(size: int) -> npt.NDArray[np.float_]:
    ident = np.eye(size)
    pad_ident = np.pad(ident, ((0, 0), (1, 1)))
    l_shift_ident = pad_ident[:, 2:]
    r_shift_ident = pad_ident[:, :-2]
    return (ident + l_shift_ident + r_shift_ident) / 3


def blur(x: Array) -> Array:
    cols = st.columns(3)
    n_small_blur = cols[0].slider("Small Blur", 0, 10, 6)
    n_big_blur = cols[1].slider("Big Blur", 10, 100, 23)
    n_huge_blur = cols[2].slider("Huge Blur", 100, 1000, 200)

    left_smooth_mat = smooth_mat(x.shape[0])
    right_smooth_mat = smooth_mat(x.shape[1])

    with timer("Blur outline"):
        small_blur = (
            np.linalg.matrix_power(left_smooth_mat, n_small_blur)
            @ x @
            np.linalg.matrix_power(right_smooth_mat, n_small_blur)
        )
        small_blur *= 255 / small_blur.max()

        big_blur = (
            np.linalg.matrix_power(left_smooth_mat, n_big_blur)
            @ x @
            np.linalg.matrix_power(right_smooth_mat, n_big_blur)
        )
        big_blur *= 255 / big_blur.max()

        huge_blur = (
            np.linalg.matrix_power(left_smooth_mat, n_huge_blur)
            @ x @
            np.linalg.matrix_power(right_smooth_mat, n_huge_blur)
        )
        huge_blur *= 255 / huge_blur.max()

    return np.stack([huge_blur, big_blur, small_blur], axis=2).max(axis=2)


def outline(image: Image.Image) -> Image.Image:
    x = np.array(image).astype(np.float64).mean(axis=2)

    def _find_edges(x: Array, n: Array) -> Array:
        return (x.reshape(*x.shape, 1) - n).max(axis=2)  # type: ignore

    edges = apply_to_neighbourhood(x, _find_edges)

    # Ray tracing

    traced_pixels: set[Point] = set()
    rays: set[Ray] = set()

    max_y = x.shape[0] - 1
    max_x = x.shape[1] - 1
    all_y = range(max_y + 1)
    all_x = range(max_x + 1)

    outer_edge_pixels = {
        *(Point(0, y) for y in all_y),
        *(Point(max_x, y) for y in all_y),
        *(Point(x, 0) for x in all_x),
        *(Point(x, max_y) for x in all_x),
    }

    img_center = Point(max_x // 2, max_y // 2)

    pbar = st.progress(0.0, "Tracing")
    total = len(outer_edge_pixels)

    # Bulk up the edges to close gaps
    for _ in range(1):
        edges = apply_to_neighbourhood(
            edges,
            lambda x, n: np.concatenate(
                [x.reshape(*x.shape, 1), n],
                axis=2,
            ).max(axis=2),
        )

    with timer("Find rays"):
        for p in outer_edge_pixels:
            ray_pixels: set[Point] = set()

            # Identify where to look for the candidate pixels
            trajectory = img_center - p

            x_offset = 1 if trajectory.x > 0 else -1
            y_offset = 1 if trajectory.y > 0 else -1

            neighbour_offsets = (
                Point(x_offset, 0),
                Point(x_offset, y_offset),
                Point(0, y_offset),
            )

            current = p

            while True:
                if current not in traced_pixels:
                    ray_pixels.add(current)

                candidate_next_pixels = [
                    current + offset for offset in neighbour_offsets
                ]

                current = min(
                    candidate_next_pixels,
                    key=partial(distance_to_line, anchor1=p, anchor2=img_center),
                )

                if edges[current.y, current.x] > 0:
                    break

                if current.distance(img_center) < 2:
                    raise ValueError("There's a leak!")

            rays.add(
                Ray(
                    ray_pixels,
                    p,
                    current,
                    np.arctan2(trajectory.y, trajectory.x),
                )
            )

            traced_pixels |= ray_pixels

            pbar.progress(len(rays) / total)

    blurred_edges = blur(edges)

    # Add butterfly to background for each colour channel

    with timer("Colour in"):
        red = x.copy()
        green = x.copy()
        blue = x.copy()

        r = st.slider("shape", -10.0, 10.0, -2.0)

        for ray in rays:
            red = ray.colour_in(r, red, img_center)
            green = ray.colour_in(
                r, green, img_center, lambda d: (d + (1 / 3)) % 1
            )
            blue = ray.colour_in(r, blue, img_center, lambda d: (d + (2 / 3)) % 1)

        if st.checkbox("Expand to range", value=True):
            red = expand_to_range(red, 0, 255)
            green = expand_to_range(green, 0, 255)
            blue = expand_to_range(blue, 0, 255)

    colours = np.stack(
        [
            np.stack([red, blurred_edges], axis=2).max(axis=2),
            np.stack([green, blurred_edges], axis=2).max(axis=2),
            np.stack([blue, blurred_edges], axis=2).max(axis=2),
        ],
        axis=2,
    )

    return Image.fromarray(colours.astype(np.uint8))


def main() -> None:
    st.title("Make Artwork for Website")

    icon = Image.new("RGB", (512, 512), RGB.GREY)

    with timer("Total"):
        build_shape(icon)
        icon = outline(icon)

        st.image(icon)
        st.sidebar.divider()

    if st.button("Save"):
        icon.save("icon.png")


main()
