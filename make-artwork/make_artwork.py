"""
Need to create:
    * an icon (24-bit JPEG or PNG, 512x512)
    * a header image (24-bit JPEG or PNG, 4096x2304)
"""
import time
from dataclasses import dataclass
from functools import partial
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
    angle: float

    def __hash__(self) -> int:
        return hash(self.angle)

    def colour_in(self, image: Array) -> None:
        domain = ((self.angle / (2 * np.pi)) % 1)
        # Colour is a function of angle
        colour = domain * 255

        for p in self.pixels:
            image[p.y, p.x] = colour


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
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points

    To save computation time, I'm dropping the denominator, as it only depends
    on the anchors.
    """
    x_delta = anchor2.x - anchor1.x
    y_delta = anchor2.y - anchor1.y

    return abs(
        x_delta * (anchor1.y - p.y) - (anchor1.x - p.x) * y_delta
    )


def blur(x: Array) -> Array:
    def _smooth(x: Array, n: Array) -> Array:
        n = np.concatenate([x.reshape(*x.shape, 1), n], axis=2)
        smoothed = n.mean(axis=2)

        # Bring max back up
        smoothed *= 255 / smoothed.max()

        return smoothed  # type: ignore

    huge_blur = x.copy()
    big_blur = x.copy()
    small_blur = x.copy()

    cols = st.columns(3)
    n_small_blur = cols[0].slider("Small Blur", 0, 10, 3)
    n_big_blur = cols[1].slider("Big Blur", 10, 100, 23)
    n_huge_blur = cols[2].slider("Huge Blur", 100, 1000, 265)

    total_blur = n_small_blur + n_big_blur + n_huge_blur
    pbar = st.progress(0.0)
    blurs_completed = 0

    t0 = time.time()

    for _ in range(n_small_blur):
        small_blur = apply_to_neighbourhood(small_blur, _smooth)
        blurs_completed += 1
        pbar.progress(blurs_completed / total_blur)

    with st.sidebar:
        st.write(f"Small blur: `{time.time() - t0:.02f} seconds`")

    t0 = time.time()

    for _ in range(n_big_blur):
        big_blur = apply_to_neighbourhood(big_blur, _smooth)
        blurs_completed += 1
        pbar.progress(blurs_completed / total_blur)

    with st.sidebar:
        st.write(f"Big blur: `{time.time() - t0:.02f} seconds`")

    t0 = time.time()

    for _ in range(n_huge_blur):
        huge_blur = apply_to_neighbourhood(huge_blur, _smooth)
        blurs_completed += 1
        pbar.progress(blurs_completed / total_blur)

    with st.sidebar:
        st.write(f"Huge blur: `{time.time() - t0:.02f} seconds`")

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

    t0 = time.time()

    for p in outer_edge_pixels:

        ray_pixels: set[Point] = set()

        # Identify the direction, so we know which are the candidate pixels
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

        rays.add(Ray(ray_pixels, np.arctan2(trajectory.y, trajectory.x)))

        traced_pixels |= ray_pixels

        pbar.progress(len(rays) / total)

    with st.sidebar:
        st.write(f"Find rays: `{time.time() - t0:.02f} seconds`")

    t0 = time.time()

    for ray in rays:
        ray.colour_in(x)

    with st.sidebar:
        st.write(f"Colour in: `{time.time() - t0:.02f} seconds`")

    x = np.stack([x, blur(edges)], axis=2).max(axis=2)

    colours = np.stack([x for _ in range(3)], axis=2)
    return Image.fromarray(colours.astype(np.uint8))


def main() -> None:
    st.title("Make Artwork for Website")

    icon = Image.new("RGB", (512, 512), RGB.GREY)

    t0 = time.time()

    build_shape(icon)
    icon = outline(icon)

    st.image(icon)

    with st.sidebar:
        st.divider()
        st.write(f"Total Duration: `{time.time() - t0:.02f} seconds`")


main()
