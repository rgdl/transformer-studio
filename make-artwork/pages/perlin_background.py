from itertools import product

from matplotlib.colors import hsv_to_rgb  # type: ignore
from matplotlib.colors import rgb_to_hsv
import numpy as np
import numpy.typing as npt
import streamlit as st
from PIL import Image

from utils import timer
from utils.perlin import CHUNK_SIZE
from utils.perlin import perlin


# TODO: not for the first transformer studio icon and header, but some time, re-write the perlin generator in rust, so that the compiled binary can be called on to generate the image

PARAMS = {
    "red": 0.4,
    "green": 0.77,
    "blue": 0.83,
    "hue": 0.26,
    "saturation": 0.11,
    "value": 0.25,
    "scale": 7.5,
}

#OUTPUT_SIZE = (2304, 4096)
OUTPUT_SIZE = (CHUNK_SIZE * 2, CHUNK_SIZE * 3)

assert all(s % CHUNK_SIZE == 0 for s in OUTPUT_SIZE)

GRID_SIZE = tuple(s // 256 for s in OUTPUT_SIZE)
GRID_POINTS = np.zeros((*GRID_SIZE, 2), dtype=int)

for i, j in product(*[range(s) for s in GRID_POINTS.shape[:2]]):
    GRID_POINTS[i, j] = [i * CHUNK_SIZE, j * CHUNK_SIZE]


def get_grid() -> npt.NDArray[np.int_]:
    grid = np.zeros(OUTPUT_SIZE, dtype=np.uint8)
    for y, x in product(*[range(s) for s in OUTPUT_SIZE]):
        if (x % CHUNK_SIZE == 0) or (y % CHUNK_SIZE == 0):
            grid[y, x] = 255
    return grid  # type: ignore


def main() -> Image.Image:
    st.title("Perlin noise for a header image")
    rust = st.checkbox("Rust", value=True)

    with st.sidebar:
        draw_grid = st.checkbox("Draw Grid")

        cols = st.columns(3)

        colour_weights = np.array(
            [
                cols[0].slider("Red", 0.0, 1.0, PARAMS["red"]),
                cols[1].slider("Green", 0.0, 1.0, PARAMS["green"]),
                cols[2].slider("Blue", 0.0, 1.0, PARAMS["blue"]),
            ]
        ).reshape(1, 1, 3)

        hsv_shifts = np.array(
            [
                cols[0].slider("Hue Shift", -1.0, 1.0, PARAMS["hue"]),
                cols[1].slider("Saturation Shift", -1.0, 1.0, PARAMS["saturation"]),
                cols[2].slider("Value Shift", -1.0, 1.0, PARAMS["value"]),
            ]
        ).reshape(1, 1, 3)


    scale = CHUNK_SIZE / st.sidebar.slider("Scale Factor", 1.0, 20.0, PARAMS["scale"])

    octave_mix = st.slider("8ve mix", 0.0, 1.0, 0.38)

    # DO all 3 in one go, then slice into 3rds to put in each channel
    with timer("Perlin"):
        all_colours = np.dstack(
            [
                octave_mix * perlin(
                    [OUTPUT_SIZE[0] * 3, OUTPUT_SIZE[1]], scale, rust
                ),
                (1 - octave_mix) * perlin(
                    [OUTPUT_SIZE[0] * 3, OUTPUT_SIZE[1]], scale / 4, rust
                ),
            ]
        ).sum(axis=2)

    red = all_colours[:OUTPUT_SIZE[0], :]
    green = all_colours[OUTPUT_SIZE[0]:2 * OUTPUT_SIZE[0], :]
    blue = all_colours[2 * OUTPUT_SIZE[0]:, :]

    if draw_grid:
        with timer("Draw grid"):
            grid = get_grid()
            red = np.stack([red, grid], axis=2).max(axis=2)

    image_arr = np.stack([red, green, blue], axis=2) * colour_weights

    assert image_arr.max() <= 1.0

    image_arr_hsv = rgb_to_hsv(image_arr)

    image_arr = hsv_to_rgb((image_arr_hsv + hsv_shifts).clip(min=0, max=1))

    image = Image.fromarray((255 * image_arr).astype(np.uint8))

    return image


# TODO: what if I generate the layers as hsv, not rgb?

if __name__ == "__main__":
    with timer("Total"):
        image = main()
        st.sidebar.divider()

    st.image(image)

    if st.button("Save"):
        image.save("header.jpeg")
