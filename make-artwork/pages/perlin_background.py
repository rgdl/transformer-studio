import contextlib
import sys
import time
from itertools import product
from pathlib import Path
from typing import Generator

from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
import numpy as np
import numpy.typing as npt
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from perlin import CHUNK_SIZE  # type: ignore
from perlin import perlin  # type: ignore


@contextlib.contextmanager
def timer(label: str) -> Generator:
    t0 = time.time()
    yield
    t1 = time.time()
    with st.sidebar:
        st.write(f"{label} `{t1 - t0:.02f} seconds`")


# OUTPUT_SIZE = (4096, 2304)
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


def main() -> None:
    st.title("Perlin noise for a header image")
    st.write(
        "How about an independent layer of noise for each colour channel?"
    )

    with st.sidebar:
        draw_grid = st.checkbox("Draw Grid")

        cols = st.columns(3)

        colour_weights = np.array(
            [
                cols[0].slider("Red", 0.0, 1.0, 1.0),
                cols[1].slider("Green", 0.0, 1.0, 1.0),
                cols[2].slider("Blue", 0.0, 1.0, 1.0),
            ]
        ).reshape(1, 1, 3)

        hsv_shifts = np.array(
            [
                cols[0].slider("Hue Shift", -1.0, 1.0, 0.0),
                cols[1].slider("Saturation Shift", -1.0, 1.0, 0.0),
                cols[2].slider("Value Shift", -1.0, 1.0, 0.0),
            ]
        ).reshape(1, 1, 3)


    scale = CHUNK_SIZE / st.sidebar.slider("Scale Factor", 1.0, 20.0)

    octave_mix = st.slider("8ve mix", 0.0, 1.0, 0.5)

    # DO all 3 in one go, then slice into 3rds to put in each channel
    with timer("Perlin"):
        all_colours = np.dstack(
            [
                octave_mix * perlin(
                    [OUTPUT_SIZE[0] * 3, OUTPUT_SIZE[1]], scale
                ),
                (1 - octave_mix) * perlin(
                    [OUTPUT_SIZE[0] * 3, OUTPUT_SIZE[1]], scale / 4
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

    st.image(image)


# TODO: what if I generate the layers as hsv, not rgb?

if __name__ == "__main__":
    with timer("Total"):
        main()

        with st.sidebar:
            st.divider()
