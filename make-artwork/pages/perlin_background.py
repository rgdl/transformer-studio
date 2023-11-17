import contextlib
import sys
import time
from itertools import product
from pathlib import Path
from typing import Generator

import numpy as np
import numpy.typing as npt
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


OUTPUT_SIZE = (4096, 2304)
#OUTPUT_SIZE = (CHUNK_SIZE * 2, CHUNK_SIZE * 3)

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

    red = np.zeros(OUTPUT_SIZE)
    green = np.zeros(OUTPUT_SIZE)
    blue = np.zeros(OUTPUT_SIZE)

    with timer("Red"):
        red = perlin(red, CHUNK_SIZE)

    with timer("Green"):
        green = perlin(green, CHUNK_SIZE)

    with timer("Blue"):
        blue = perlin(blue, CHUNK_SIZE)

    if draw_grid:
        with timer("Draw grid"):
            grid = get_grid()
            red = np.stack([red, grid], axis=2).max(axis=2)

    image = Image.fromarray(
        np.stack([red, green, blue], axis=2).astype(np.uint8)
    )

    st.image(image)


# TODO: Work out and fix the weird correlation between colour and grid position

if __name__ == "__main__":
    with timer("Total"):
        main()

        with st.sidebar:
            st.divider()
