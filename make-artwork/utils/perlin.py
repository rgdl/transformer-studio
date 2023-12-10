from itertools import product

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

import rust_perlin
from utils import timer

CHUNK_SIZE = 256


def _as_heatmap(array, title):
    fig = plt.figure()
    sns.heatmap(array)
    plt.title(title)
    st.pyplot(fig)

def smoothstep(x: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    if (x.min() < 0) or (x.max() > 1):
        st.write(pd.Series(x).describe())
        raise ValueError("Bad inputs to `smoothstep`")

    return x * x * (3 - 2 * x)


def get_gradients(shape: tuple[int, int]) -> npt.NDArray[np.float_]:
    # TODO: keep this as the python version for reproducibility checks, given its random nature
    gradient_shape = (shape[0] + 1, shape[1] + 1, 2)
    np.random.seed(777)
    return np.random.normal(size=gradient_shape)


#@st.cache_data
def perlin(shape: tuple[int, int], scale=1.0, rust: bool = False):
    with timer("PERLIN - generate_gradients"):
        if rust and False:
            gradients = np.array(rust_perlin.random_normal(*shape))
        else:
            gradients = get_gradients(shape)

    with timer("PERLIN - grid"):
        # Generate a grid of coordinates
        if rust:
            x_grid, y_grid = (np.array(a) for a in rust_perlin.make_grids(*shape, scale))
        else:
            x = np.linspace(0, scale, num=shape[1], endpoint=False)
            y = np.linspace(0, scale, num=shape[0], endpoint=False)
            x_grid, y_grid = np.meshgrid(x, y)

    with timer("PERLIN - indices"):
        # Calculate the indices of the four nearest grid points
        if rust:
            x0 = np.array(rust_perlin.quantise_grid(x_grid, quantise_up=False))
            x1 = np.array(rust_perlin.quantise_grid(x_grid, quantise_up=True))
            y0 = np.array(rust_perlin.quantise_grid(y_grid, quantise_up=False))
            y1 = np.array(rust_perlin.quantise_grid(y_grid, quantise_up=True))
        else:
            x0 = np.floor(x_grid).astype(int)
            x1 = x0 + 1
            y0 = np.floor(y_grid).astype(int)
            y1 = y0 + 1

    with timer("PERLIN - distance vectors"):
        # Calculate the distance vectors from the grid points to the coordinates
        if rust:
            dx0 = rust_perlin.add_grid(
                x_grid,
                rust_perlin.multiply_grid(-1, x0),
            )
            dx1 = rust_perlin.add_grid(
                x_grid,
                rust_perlin.multiply_grid(-1, x1),
            )
            dy0 = rust_perlin.add_grid(
                y_grid,
                rust_perlin.multiply_grid(-1, y0),
            )
            dy1 = rust_perlin.add_grid(
                y_grid,
                rust_perlin.multiply_grid(-1, y1),
            )
        else:
            dx0 = x_grid - x0
            dx1 = x_grid - x1
            dy0 = y_grid - y0
            dy1 = y_grid - y1

    with timer("PERLIN - dot products"):
        # Calculate the dot products between the gradient vectors and the distance vectors
        dot00 = np.sum(gradients[y0, x0] * np.dstack((dx0, dy0)), axis=2)
        dot10 = np.sum(gradients[y0, x1] * np.dstack((dx1, dy0)), axis=2)
        dot01 = np.sum(gradients[y1, x0] * np.dstack((dx0, dy1)), axis=2)
        dot11 = np.sum(gradients[y1, x1] * np.dstack((dx1, dy1)), axis=2)

    with timer("PERLIN - interpolation"):
        # Interpolate the dot product values using smoothstep function
        if rust:
            u = rust_perlin.smoothstep(dx0)
            v = rust_perlin.smoothstep(dy0)
        else:
            u = smoothstep(dx0)
            v = smoothstep(dy0)

        # Interpolate along the x-axis
        interpolate0 = dot00 + u * (dot10 - dot00)
        interpolate1 = dot01 + u * (dot11 - dot01)

        # Interpolate along the y-axis to get the final noise value
        final_noise = interpolate0 + v * (interpolate1 - interpolate0)

        final_noise -= final_noise.min()
        final_noise /= final_noise.max()

    st.sidebar.divider()
    return final_noise


if __name__ == "__main__":
    print(perlin(2, 3))
