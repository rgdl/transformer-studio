from itertools import product

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

import rust_perlin

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
    if rust and False:
        gradients = np.array(rust_perlin.random_normal(*shape))
    else:
        gradients = get_gradients(shape)

    if rust:
        return np.array(rust_perlin.perlin(gradients, scale))

    # Generate a grid of coordinates
    x = np.linspace(0, scale, num=shape[1], endpoint=False)
    y = np.linspace(0, scale, num=shape[0], endpoint=False)
    x_grid, y_grid = np.meshgrid(x, y)

    # Calculate the indices of the four nearest grid points
    x0 = np.floor(x_grid).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y_grid).astype(int)
    y1 = y0 + 1

    # Calculate the distance vectors from the grid points to the coordinates
    dx0 = x_grid - x0
    dx1 = x_grid - x1
    dy0 = y_grid - y0
    dy1 = y_grid - y1

    # Calculate the dot products between the gradient vectors and the distance vectors
    corner00 = gradients[y0, x0]
    corner10 = gradients[y0, x1]
    corner01 = gradients[y1, x0]
    corner11 = gradients[y1, x1]

    dist00 = np.dstack((dx0, dy0))
    dist10 = np.dstack((dx1, dy0))
    dist01 = np.dstack((dx0, dy1))
    dist11 = np.dstack((dx1, dy1))

    dot00 = np.sum(corner00 * dist00, axis=2)
    dot10 = np.sum(corner10 * dist10, axis=2)
    dot01 = np.sum(corner01 * dist01, axis=2)
    dot11 = np.sum(corner11 * dist11, axis=2)

    # Interpolate the dot product values using smoothstep function
    u = smoothstep(dx0)
    v = smoothstep(dy0)

    # Interpolate along the x-axis
    interpolate0 = dot00 + u * (dot10 - dot00)
    interpolate1 = dot01 + u * (dot11 - dot01)

    # Interpolate along the y-axis to get the final noise value
    final_noise = interpolate0 + v * (interpolate1 - interpolate0)

    final_noise -= final_noise.min()
    final_noise /= final_noise.max()

    return final_noise


if __name__ == "__main__":
    print(perlin(2, 3))
