from itertools import product

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

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


def perlin(shape: tuple[int], scale=1.0):
    # Generate random gradient vectors for each grid point
    gradient_shape = (shape[0] + 1, shape[1] + 1, 2)
    gradients = np.random.normal(size=gradient_shape)

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
    dot00 = np.sum(gradients[y0, x0] * np.dstack((dx0, dy0)), axis=2)
    dot10 = np.sum(gradients[y0, x1] * np.dstack((dx1, dy0)), axis=2)
    dot01 = np.sum(gradients[y1, x0] * np.dstack((dx0, dy1)), axis=2)
    dot11 = np.sum(gradients[y1, x1] * np.dstack((dx1, dy1)), axis=2)

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
