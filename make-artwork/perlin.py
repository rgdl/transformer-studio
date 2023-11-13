from itertools import product

import numpy as np
import numpy.typing as npt

CHUNK_SIZE = 256


def _get_node_distances_and_angles(
    orig: npt.NDArray[np.float_],
    is_left: bool,
    is_top: bool,
    chunk_size: int,
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    points = np.stack(
        [
            np.repeat(np.array([range(orig.shape[0])]).T, orig.shape[1], axis=1),
            np.repeat([range(orig.shape[1])], orig.shape[0], axis=0),
        ],
        axis=2
    )

    assert points.shape == (*orig.shape, 2)

    compare_points = chunk_size * (
        (points // chunk_size) + np.array(
            np.array([[[0 if is_left else 1, 0 if is_top else 1]]])
        )
    )

    displacements = points - compare_points

    distances = (displacements ** 2).sum(axis=2) ** 0.5
    angles = np.arctan2(displacements[:, :, 1], displacements[:, :, 0])

    return distances, angles


def perlin(
    canvas: npt.NDArray[np.float_],
    chunk_size: int = CHUNK_SIZE,
) -> npt.NDArray[np.float_]:
    grid_size = tuple(s // chunk_size for s in canvas.shape)

    # Random unit gradient vectors for each grid node
    gradient_angles = np.random.uniform(0, 2 * np.pi, grid_size)

    all_gradient_angles = np.zeros_like(canvas)

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            all_gradient_angles[
                i * chunk_size:(i + 1) * chunk_size,
                j * chunk_size:(j + 1) * chunk_size,
            ] = gradient_angles[i, j]

    # Dot products between each point and its 4 closest gradient vectors
    all_dot_products = []

    for is_left, is_top in product([True, False], repeat=2):
        distances, angles = _get_node_distances_and_angles(
            canvas, is_left, is_top, chunk_size
        )

        dot_products = distances * (
            np.cos(all_gradient_angles) * np.cos(angles)
            + np.sin(all_gradient_angles) * np.sin(angles)
        )

        row_weights = 0.5 * np.cos(
            np.pi * np.arange(canvas.shape[0]) / chunk_size
        ) + 0.5

        if not is_top:
            row_weights = 1 - row_weights

        col_weights = 0.5 * np.cos(
            np.pi * np.arange(canvas.shape[1]) / chunk_size
        ) + 0.5

        if not is_left:
            col_weights = 1 - col_weights

        cell_weights = row_weights.reshape(-1, 1) @ col_weights.reshape(1, -1)

        all_dot_products.append(dot_products * cell_weights)

    result = np.stack(all_dot_products, axis=2).sum(axis=2)

    # Return a noise map with values in the interval [0, 255]
    result -= result.min()
    result /= result.max()
    result *= 255
    return result


if __name__ == "__main__":
    # TODO: there's something wrong with the smoothing
    perlin(np.zeros((4096, 2304)))
