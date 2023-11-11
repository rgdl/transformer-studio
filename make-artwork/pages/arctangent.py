import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.typing as npt
import streamlit as st


def expand_to_range(
    x: npt.NDArray[np.float_],
    target_min: float,
    target_max: float,
) -> npt.NDArray[np.float_]:
    x = x.copy()

    x -= x.min()
    x += target_min

    x /= x.max()
    x *= target_max

    return x


if __name__ == "__main__":
    st.title("Arctangent")
    st.write(
        "How to transform the inputs and outputs of an arctangent function to "
        "give the shape I want."
    )
    st.write("The domain and range of the function should both be `[0,1]`")

    with st.sidebar:
        lower_limit = st.slider("Lower X", -100, 100, 0)
        upper_limit = st.slider("Upper X", -100, 100, 1)
        multiplier = st.slider("Multiplier", -100, 100, 36)
        expand = st.checkbox("Expand to range")

    x = np.linspace(lower_limit, upper_limit)
    y = 0.5 + np.arctan(multiplier * (x - 0.5)) / np.pi

    if expand:
        y = expand_to_range(y, 0, 1)

    fig = plt.figure()

    plt.plot(x, y)
    plt.hlines(
        [0, 1], lower_limit, upper_limit, color="red", linestyle="dashed"
    )

    plt.hlines(
        [0.5], lower_limit, upper_limit, color="black", linestyle="dashed"
    )
    plt.vlines([0.5], 0, 1, color="black", linestyle="dashed")

    st.pyplot(fig)
