import contextlib
import time
from typing import Generator

import streamlit as st


@contextlib.contextmanager
def timer(label: str) -> Generator:
    t0 = time.time()
    yield
    t1 = time.time()
    with st.sidebar:
        st.write(f"{label} `{t1 - t0:.02f} seconds`")
