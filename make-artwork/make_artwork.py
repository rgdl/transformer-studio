"""
Need to create an icon (24-bit JPEG or PNG, 512x512) and a header image (24-bit JPEG or PNG, 4096x2304)
"""
from dataclasses import dataclass
from pathlib import Path
from typing_extensions import Self

import streamlit as st
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from font import BUTTERFLIES
from font import FONT
from font import Font
from font import draw_text


def main() -> None:
    st.title("Make Artwork for Website")

    with st.sidebar:
        font_size = st.slider("Font Size", 1, 500, 100)
        glyph = st.selectbox("Glyph", BUTTERFLIES)

    icon = Image.new("RGB", (512, 512), (255, 0, 255))
    font = Font(FONT, font_size)
    img = draw_text(glyph, font, icon)
    st.image(img)


main()
