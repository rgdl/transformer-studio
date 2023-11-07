from dataclasses import dataclass
from pathlib import Path
from typing_extensions import Self

import streamlit as st
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

FONT_DIR = Path("~/Library/Fonts").expanduser()
FONT = FONT_DIR / "SauceCodeProNerdFontPropo-Regular.ttf"
BUTTERFLIES = ["", "󱖉", "󱖊"]


class Font:
    file: Path
    name: str
    font_size: int
    image: ImageFont

    def __init__(self, file: Path, size: int) -> Self:
        self.file = file
        self.name = file.name
        self.size = size
        self.image = ImageFont.truetype(str(file), size)

    def get_path(self) -> str:
        return str(FONT_DIR / self.filename)
    
    def __repr__(self) -> str:
        return str(self.filename.name)


def get_font_files() -> list[str]:
    return list(FONT_DIR.glob("*"))


def draw_text(text: str, font: Font) -> Image.Image:
    image = Image.new("RGBA", (font.size, font.size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    text_width, text_height = draw.textsize(text, font=font.image)

    x = (font.size - text_width) // 2
    y = (font.size - text_height) // 2

    draw.text((x, y), text, font=font.image, fill=(0, 0, 0, 255))
    
    return image


def main() -> None:
    st.title("Make Artwork for Website")

    with st.sidebar:
        font_size = st.slider("Font Size", 1, 500, 100)
        glyph = st.selectbox("Glyph", BUTTERFLIES)

    font = Font(FONT, font_size)
    img = draw_text(glyph, font)
    st.image(img)


main()
