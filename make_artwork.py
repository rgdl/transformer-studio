from dataclasses import dataclass
from pathlib import Path
from typing_extensions import Self

import streamlit as st
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

FONT_DIR = Path("/System/Library/Fonts")


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
        #font_file = st.selectbox("Font", get_font_files(), format_func=lambda f: f.name)

    img_per_row = 4

    for idx, font_file in enumerate(get_font_files()):

        if idx % img_per_row == 0:
            cont = st.container()
            cols = iter(st.columns(img_per_row))

        col = next(cols)

        with cont:
            try:
                font = Font(font_file, font_size)
                col.write(font.name)
                img = draw_text("🦋", font)
                col.image(img)

            except Exception as e:
                col.error(e)



main()
