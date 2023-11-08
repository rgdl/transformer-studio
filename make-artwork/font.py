from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from typing_extensions import Self

FONT_DIR = Path("~/Library/Fonts").expanduser()
FONT = FONT_DIR / "SauceCodeProNerdFontPropo-Regular.ttf"
BUTTERFLIES = ["", "󱖉", "󱖊"]


class Font:
    file: Path
    name: str
    font_size: int
    image: ImageFont.FreeTypeFont

    def __init__(self, file: Path, size: int) -> None:
        self.file = file
        self.name = file.name
        self.size = size
        self.image = ImageFont.truetype(str(file), size)

    def get_path(self) -> str:
        return str(FONT_DIR / self.name)

    def __repr__(self) -> str:
        return str(self.name)


def get_font_files() -> list[str]:
    return list(FONT_DIR.glob("*"))  # type: ignore


def draw_text(text: str, font: Font, image: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(image)

    text_width, text_height = draw.textsize(  # type: ignore
        text, font=font.image
    )
    img_w, img_h = image.size

    x = (img_w - font.size) // 2
    y = (img_h - font.size) // 2

    draw.text((x, y), text, font=font.image, fill=(0, 0, 0, 255))

    return image
