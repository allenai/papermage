from typing import AnyStr
from charset_normalizer import detect


def maybe_normalize(text: AnyStr) -> str:
    if isinstance(text, bytes):
        enc = str(detect(text)['encoding'])
        text = text.decode(enc)     # type: ignore
    return text     # type: ignore
