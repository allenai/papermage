from typing import Union
from charset_normalizer import detect


def maybe_normalize(text: Union[str, bytes]) -> str:
    """If text is bytes, detect encoding and decode to str."""

    if isinstance(text, bytes):
        enc = str(detect(text)['encoding'])
        text = text.decode(enc)

    return text
