"""key 规范：kebab-case，可含中文。"""

import re
import unicodedata


def keyify(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"[^\w\-一-鿿]", "", text)
    return text.strip("-")
