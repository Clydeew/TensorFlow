"""Dataset language metadata. Presets never imply pretrained-model competence."""

import re

PRESETS = {
    "id": {"name": "Indonesian", "ocr": "ind", "scripts": ["Latn"]},
    "en": {"name": "English", "ocr": "eng", "scripts": ["Latn"]},
    "jv": {"name": "Javanese", "ocr": "jav", "scripts": ["Latn", "Java"]},
    "su": {"name": "Sundanese", "ocr": "sun", "scripts": ["Latn", "Sund"]},
    "mad": {"name": "Madurese", "ocr": None, "scripts": ["Latn"]},
    "min": {"name": "Minangkabau", "ocr": None, "scripts": ["Latn", "Arab"]},
    "ban": {"name": "Balinese", "ocr": None, "scripts": ["Latn", "Bali"]},
    "bug": {"name": "Buginese", "ocr": None, "scripts": ["Latn", "Bugi"]},
    "ace": {"name": "Acehnese", "ocr": None, "scripts": ["Latn", "Arab"]},
    "bjn": {"name": "Banjarese", "ocr": None, "scripts": ["Latn"]},
    "bbc": {"name": "Toba Batak", "ocr": None, "scripts": ["Latn", "Batk"]},
    "mak": {"name": "Makassarese", "ocr": None, "scripts": ["Latn", "Bugi"]},
    "sas": {"name": "Sasak", "ocr": None, "scripts": ["Latn"]},
    "ms": {"name": "Malay", "ocr": "msa", "scripts": ["Latn", "Arab"]},
    "und": {"name": "Undetermined", "ocr": None, "scripts": []},
}


def validate_language(value):
    if not isinstance(value, dict):
        raise ValueError("language must be a mapping")
    if not isinstance(value.get("additional", []), list):
        raise ValueError("language.additional must be a list of language tags")
    tags = [value.get("primary", "und"), *value.get("additional", [])]
    if not all(isinstance(t, str) and re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", t) for t in tags):
        raise ValueError("Use language tags such as id, en, jv, su, or a custom two/three-letter tag")
    if value.get("normalization", "NFC") not in ("NFC", "NFD", "none"):
        raise ValueError("language.normalization must be NFC, NFD or none")
    for key in ("code_switching", "preserve_case"):
        if key in value and not isinstance(value[key], bool):
            raise ValueError(f"language.{key} must be boolean")
    script = value.get("script", "Latn")
    if not isinstance(script, str) or not re.fullmatch(r"[A-Z][a-z]{3}|mixed", script):
        raise ValueError("language.script must be an ISO 15924-style code (Latn, Java, Sund) or mixed")
    return value
