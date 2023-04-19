import json
import pathlib

with open(pathlib.Path(__file__).parent / "AllowedChars.json", "r", encoding="utf8") as f:
    AllowedCharacters = json.load(f)
