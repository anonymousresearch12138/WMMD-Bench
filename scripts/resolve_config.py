"""Bind storage placeholders without changing scientific configuration fields."""
import argparse
import os
from pathlib import Path
import re


def resolve(text, bindings):
    names = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)\}", text))
    missing = sorted(name for name in names if not bindings.get(name))
    if missing:
        raise ValueError("Missing bindings: " + ", ".join(missing))
    # JSON and YAML templates use POSIX paths; the research workers target Linux.
    for name in names:
        value = bindings[name]
        if any(c in value for c in ('"', "'", "\\", "\n", "\r", "$")):
            raise ValueError(f"Use a plain POSIX path for {name}")
        text = text.replace("${" + name + "}", value)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = resolve(args.input.read_text(encoding="utf-8"), os.environ)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(text)


if __name__ == "__main__":
    main()
