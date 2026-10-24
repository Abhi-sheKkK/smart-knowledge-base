"""Build a collection directly from folders of PDFs (dev shortcut for /index).

usage: python scripts/index_folder.py <collection_name> <folder> [<folder> ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb.index import build_collection


def main():
    name, folders = sys.argv[1], sys.argv[2:]
    pdfs = [p for f in folders for p in Path(f).glob("*.pdf")]
    build_collection(name, pdfs)


if __name__ == "__main__":
    main()
