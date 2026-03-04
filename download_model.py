"""
download_model.py — fetch GoogLeNet model files for the CV Workshop.

Run once before the workshop:
    python download_model.py

Downloads ~50 MB total. Files are saved to the repo root.
"""

import urllib.request
import os
import sys


FILES = {
    "bvlc_googlenet.caffemodel": (
        "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel",
        "~50 MB — may take a minute"
    ),
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/BVLC/caffe/master/"
        "models/bvlc_googlenet/deploy.prototxt",
        "small"
    ),
    "synset_words.txt": (
        "https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/"
        "master/data/ilsvrc12/synset_words.txt",
        "small"
    ),
}


def download(filename, url, note):
    if os.path.exists(filename):
        print(f"  [✓] {filename} already exists, skipping.")
        return

    print(f"  Downloading {filename}  ({note}) ...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"  [✓] {filename} saved.")
    except Exception as e:
        print(f"  [✗] Failed to download {filename}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("\nCV Workshop — downloading model files\n")
    for filename, (url, note) in FILES.items():
        download(filename, url, note)
    print("\nAll files ready. You can now run:  python main.py\n")