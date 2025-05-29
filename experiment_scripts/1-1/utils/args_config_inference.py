import argparse
import os
from typing import Optional

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for generated images")
    parser.add_argument("--embed_dir", type=str, default="", help="Embedding directory for loading moe weights")
    args = parser.parse_args()

    return args