import os
import sys
import json
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="change")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="The path of input file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="The path of output file.",
    )
    args = parser.parse_args()
    return args


args = parse_args()
data = None
with open(args.input, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

with open(args.output, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=2, ensure_ascii=False)
