import argparse
import sys
from loaders import *
import os
import json
import pathlib

ROOT_PATH = pathlib.Path(__file__).parent.parent.resolve()
OUTPUT_PATH = os.path.join(ROOT_PATH, "output")

def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("--dataset", type=str, help="Dataset to download", default="refinedweb")
    parser.add_argument("--split", type=str, default="", help="Dataset split to download")
    parser.add_argument("--index_from", type=int, default=1, help="Index to start downloading")
    parser.add_argument("--index_to", type=int, default=100, help="Index to stop downloading")
    parser.add_argument("--max_datasize", type=int, default=100, help="max_datasize to download")

    return parser.parse_args()


def main():
    args = parse_args()
    max_data_size = args.max_datasize
    
    if args.dataset == "refinedweb":
        loader = refinedweb_en_loader()
        if max_data_size == -1:
            max_data_size = 968000015
        text_list = iter(loader)
        output_dir_path = os.path.join(OUTPUT_PATH, "refinedweb")
        output_path = os.path.join(output_dir_path, "refinedweb.jsonl")
        content = "content"

    if args.dataset == "slimpajama":
        loader = slimpajama_en_loader()
        if max_data_size == -1:
            max_data_size = 2159581000
        text_list = iter(loader)
        output_dir_path = os.path.join(OUTPUT_PATH, "slimpajama")
        output_path = os.path.join(output_dir_path, "slimpajama.jsonl")
        content = "text"

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    with open(output_path, "w") as f:
        for i in range(1, max_data_size):
            if i < args.index_from or i > args.index_to: 
                continue
            else:
                text = next(text_list)
                out_text = json.dumps({"text": text[content]}, ensure_ascii=False)
                f.write(out_text+"\n")


if __name__ == "__main__":
    main()
