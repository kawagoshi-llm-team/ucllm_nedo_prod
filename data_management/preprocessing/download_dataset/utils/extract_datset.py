import argparse
from datasets import load_dataset
import json
from tqdm import tqdm
import itertools

parser = argparse.ArgumentParser(description="Extract a specific field from a dataset and save it to a JSON Lines file.")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to load")
parser.add_argument("--dataset_dir", type=str, default=None, help="Directory of the dataset to load (e.g., language)")
parser.add_argument("--dataset_split", type=str, default="train", help="Split of the dataset to load")
parser.add_argument("--field_name", type=str, default="text", help="Name of the field to extract from the dataset")
parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON Lines file")
parser.add_argument("--num_lines", type=int, default=None, help="Number of lines to load and save (default: all)")

args = parser.parse_args()

# Load the CulturaX dataset for Japanese (train split)
dataset = load_dataset(args.dataset_name, args.dataset_dir, split=args.dataset_split, streaming=True)

# Slice the dataset by the specified number of lines
if args.num_lines is not None:
    dataset = itertools.islice(dataset, args.num_lines)

# Save the specified field of the dataset to a JSON Lines file
with open(args.output_file, "w", encoding="utf-8") as f:
    for example in tqdm(dataset, desc="Processing examples"):
        # Write the value of the specified field as a JSON object
        f.write(json.dumps({args.field_name: example[args.field_name]}, ensure_ascii=False) + "\n")