from transformers import AutoTokenizer
import argparse
import os

def load_tokenizer_and_model(input_tokenizer_and_model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_and_model_dir)
  
    return tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tokenizer_dir", type=str, required=False, default="kawagoshi-llm-team/test_tokenizer")
    parser.add_argument("--output_tokenizer_dir", type=str, required=False, default="~/ucllm_nedo_prod/pipeline/step3_tokenization/output/download_tokenizer")
    args = parser.parse_args()
    print(f"{args = }")
    return args

def main() -> None:
    args = parse_arguments()
    # Loads and tests the local tokenizer and the local model.
    tokenizer = load_tokenizer_and_model(args.input_tokenizer_dir)
    tokenizer.save_pretrained(os.path.expanduser(args.output_tokenizer_dir))

if __name__ == "__main__":
    main()