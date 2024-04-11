# Appends a path to import python scripts that are in other directories.
import os
import sys
import time
import glob
sys.path.append(os.path.join(os.environ["HOME"], "ucllm_nedo_prod/train/scripts/common/"))

import argparse
import sentencepiece as spm
from special_token_list import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, CLS_TOKEN, SEP_TOKEN, EOD_TOKEN, MASK_TOKEN, NEWLINE_TOKEN


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--model_prefix", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe", "word", "char"])
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--train_extremely_large_corpus", type=bool, default=True)
    args = parser.parse_args()
    print(f"{args = }")
    return args

# 学習データサイズを計算（ディレクトリ内の全.txtファイルのサイズをKByte単位で計算）
def calculate_files_size_kbytes(directory):
    total_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(directory, "*.txt")))
    return total_size / 1024  # KBytesに変換

def main():
    args = parse_arguments()

    # 学習対象データの読み込み
    all_text_files = glob.glob(os.path.join(args.input_dir, "*.txt"))
    input_files_str = ",".join(all_text_files)  # SentencePieceに渡す形式に変換

    # 学習データサイズを計算
    data_size = calculate_files_size_kbytes(args.input_dir)
    
    # トレーニング開始時間を記録
    start_time = time.time()

      # 処理開始日時をYYYYMMDDHHMISS形式で取得
    start_time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(start_time))

    # Trains a SentencePiece tokenizer. After training, *.model and *.vocab will be saved in the current directory.
    spm.SentencePieceTrainer.train(
        input=input_files_str,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        num_threads=args.num_threads,
        train_extremely_large_corpus=args.train_extremely_large_corpus,
        user_defined_symbols=[
            BOS_TOKEN,
            EOS_TOKEN,
            PAD_TOKEN,
            CLS_TOKEN,
            SEP_TOKEN,
            EOD_TOKEN,
            MASK_TOKEN,
            NEWLINE_TOKEN,
        ],  # Note: `NEWLINE_TOKEN` is needed in `user_defined_symbols`.
        byte_fallback=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
    )

    # トレーニング終了時間を記録し、処理時間を計算
    end_time = time.time()
    processing_time = (end_time - start_time)

    # ファイル名を処理開始日時でフォーマット
    output_file_name = f"{start_time_str}.txt"

    # 結果をファイルに出力
    with open(output_file_name, "w") as file:
        file.write(f"data_size(KByte):{data_size:.2f}\n")
        file.write(f"processing_time(s)：{processing_time:.2f}")

if __name__ == "__main__":
    main()
