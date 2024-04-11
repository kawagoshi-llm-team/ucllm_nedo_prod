#!/bin/bash

# コマンドライン引数のチェック
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_dir> <output_dir> <model_prefix> <vocab_size>"
    exit 1
fi

# 引数の割り当て
INPUT_DIR=$1
OUTPUT_DIR=$2
MODEL_PREFIX=$3
VOCAB_SIZE=$4

# ディレクトリの存在確認（存在しなければ作成）
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

echo "Training with the following settings:"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Model prefix: $MODEL_PREFIX"
echo "Vocabulary size: $VOCAB_SIZE"

# 学習データサイズを計算
start_time=$(date +%s)
total_size=$(du -sk "$INPUT_DIR" | cut -f1)
echo "Data size: $total_size KBytes"

# SentencePiece トレーニングの実行
python ./train_sentencepiece_tokenizer.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_prefix "$MODEL_PREFIX" \
    --vocab_size $VOCAB_SIZE

# 処理時間を計算
end_time=$(date +%s)
processing_time=$((end_time - start_time))
echo "Processing time: $processing_time seconds"

# 結果のファイルへの出力（ファイル名は現在時刻で生成）
output_file="${OUTPUT_DIR}/$(date +%Y%m%d%H%M%S).txt"
echo "data_size(KByte):${total_size}" > "$output_file"
echo "processing_time(s):${processing_time}" >> "$output_file"

# 生成されたモデルとボキャブラリファイルを指定された出力ディレクトリに移動
mv "${MODEL_PREFIX}.model" "${OUTPUT_DIR}/"
mv "${MODEL_PREFIX}.vocab" "${OUTPUT_DIR}/"

echo "Model and vocabulary files have been moved to: $OUTPUT_DIR"

