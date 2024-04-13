import argparse
from datetime import datetime
import json
from hojichar import document_filters, tokenization, Compose, Document
import pyarrow.parquet as pq
import os
import gzip

from preprocessing import custom_token_filters, custom_tokenization, custom_document_filters

def process_json_lines(lines: list[str], output_base: str, stats: list[dict]):
    remained_lines = []
    cleaner = Compose([
        document_filters.JSONLoader(), # テキストをJsonとして解釈し,`key`で指定した要素を文字列としてdoumentに格納.デフォルト`key`は'text'.
        document_filters.DocumentNormalizer(), # 文書の正規化を行う.デフォルトでは全角文字を半角文字に変換する.(NFKC)
        #document_filters.DiscardBBSComments(), # 正規表現 "BBS Patern" に `max_allow_num`(default: 14) 回よりたくさんマッチする文書を破棄。
        #document_filters.DiscardAds(), # 主に広告キーワードを`max_allow_num`より多く含む文書を破棄.
        #document_filters.DiscardViolenceContentJa(), # 日本語の暴力表現キーワードを含む文書を破棄.
        #document_filters.DiscardDiscriminationContentJa(), # 日本語の差別キーワード(および不適切語)を含む文書を破棄.
        document_filters.DocumentLengthFilter(min_doc_len=10, max_doc_len=100000), # 文書の長さがmin_doc_lenより短い、またはmax_doc_lenより長い文書を破棄.
        custom_document_filters.DiscardAdultContentJa(), # 日本語の成人向けコンテンツを閾値に応じて排除.
        custom_document_filters.DiscardAds(),
        custom_document_filters.DiscardViolenceContentJa(),
        custom_document_filters.DiscardDiscriminationContentJa(),
        custom_document_filters.DiscardWithCharacterRatio(),
        #custom_document_filters.DiscardAdultContentWithEmbedding(),
        #custom_tokenization.NewLineSentenceTokenizer(), # 改行文字で文章を区切る.
        #custom_token_filters.RemoveOneword(), # 1単語のみ以下のパターンを削除.
        #custom_tokenization.MergeTokens(delimiter="\n"), # 破棄されていないトークンをdelimeterで結合.
        #custom_tokenization.WakatiTokenizer(), # fugashi を用いて文を分割.
        #custom_token_filters.RemoveDate(), # 日付のみのパターンを削除.
        #tokenization.MergeTokens(), # 破棄されていないトークンを結合.
        #custom_document_filters.SelectJapanese(lookup_size=50), # 日本語以外の文書を排除.
        document_filters.MaskPersonalInformation(), # キュメントに含まれる電話番号・電子メールアドレスを一部マスキング.
        document_filters.JSONDumper(dump_reason=False), # dump_reason=Tureの場合、ドキュメントの破棄事由をJSON形式で付与して出力. JSON形式で出力するため、最後に実行する.（デバッグ用）
    ])

    #checkpoint

    with open(os.path.join(output_base, "rejected.filtering.jsonl"), "w") as rejected:
        with open(os.path.join(output_base, "result.filtering.jsonl"), "w") as writer:
            for line in lines:
                result = cleaner.apply(Document(line)) # textの中身をDocumentに入れてapplyし、cleanerの中で定義された処理を実行   
                if result.is_rejected:
                    rejected.write(result.text + "\n")
                else:
                    writer.write(result.text + "\n")
                    remained_lines.append(json.loads(result.text))

    with open(os.path.join(output_base, "stat.filtering.jsonl"), "w") as writer:
        writer.write(json.dumps(cleaner.statistics, ensure_ascii=False) + "\n")

    stats.append(cleaner.statistics)

    return remained_lines

def __readline(input_file: str):
    if input_file.endswith(".gz"):
        with gzip.open(input_file, "rt") as fp:
            for line in fp:
                yield line.strip()
    elif input_file.endswith(".parquet"):
        table = pq.read_table(input_file)
        for batch in table.to_batches():
            for i in range(batch.num_rows):
                yield json.dumps({"text": batch[0].to_pylist()[i]}, ensure_ascii=False)
    else:
        with open(input_file, "r") as fp:
            for line in fp:
                yield line.strip()

def filtering(input_dir: str, output_base: str):
    os.makedirs(output_base, exist_ok=True)

    input_files = [input_file for input_file in os.listdir(input_dir)
                   if input_file.endswith((".jsonl", ".jsonl.gz", ".txt", ".txt.gz", ".parquet"))]

    stats = []
    with gzip.open(os.path.join(output_base, "results.filtering.jsonl.gz"), "wt") as writer:
        for input_file in input_files:
            input_file_prefix = os.path.splitext(os.path.basename(input_file))[0]
            output_base_for_input: str = os.path.join(output_base, input_file_prefix)
            os.makedirs(output_base_for_input, exist_ok=True)

            for line in process_json_lines(__readline(os.path.join(input_dir, input_file)), output_base_for_input, stats):
                json.dump(line, writer, ensure_ascii=False)
                writer.write("\n")

    with gzip.open(os.path.join(output_base, "stats.filtering.jsonl.gz"), "wt") as writer:
        for stat in stats:
            json.dump(stat, writer, ensure_ascii=False)
            writer.write("\n")

def main():
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--input_dir', type=str,
                        help='The input directory containing documents to process', required=False, default="../step00_download_datasets/output/wiki_ja")
    parser.add_argument('--output_dir', type=str,
                        help='The input file containing documents to process', required=False, default="./output")
    args = parser.parse_args()

    output_base = os.path.join(args.output_dir, "filterd_documents")

    filtering(input_dir=args.input_dir, output_base=output_base)


if __name__ == "__main__":
    main()
