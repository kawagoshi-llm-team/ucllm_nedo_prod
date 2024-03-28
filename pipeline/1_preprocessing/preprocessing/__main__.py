import argparse
from datetime import datetime
import json
from hojichar import document_filters, tokenization, Compose, Document
import os

from preprocessing.filtering import custom_token_filters, custom_tokenization, custom_document_filters


def process_json_lines(lines: list[str], output_base: str, stats: list[dict]):
    remained_lines = []
    cleaner = Compose([
        document_filters.JSONLoader(), # テキストをJsonとして解釈し,`key`で指定した要素を文字列としてdoumentに格納.デフォルト`key`は'text'.
        document_filters.DocumentNormalizer(), # 文書の正規化を行う.デフォルトでは全角文字を半角文字に変換する.(NFKC)
        document_filters.DiscardBBSComments(), # 正規表現 "BBS Patern" に `max_allow_num`(default: 14) 回よりたくさんマッチする文書を破棄。
        document_filters.DiscardAds(), # 主に広告キーワードを`max_allow_num`より多く含む文書を破棄.
        document_filters.DiscardDiscriminationContentJa(), # 日本語の差別キーワード(および不適切語)を含む文書を破棄.
        document_filters.DocumentLengthFilter(min_doc_len=10, max_doc_len=100000), # 文書の長さがmin_doc_lenより短い、またはmax_doc_lenより長い文書を破棄.
        custom_document_filters.DiscardAdultContentJa(), # 日本語の成人向けコンテンツを閾値に応じて排除.
        custom_tokenization.NewLineSentenceTokenizer(), # 改行文字で文章を区切る.
        custom_token_filters.RemoveOneword(), # 1単語のみ以下のパターンを削除.
        custom_tokenization.MergeTokens(delimiter="\n"), # 破棄されていないトークンをdelimeterで結合.
        custom_tokenization.WakatiTokenizer(), # fugashi を用いて文を分割.
        custom_token_filters.RemoveDate(), # 日付のみのパターンを削除.
        tokenization.MergeTokens(), # 破棄されていないトークンを結合.
        custom_document_filters.SelectJapanese(lookup_size=50), # 日本語以外の文書を排除.
        document_filters.MaskPersonalInformation(), # キュメントに含まれる電話番号・電子メールアドレスを一部マスキング.
        document_filters.JSONDumper(dump_reason=True), # ドキュメントの破棄事由をJSON形式で付与して出力. JSON形式で出力するため、最後に実行する. # デバッグ用
        # （出力textがjsonになり、その後jsonに変換してresults.filtering.jsonlに保存するため、おかしくなる）
    ])

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


def __readlines(input_file: str):
    with open(input_file) as fp:
        return fp.readlines()


def filtering(input_dir: str, output_base: str):
    os.makedirs(output_base, exist_ok=True)

    file_lines = {input_file: __readlines(os.path.join(input_dir, input_file))
                  for input_file in os.listdir(input_dir) if input_file.endswith(".jsonl")}

    stats = []
    for input_file, json_lines in file_lines.items():
        input_file_prefix = os.path.splitext(os.path.basename(input_file))[0]
        output_base_for_input: str = os.path.join(output_base, input_file_prefix)
        os.makedirs(output_base_for_input, exist_ok=True)

        lines = process_json_lines(json_lines, output_base_for_input, stats)
        file_lines[input_file] = lines

    with open(os.path.join(output_base, "results.filtering.jsonl"), "w", encoding="utf8") as writer:
        for _, lines in file_lines.items():
            for line in lines:
                json.dump(line, writer, ensure_ascii=False)
                writer.write("\n")

    with open(os.path.join(output_base, "stats.filtering.jsonl"), "w", encoding="utf8") as writer:
        for stat in stats:
            json.dump(stat, writer, ensure_ascii=False)
            writer.write("\n")


def main():
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--input_dir', type=str,
                        help='The input directory containing documents to process', required=False, default="/home/ubuntu/ucllm_nedo_prod_kawagoshi/data_management/output/datasets")
    parser.add_argument('--output_dir', type=str,
                        help='The input file containing documents to process', required=False, default="/home/ubuntu/ucllm_nedo_prod_kawagoshi/data_management/output")
    args = parser.parse_args()

    #start = datetime.now()
    #output_base = os.path.join(args.output_dir, start.strftime("%Y%m%d%H%M%S"))
    output_base = os.path.join(args.output_dir, "filterd_documents")

    filtering(input_dir=args.input_dir, output_base=output_base)


if __name__ == "__main__":
    main()
